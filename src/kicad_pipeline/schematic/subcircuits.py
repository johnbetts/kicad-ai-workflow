"""Standard circuit template generators for the schematic builder.

Each function returns a :class:`SubcircuitResult` containing the
:class:`~kicad_pipeline.models.requirements.Component` and
:class:`~kicad_pipeline.models.requirements.Net` objects that make up the
subcircuit.  Callers can merge multiple results into a single
:class:`~kicad_pipeline.models.requirements.ProjectRequirements` before
passing them to the schematic builder.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from kicad_pipeline.models.requirements import (
    Component,
    Net,
    NetConnection,
    Pin,
    PinType,
)
from kicad_pipeline.requirements.component_db import ComponentDB, nearest_e_series_value

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubcircuitResult:
    """Output of a subcircuit generator.

    Attributes:
        components: Tuple of fully-specified components produced by the
            generator.
        nets: Tuple of nets that interconnect those components.
        description: Human-readable description of the subcircuit, suitable
            for BOM notes or schematic annotations.
    """

    components: tuple[Component, ...]
    nets: tuple[Net, ...]
    description: str


# ---------------------------------------------------------------------------
# Multi-instance instantiation
# ---------------------------------------------------------------------------


def _split_ref(ref: str) -> tuple[str, int]:
    """Split a ref like ``'R10'`` into ``('R', 10)``.

    Raises:
        ValueError: If *ref* has no trailing digits.
    """
    m = re.match(r"^([A-Za-z_]+)(\d+)$", ref)
    if m is None:
        raise ValueError(f"Cannot split ref '{ref}' into prefix + number")
    return m.group(1), int(m.group(2))


def instantiate_subcircuit(
    generator: Callable[..., SubcircuitResult],
    count: int,
    ref_start: dict[str, int],
    net_prefix: str,
    **kwargs: Any,
) -> tuple[SubcircuitResult, ...]:
    """Create *count* copies of a subcircuit with auto-numbered refs and nets.

    For each instance *i* (1-based), the generator is called with ref
    arguments derived from *ref_start* and a GPIO/signal net derived from
    *net_prefix*.

    The generator's first positional arguments must be reference designators
    (string parameters whose names start with ``ref_``).  For each such
    parameter, the ref is constructed as ``prefix + (start + i - 1)`` using
    the entries in *ref_start*.  The next positional argument after refs is
    assumed to be the signal net name (e.g. ``gpio_net``), set to
    ``'{net_prefix}_{i}'``.

    Args:
        generator: Subcircuit generator function (e.g. :func:`relay_driver`).
        count: Number of instances to create.
        ref_start: Mapping from ref prefix (e.g. ``'Q'``, ``'R'``, ``'K'``)
            to starting number for that prefix.
        net_prefix: Base name for the per-instance signal net (e.g.
            ``'RELAY'`` → ``'RELAY_1'``, ``'RELAY_2'``, ...).
        **kwargs: Additional keyword arguments forwarded to the generator.

    Returns:
        Tuple of :class:`SubcircuitResult`, one per instance.
    """
    import inspect

    sig = inspect.signature(generator)
    params = list(sig.parameters.keys())

    # Identify ref parameters (name starts with "ref_")
    ref_params = [p for p in params if p.startswith("ref_")]
    # The first non-ref_ parameter after the ref block is the signal net
    first_non_ref_idx = len(ref_params)
    signal_param = params[first_non_ref_idx] if first_non_ref_idx < len(params) else None

    results: list[SubcircuitResult] = []
    for i in range(1, count + 1):
        call_kwargs: dict[str, str | None] = {}
        for rp in ref_params:
            # Extract the prefix from the param name heuristic:
            # ref_q -> Q, ref_r_base -> R, ref_d -> D, ref_k -> K, ref_j -> J
            suffix = rp[4:]  # strip "ref_"
            prefix = suffix.split("_")[0].upper()
            start = ref_start.get(prefix)
            if start is not None:
                call_kwargs[rp] = f"{prefix}{start + i - 1}"
            else:
                # Might be None (e.g. ref_j can be None)
                call_kwargs[rp] = None

        if signal_param is not None and signal_param not in kwargs:
            call_kwargs[signal_param] = f"{net_prefix}_{i}"

        call_kwargs.update(kwargs)
        results.append(generator(**call_kwargs))

    return tuple(results)


def merge_subcircuit_results(
    results: tuple[SubcircuitResult, ...] | list[SubcircuitResult],
) -> SubcircuitResult:
    """Merge multiple :class:`SubcircuitResult` into a single flat result.

    Components are concatenated. Nets with the same name (e.g. GND, +5V)
    have their connections merged into one net entry.

    Args:
        results: Subcircuit results to merge.

    Returns:
        A single :class:`SubcircuitResult` combining all inputs.
    """
    all_components: list[Component] = []
    net_map: dict[str, list[NetConnection]] = {}
    descriptions: list[str] = []

    for r in results:
        all_components.extend(r.components)
        descriptions.append(r.description)
        for net in r.nets:
            net_map.setdefault(net.name, []).extend(net.connections)

    merged_nets = tuple(
        Net(name=name, connections=tuple(conns))
        for name, conns in net_map.items()
    )

    return SubcircuitResult(
        components=tuple(all_components),
        nets=merged_nets,
        description="; ".join(descriptions),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nearest_e24(value: float) -> float:
    """Return the nearest E24 standard value to *value*.

    Falls back to *value* unchanged if the E-series file cannot be loaded
    (e.g. during unit tests without the data directory).

    Args:
        value: Target value (units are agnostic — ohms, farads, etc.).

    Returns:
        Nearest E24 standard value.
    """
    try:
        return nearest_e_series_value(value, series="E24")
    except Exception:
        log.debug("E24 lookup failed for %g; using raw value", value)
        return value


def _format_resistance(ohms: float) -> str:
    """Format a resistance in ohms to a human-readable string.

    Args:
        ohms: Resistance in ohms.

    Returns:
        Formatted string such as ``'10k'``, ``'4.7k'``, ``'120R'``.
    """
    if ohms >= 1_000_000:
        v = ohms / 1_000_000
        return f"{v:g}M"
    if ohms >= 1_000:
        v = ohms / 1_000
        return f"{v:g}k"
    return f"{ohms:g}R"


def _format_capacitance(value_uf: float) -> str:
    """Format a capacitance in µF to a human-readable string.

    Args:
        value_uf: Capacitance in microfarads.

    Returns:
        Formatted string such as ``'100nF'``, ``'10uF'``, ``'1uF'``.
    """
    if value_uf >= 1.0:
        return f"{value_uf:g}uF"
    if value_uf >= 1e-3:
        return f"{value_uf * 1_000:g}nF"
    return f"{value_uf * 1_000_000:g}pF"


def _resistor_component(
    ref: str,
    ohms: float,
    package: str,
    pin1_net: str,
    pin2_net: str,
    db: ComponentDB | None,
) -> Component:
    """Build a :class:`Component` for a resistor.

    Args:
        ref: Reference designator, e.g. ``'R1'``.
        ohms: Resistance in ohms.
        package: Footprint package size string, e.g. ``'0805'``.
        pin1_net: Net name connected to pin 1.
        pin2_net: Net name connected to pin 2.
        db: Optional component database for LCSC number lookup.

    Returns:
        A frozen :class:`Component` representing the resistor.
    """
    value_str = _format_resistance(ohms)
    lcsc: str | None = None
    if db is not None:
        part = db.find_resistor(ohms, package=package)
        if part is not None:
            lcsc = part.lcsc
            value_str = part.value
    return Component(
        ref=ref,
        value=value_str,
        footprint=f"R_{package}",
        lcsc=lcsc,
        description=f"Resistor {value_str} {package}",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, net=pin1_net),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, net=pin2_net),
        ),
    )


def _capacitor_component(
    ref: str,
    value_uf: float,
    package: str,
    pin1_net: str,
    pin2_net: str,
    db: ComponentDB | None,
) -> Component:
    """Build a :class:`Component` for a capacitor.

    Args:
        ref: Reference designator, e.g. ``'C1'``.
        value_uf: Capacitance in microfarads.
        package: Footprint package size string.
        pin1_net: Net name connected to pin 1 (positive / VCC side).
        pin2_net: Net name connected to pin 2 (GND side).
        db: Optional component database for LCSC number lookup.

    Returns:
        A frozen :class:`Component` representing the capacitor.
    """
    value_str = _format_capacitance(value_uf)
    lcsc: str | None = None
    if db is not None:
        part = db.find_capacitor(value_uf, package=package)
        if part is not None:
            lcsc = part.lcsc
            value_str = part.value
    return Component(
        ref=ref,
        value=value_str,
        footprint=f"C_{package}",
        lcsc=lcsc,
        description=f"Capacitor {value_str} {package}",
        pins=(
            Pin(number="1", name="+", pin_type=PinType.PASSIVE, net=pin1_net),
            Pin(number="2", name="-", pin_type=PinType.PASSIVE, net=pin2_net),
        ),
    )


# ---------------------------------------------------------------------------
# Public subcircuit generators
# ---------------------------------------------------------------------------


def voltage_divider(
    ref_top: str,
    ref_bot: str,
    vin_net: str,
    vout_net: str,
    gnd_net: str = "GND",
    r_top_ohms: float = 10000.0,
    r_bot_ohms: float = 10000.0,
    package: str = "0805",
    db: ComponentDB | None = None,
) -> SubcircuitResult:
    """Generate a resistive voltage divider subcircuit.

    Topology::

        VIN ─── R_top ─── VOUT ─── R_bot ─── GND

    The actual output voltage is::

        Vout = Vin * r_bot / (r_top + r_bot)

    E24 standard values are used when a :class:`ComponentDB` is provided.

    Args:
        ref_top: Reference designator for the top (series) resistor.
        ref_bot: Reference designator for the bottom (shunt) resistor.
        vin_net: Net name at the divider input (top of R_top).
        vout_net: Net name at the divider mid-point / output.
        gnd_net: Net name at the bottom of R_bot (default ``'GND'``).
        r_top_ohms: Top resistor value in ohms (E24-snapped if *db* given).
        r_bot_ohms: Bottom resistor value in ohms (E24-snapped if *db* given).
        package: Resistor footprint package (default ``'0805'``).
        db: Optional JLCPCB component database for exact part lookup.

    Returns:
        :class:`SubcircuitResult` with two resistor components and three nets.
    """
    r_top_actual = _nearest_e24(r_top_ohms) if db is None else r_top_ohms
    r_bot_actual = _nearest_e24(r_bot_ohms) if db is None else r_bot_ohms

    vout_ratio = r_bot_actual / (r_top_actual + r_bot_actual)
    log.debug(
        "voltage_divider: Vout = Vin x %.4f  (R_top=%g ohm, R_bot=%g ohm)",
        vout_ratio,
        r_top_actual,
        r_bot_actual,
    )

    r_top_comp = _resistor_component(ref_top, r_top_actual, package, vin_net, vout_net, db)
    r_bot_comp = _resistor_component(ref_bot, r_bot_actual, package, vout_net, gnd_net, db)

    net_vin = Net(
        name=vin_net,
        connections=(NetConnection(ref=ref_top, pin="1"),),
    )
    net_vout = Net(
        name=vout_net,
        connections=(
            NetConnection(ref=ref_top, pin="2"),
            NetConnection(ref=ref_bot, pin="1"),
        ),
    )
    net_gnd = Net(
        name=gnd_net,
        connections=(NetConnection(ref=ref_bot, pin="2"),),
    )

    desc = (
        f"Voltage divider: {_format_resistance(r_top_actual)} / "
        f"{_format_resistance(r_bot_actual)}, "
        f"Vout = Vin x {vout_ratio:.4f}"
    )
    return SubcircuitResult(
        components=(r_top_comp, r_bot_comp),
        nets=(net_vin, net_vout, net_gnd),
        description=desc,
    )


def decoupling_cap(
    ref: str,
    vcc_net: str,
    gnd_net: str = "GND",
    value_uf: float = 0.1,
    package: str = "0805",
    db: ComponentDB | None = None,
) -> SubcircuitResult:
    """Generate a single decoupling capacitor between VCC and GND.

    Topology::

        VCC ─── C ─── GND

    Args:
        ref: Reference designator for the capacitor.
        vcc_net: Power supply net name (positive terminal of the cap).
        gnd_net: Ground net name (negative terminal of the cap).
        value_uf: Capacitor value in microfarads (default ``0.1`` µF = 100 nF).
        package: Capacitor footprint package (default ``'0805'``).
        db: Optional JLCPCB component database for exact part lookup.

    Returns:
        :class:`SubcircuitResult` with one capacitor component and two nets.
    """
    cap = _capacitor_component(ref, value_uf, package, vcc_net, gnd_net, db)
    net_vcc = Net(
        name=vcc_net,
        connections=(NetConnection(ref=ref, pin="1"),),
    )
    net_gnd = Net(
        name=gnd_net,
        connections=(NetConnection(ref=ref, pin="2"),),
    )
    return SubcircuitResult(
        components=(cap,),
        nets=(net_vcc, net_gnd),
        description=f"Decoupling cap {_format_capacitance(value_uf)} {package} on {vcc_net}",
    )


def led_drive(
    ref_led: str,
    ref_r: str,
    anode_net: str,
    gpio_net: str,
    gnd_net: str = "GND",
    vcc_v: float = 3.3,
    vf_v: float = 2.1,
    target_ma: float = 10.0,
    package: str = "0805",
    db: ComponentDB | None = None,
) -> SubcircuitResult:
    """Generate an LED + current-limiting resistor subcircuit.

    Topology (active-low GPIO drive)::

        VCC ─── LED(anode→cathode) ─── R_limit ─── GPIO

    The current-limit resistor value is::

        R = (Vcc - Vf) / I_target

    rounded to the nearest E24 standard value.

    Args:
        ref_led: Reference designator for the LED.
        ref_r: Reference designator for the current-limiting resistor.
        anode_net: VCC net connected to the LED anode.
        gpio_net: GPIO control net on the resistor side (active-low drive).
        gnd_net: Ground net (unused in this topology but retained for
            caller convenience — not connected here).
        vcc_v: Supply voltage in volts (default ``3.3``).
        vf_v: LED forward voltage in volts (default ``2.1``).
        target_ma: Target LED forward current in milliamps (default ``10.0``).
        package: Component footprint package (default ``'0805'``).
        db: Optional JLCPCB component database.

    Returns:
        :class:`SubcircuitResult` with LED + resistor components and nets.
    """
    r_exact = (vcc_v - vf_v) / (target_ma / 1000.0)
    r_actual = _nearest_e24(r_exact)
    log.debug(
        "led_drive: R = (%.2f - %.2f) / %.4f A = %.1f Ω → E24 %.1f Ω",
        vcc_v,
        vf_v,
        target_ma / 1000.0,
        r_exact,
        r_actual,
    )

    # Internal net between LED cathode and resistor
    internal_net = f"{ref_led}_K"

    lcsc_led: str | None = None
    vf_actual = vf_v
    if db is not None:
        led_part = db.find_led(package=package)
        if led_part is not None:
            lcsc_led = led_part.lcsc
            if led_part.vf is not None:
                vf_actual = led_part.vf

    led_comp = Component(
        ref=ref_led,
        value=f"LED_Vf{vf_actual:.1f}V",
        footprint=f"LED_{package}",
        lcsc=lcsc_led,
        description=f"LED Vf={vf_actual:.1f}V {package}",
        pins=(
            Pin(number="A", name="A", pin_type=PinType.PASSIVE, net=anode_net),
            Pin(number="K", name="K", pin_type=PinType.PASSIVE, net=internal_net),
        ),
    )
    r_comp = _resistor_component(ref_r, r_actual, package, internal_net, gpio_net, db)

    net_anode = Net(
        name=anode_net,
        connections=(NetConnection(ref=ref_led, pin="A"),),
    )
    net_internal = Net(
        name=internal_net,
        connections=(
            NetConnection(ref=ref_led, pin="K"),
            NetConnection(ref=ref_r, pin="1"),
        ),
    )
    net_gpio = Net(
        name=gpio_net,
        connections=(NetConnection(ref=ref_r, pin="2"),),
    )

    return SubcircuitResult(
        components=(led_comp, r_comp),
        nets=(net_anode, net_internal, net_gpio),
        description=(
            f"LED drive: {ref_led} (Vf={vf_actual:.1f}V) + "
            f"{ref_r} ({_format_resistance(r_actual)}), "
            f"I={target_ma:.1f}mA"
        ),
    )


def relay_driver(
    ref_q: str,
    ref_r_base: str,
    ref_d: str,
    ref_k: str,
    ref_j: str | None,
    gpio_net: str,
    vcc_net: str = "+5V",
    gnd_net: str = "GND",
    relay_type: str = "SPDT",
    db: ComponentDB | None = None,
) -> SubcircuitResult:
    """Generate an NPN relay driver circuit with flyback protection.

    Topology::

        GPIO ─── R_base(1k) ─── Q_base
        Q_collector ─── K_COIL- ; K_COIL+ ─── VCC
        Q_emitter ─── GND
        Flyback diode: anode → Q_collector, cathode → VCC
        K_COM/NO/NC ─── J screw terminal (optional)

    Args:
        ref_q: Reference designator for the NPN transistor (e.g. ``'Q1'``).
        ref_r_base: Reference designator for the base resistor (e.g. ``'R10'``).
        ref_d: Reference designator for the flyback diode (e.g. ``'D6'``).
        ref_k: Reference designator for the relay (e.g. ``'K1'``).
        ref_j: Reference designator for the output screw terminal, or ``None``
            to omit the terminal.
        gpio_net: GPIO control signal net name.
        vcc_net: Supply net for relay coil and diode cathode (default ``'+5V'``).
        gnd_net: Ground net (default ``'GND'``).
        relay_type: ``'SPDT'`` (3 contact pins) or ``'SPST'`` (2 contact pins).
        db: Optional component database for LCSC lookup.

    Returns:
        :class:`SubcircuitResult` with transistor, base resistor, flyback diode,
        relay, and optional screw terminal components plus their interconnecting nets.
    """
    base_net = f"{ref_q}_BASE"
    coil_net = f"{ref_k}_COIL"

    r_base_comp = _resistor_component(
        ref_r_base, 1000.0, "0402", gpio_net, base_net, db,
    )

    q_comp = Component(
        ref=ref_q,
        value="BC817",
        footprint="SOT-23",
        description=f"Relay driver NPN transistor for {ref_k}",
        pins=(
            Pin(number="1", name="B", pin_type=PinType.INPUT, net=base_net),
            Pin(number="2", name="C", pin_type=PinType.PASSIVE, net=coil_net),
            Pin(number="3", name="E", pin_type=PinType.PASSIVE, net=gnd_net),
        ),
    )

    diode_comp = Component(
        ref=ref_d,
        value="1N4148",
        footprint="SOD-123",
        description=f"Flyback diode for {ref_k}",
        pins=(
            Pin(number="1", name="A", pin_type=PinType.PASSIVE, net=coil_net),
            Pin(number="2", name="K", pin_type=PinType.PASSIVE, net=vcc_net),
        ),
    )

    is_spdt = relay_type.upper() == "SPDT"
    com_net = f"{ref_k}_COM"
    no_net = f"{ref_k}_NO"
    nc_net = f"{ref_k}_NC" if is_spdt else None

    relay_pins: list[Pin] = [
        Pin(number="1", name="COIL+", pin_type=PinType.PASSIVE, net=vcc_net),
        Pin(number="2", name="COIL-", pin_type=PinType.PASSIVE, net=coil_net),
        Pin(number="3", name="COM", pin_type=PinType.PASSIVE, net=com_net),
        Pin(number="4", name="NO", pin_type=PinType.PASSIVE, net=no_net),
    ]
    if is_spdt:
        relay_pins.append(
            Pin(number="5", name="NC", pin_type=PinType.PASSIVE, net=nc_net),
        )

    relay_footprint = "Relay_SPDT_SANYOU_SRD" if is_spdt else "Relay_SPST"
    relay_comp = Component(
        ref=ref_k,
        value="SRD-05VDC-SL-C" if is_spdt else "SRD-05VDC-SL-A",
        footprint=relay_footprint,
        description=f"Relay {relay_type}",
        pins=tuple(relay_pins),
    )

    components: list[Component] = [r_base_comp, q_comp, diode_comp, relay_comp]
    nets: list[Net] = []

    # GPIO net
    nets.append(Net(
        name=gpio_net,
        connections=(NetConnection(ref=ref_r_base, pin="1"),),
    ))

    # Base net
    nets.append(Net(
        name=base_net,
        connections=(
            NetConnection(ref=ref_r_base, pin="2"),
            NetConnection(ref=ref_q, pin="1"),
        ),
    ))

    # Coil net
    coil_conns: list[NetConnection] = [
        NetConnection(ref=ref_q, pin="2"),
        NetConnection(ref=ref_d, pin="1"),
        NetConnection(ref=ref_k, pin="2"),
    ]
    nets.append(Net(name=coil_net, connections=tuple(coil_conns)))

    # VCC net
    nets.append(Net(
        name=vcc_net,
        connections=(
            NetConnection(ref=ref_d, pin="2"),
            NetConnection(ref=ref_k, pin="1"),
        ),
    ))

    # GND net
    nets.append(Net(
        name=gnd_net,
        connections=(NetConnection(ref=ref_q, pin="3"),),
    ))

    # Contact nets + optional screw terminal
    terminal_pins: list[Pin] = []
    com_conns: list[NetConnection] = [NetConnection(ref=ref_k, pin="3")]
    no_conns: list[NetConnection] = [NetConnection(ref=ref_k, pin="4")]
    nc_conns: list[NetConnection] = []
    if is_spdt:
        nc_conns.append(NetConnection(ref=ref_k, pin="5"))

    if ref_j is not None:
        terminal_pins.append(
            Pin(number="1", name="NO", pin_type=PinType.PASSIVE, net=no_net),
        )
        terminal_pins.append(
            Pin(number="2", name="COM", pin_type=PinType.PASSIVE, net=com_net),
        )
        no_conns.append(NetConnection(ref=ref_j, pin="1"))
        com_conns.append(NetConnection(ref=ref_j, pin="2"))

        if is_spdt:
            terminal_pins.append(
                Pin(number="3", name="NC", pin_type=PinType.PASSIVE, net=nc_net),
            )
            nc_conns.append(NetConnection(ref=ref_j, pin="3"))

        terminal_footprint = (
            "TerminalBlock_01x03_P5.08mm" if is_spdt
            else "TerminalBlock_01x02_P5.08mm"
        )
        terminal_comp = Component(
            ref=ref_j,
            value=f"Screw_Terminal_01x0{'3' if is_spdt else '2'}",
            footprint=terminal_footprint,
            description=f"Relay {ref_k} output connector",
            pins=tuple(terminal_pins),
        )
        components.append(terminal_comp)

    nets.append(Net(name=com_net, connections=tuple(com_conns)))
    nets.append(Net(name=no_net, connections=tuple(no_conns)))
    if is_spdt and nc_net is not None:
        nets.append(Net(name=nc_net, connections=tuple(nc_conns)))

    contact_desc = "COM/NO/NC" if is_spdt else "COM/NO"
    terminal_desc = f" + {ref_j} terminal" if ref_j else ""
    desc = (
        f"Relay driver: {ref_q} (BC817) + {ref_r_base} (1k) + "
        f"{ref_d} (1N4148 flyback) + {ref_k} ({relay_type} {contact_desc})"
        f"{terminal_desc}"
    )
    return SubcircuitResult(
        components=tuple(components),
        nets=tuple(nets),
        description=desc,
    )


def npn_buzzer_drive(
    ref_q: str,
    ref_r_base: str,
    ref_d: str,
    buzzer_net: str,
    gpio_net: str,
    vcc_net: str,
    gnd_net: str = "GND",
) -> SubcircuitResult:
    """Generate an NPN transistor buzzer drive circuit.

    Topology::

        GPIO ─── R_base ─── Q_base
        Q_collector ─── Buzzer(+) ─── VCC
        Q_emitter ─── GND
        Flyback diode: anode → Q_collector, cathode → VCC

    Args:
        ref_q: Reference designator for the NPN transistor.
        ref_r_base: Reference designator for the base resistor.
        ref_d: Reference designator for the flyback diode.
        buzzer_net: Net connecting Q_collector to the buzzer positive terminal.
        gpio_net: GPIO output net driving the base resistor.
        vcc_net: Supply net connected to the buzzer other terminal and diode
            cathode.
        gnd_net: Ground net connected to the transistor emitter.

    Returns:
        :class:`SubcircuitResult` with transistor, base resistor, and flyback
        diode components plus their interconnecting nets.
    """
    # Internal net: base resistor output → transistor base
    base_net = f"{ref_q}_B"

    q_comp = Component(
        ref=ref_q,
        value="NPN_BJT",
        footprint="SOT-23",
        description="NPN transistor buzzer driver",
        pins=(
            Pin(number="B", name="B", pin_type=PinType.INPUT, net=base_net),
            Pin(number="C", name="C", pin_type=PinType.PASSIVE, net=buzzer_net),
            Pin(number="E", name="E", pin_type=PinType.PASSIVE, net=gnd_net),
        ),
    )
    r_base_comp = _resistor_component(
        ref_r_base, 1000.0, "0805", gpio_net, base_net, None
    )
    diode_comp = Component(
        ref=ref_d,
        value="1N4148",
        footprint="SOD-123",
        description="Flyback diode for buzzer inductive load protection",
        pins=(
            Pin(number="A", name="A", pin_type=PinType.PASSIVE, net=buzzer_net),
            Pin(number="K", name="K", pin_type=PinType.PASSIVE, net=vcc_net),
        ),
    )

    net_gpio = Net(
        name=gpio_net,
        connections=(NetConnection(ref=ref_r_base, pin="1"),),
    )
    net_base = Net(
        name=base_net,
        connections=(
            NetConnection(ref=ref_r_base, pin="2"),
            NetConnection(ref=ref_q, pin="B"),
        ),
    )
    net_buzzer = Net(
        name=buzzer_net,
        connections=(
            NetConnection(ref=ref_q, pin="C"),
            NetConnection(ref=ref_d, pin="A"),
        ),
    )
    net_vcc = Net(
        name=vcc_net,
        connections=(NetConnection(ref=ref_d, pin="K"),),
    )
    net_gnd = Net(
        name=gnd_net,
        connections=(NetConnection(ref=ref_q, pin="E"),),
    )

    return SubcircuitResult(
        components=(q_comp, r_base_comp, diode_comp),
        nets=(net_gpio, net_base, net_buzzer, net_vcc, net_gnd),
        description=(
            f"NPN buzzer drive: {ref_q} + base R {ref_r_base} + flyback {ref_d}"
        ),
    )


def ldo_regulator(
    ref_ldo: str,
    ref_cin: str,
    ref_cout: str,
    vin_net: str,
    vout_net: str,
    gnd_net: str = "GND",
    vout_v: float = 3.3,
    iout_ma: float = 600.0,
) -> SubcircuitResult:
    """Generate an LDO voltage regulator with input and output decoupling caps.

    Topology::

        VIN ─── C_in(10µF) ─── GND
        VIN ─── LDO_IN ; LDO_OUT ─── VOUT
        VOUT ─── C_out(10µF) ─── GND
        LDO_GND ─── GND

    Args:
        ref_ldo: Reference designator for the LDO IC.
        ref_cin: Reference designator for the input decoupling capacitor.
        ref_cout: Reference designator for the output decoupling capacitor.
        vin_net: Input power net.
        vout_net: Output regulated net.
        gnd_net: Ground net.
        vout_v: Required output voltage in volts (default ``3.3``).
        iout_ma: Required output current in milliamps (default ``600.0``).

    Returns:
        :class:`SubcircuitResult` with LDO + two caps and their nets.
    """
    ldo_comp = Component(
        ref=ref_ldo,
        value=f"LDO_{vout_v:.1f}V",
        footprint="SOT-223-3",
        description=f"LDO regulator {vout_v:.1f}V {iout_ma:.0f}mA",
        pins=(
            Pin(number="IN", name="IN", pin_type=PinType.POWER_IN, net=vin_net),
            Pin(number="OUT", name="OUT", pin_type=PinType.POWER_OUT, net=vout_net),
            Pin(number="GND", name="GND", pin_type=PinType.POWER_IN, net=gnd_net),
        ),
    )
    cin_comp = _capacitor_component(ref_cin, 10.0, "0805", vin_net, gnd_net, None)
    cout_comp = _capacitor_component(ref_cout, 10.0, "0805", vout_net, gnd_net, None)

    net_vin = Net(
        name=vin_net,
        connections=(
            NetConnection(ref=ref_ldo, pin="IN"),
            NetConnection(ref=ref_cin, pin="1"),
        ),
    )
    net_vout = Net(
        name=vout_net,
        connections=(
            NetConnection(ref=ref_ldo, pin="OUT"),
            NetConnection(ref=ref_cout, pin="1"),
        ),
    )
    net_gnd = Net(
        name=gnd_net,
        connections=(
            NetConnection(ref=ref_ldo, pin="GND"),
            NetConnection(ref=ref_cin, pin="2"),
            NetConnection(ref=ref_cout, pin="2"),
        ),
    )

    return SubcircuitResult(
        components=(ldo_comp, cin_comp, cout_comp),
        nets=(net_vin, net_vout, net_gnd),
        description=(
            f"LDO regulator {ref_ldo}: {vin_net}→{vout_net} "
            f"{vout_v:.1f}V/{iout_ma:.0f}mA + decoupling caps"
        ),
    )


def dip_switch_address(
    switch_ref: str,
    bit_count: int = 4,
    addr_net: str = "ADDR",
    target_nets: tuple[str, ...] = (),
    series_resistance: float = 10_000.0,
    package: str = "0805",
    db: ComponentDB | None = None,
) -> SubcircuitResult:
    """Generate a DIP switch address selector with series resistor protection.

    Each switch output goes through a series resistor before connecting to
    the target net, preventing short circuits when multiple switches are
    activated simultaneously.

    Topology (per bit)::

        ADDR_n --- SW_n --- R_n --- TARGET_n

    Args:
        switch_ref: Reference designator for the DIP switch (e.g. "SW1").
        bit_count: Number of switch positions (default 4).
        addr_net: Base name for address nets (default "ADDR").
        target_nets: Net names each switch bridges to. If empty,
            auto-generated as "ADDR_0", "ADDR_1", etc.
        series_resistance: Series resistor value in ohms (default 10k).
        package: Resistor footprint package (default "0805").
        db: Optional component database for LCSC lookup.

    Returns:
        :class:`SubcircuitResult` with DIP switch + series resistors.
    """
    if not target_nets:
        target_nets = tuple(f"{addr_net}_{i}" for i in range(bit_count))

    if len(target_nets) < bit_count:
        target_nets = target_nets + tuple(
            f"{addr_net}_{i}" for i in range(len(target_nets), bit_count)
        )

    components: list[Component] = []
    nets: list[Net] = []

    # DIP switch component: 2 * bit_count pins
    # Left column: pins 1..bit_count (inputs)
    # Right column: pins bit_count+1..2*bit_count (outputs, reversed order)
    sw_pins: list[Pin] = []
    for i in range(bit_count):
        input_net = f"{addr_net}_SW{i}_IN"
        sw_pins.append(
            Pin(
                number=str(i + 1),
                name=f"IN{i + 1}",
                pin_type=PinType.PASSIVE,
                net=input_net,
            )
        )
    for i in range(bit_count):
        output_net = f"{addr_net}_SW{bit_count - 1 - i}_OUT"
        sw_pins.append(
            Pin(
                number=str(bit_count + i + 1),
                name=f"OUT{bit_count - i}",
                pin_type=PinType.PASSIVE,
                net=output_net,
            )
        )

    sw_comp = Component(
        ref=switch_ref,
        value=f"DIPx{bit_count:02d}",
        footprint=f"SW_DIP_SPSTx{bit_count:02d}",
        description=f"DIP switch {bit_count}-position address selector",
        pins=tuple(sw_pins),
    )
    components.append(sw_comp)

    # Series resistors: one per bit
    for i in range(bit_count):
        r_ref = f"R_{switch_ref}_{i + 1}"
        input_net = f"{addr_net}_SW{i}_IN"
        output_net = f"{addr_net}_SW{i}_OUT"

        r_comp = _resistor_component(
            r_ref, series_resistance, package,
            output_net, target_nets[i], db,
        )
        components.append(r_comp)

        # Net: switch input to source
        nets.append(Net(
            name=input_net,
            connections=(NetConnection(ref=switch_ref, pin=str(i + 1)),),
        ))

        # Net: switch output through resistor
        nets.append(Net(
            name=output_net,
            connections=(
                NetConnection(ref=switch_ref, pin=str(bit_count + (bit_count - 1 - i) + 1)),
                NetConnection(ref=r_ref, pin="1"),
            ),
        ))

        # Net: resistor output to target
        nets.append(Net(
            name=target_nets[i],
            connections=(NetConnection(ref=r_ref, pin="2"),),
        ))

    desc = (
        f"DIP switch {switch_ref}: {bit_count}-bit address selector "
        f"with {_format_resistance(series_resistance)} series protection. "
        f"WARNING: activate only ONE switch at a time to avoid contention."
    )
    return SubcircuitResult(
        components=tuple(components),
        nets=tuple(nets),
        description=desc,
    )


def usb_c_input(
    ref_conn: str,
    ref_cc1: str,
    ref_cc2: str,
    vbus_net: str = "VBUS",
    gnd_net: str = "GND",
    dp_net: str = "USB_DP",
    dm_net: str = "USB_DM",
    cc_net1: str = "USB_CC1",
    cc_net2: str = "USB_CC2",
) -> SubcircuitResult:
    """Generate a USB-C input connector with CC pull-down resistors.

    For a USB-C power sink, CC1 and CC2 each require a 5.1 kΩ pull-down
    resistor to GND so the host recognises the device as a 5 V consumer.

    Topology::

        J_USB_C:
          VBUS → VBUS net
          GND  → GND net
          D+   → USB_DP net
          D-   → USB_DM net
          CC1  → USB_CC1 net ─── R_CC1(5.1k) ─── GND
          CC2  → USB_CC2 net ─── R_CC2(5.1k) ─── GND

    Args:
        ref_conn: Reference designator for the USB-C connector.
        ref_cc1: Reference designator for the CC1 pull-down resistor.
        ref_cc2: Reference designator for the CC2 pull-down resistor.
        vbus_net: VBUS power net name.
        gnd_net: Ground net name.
        dp_net: USB D+ net name.
        dm_net: USB D- net name.
        cc_net1: CC1 configuration channel net name.
        cc_net2: CC2 configuration channel net name.

    Returns:
        :class:`SubcircuitResult` with connector + two CC resistors and nets.
    """
    conn_comp = Component(
        ref=ref_conn,
        value="USB_C_Receptacle",
        footprint="USB_C_Receptacle_HRO_TYPE-C-31-M-12",
        description="USB-C receptacle connector",
        pins=(
            Pin(number="A1", name="GND", pin_type=PinType.POWER_IN, net=gnd_net),
            Pin(number="A4", name="VBUS", pin_type=PinType.POWER_IN, net=vbus_net),
            Pin(number="A5", name="CC1", pin_type=PinType.BIDIRECTIONAL, net=cc_net1),
            Pin(number="A6", name="DP1", pin_type=PinType.BIDIRECTIONAL, net=dp_net),
            Pin(number="A7", name="DM1", pin_type=PinType.BIDIRECTIONAL, net=dm_net),
            Pin(number="B5", name="CC2", pin_type=PinType.BIDIRECTIONAL, net=cc_net2),
            Pin(number="B6", name="DP2", pin_type=PinType.BIDIRECTIONAL, net=dp_net),
            Pin(number="B7", name="DM2", pin_type=PinType.BIDIRECTIONAL, net=dm_net),
        ),
    )
    # 5.1 kΩ CC pull-down resistors (USB-C power sink identification)
    r_cc1 = _resistor_component(ref_cc1, 5100.0, "0402", cc_net1, gnd_net, None)
    r_cc2 = _resistor_component(ref_cc2, 5100.0, "0402", cc_net2, gnd_net, None)

    net_vbus = Net(
        name=vbus_net,
        connections=(NetConnection(ref=ref_conn, pin="A4"),),
    )
    net_gnd = Net(
        name=gnd_net,
        connections=(
            NetConnection(ref=ref_conn, pin="A1"),
            NetConnection(ref=ref_cc1, pin="2"),
            NetConnection(ref=ref_cc2, pin="2"),
        ),
    )
    net_dp = Net(
        name=dp_net,
        connections=(
            NetConnection(ref=ref_conn, pin="A6"),
            NetConnection(ref=ref_conn, pin="B6"),
        ),
    )
    net_dm = Net(
        name=dm_net,
        connections=(
            NetConnection(ref=ref_conn, pin="A7"),
            NetConnection(ref=ref_conn, pin="B7"),
        ),
    )
    net_cc1 = Net(
        name=cc_net1,
        connections=(
            NetConnection(ref=ref_conn, pin="A5"),
            NetConnection(ref=ref_cc1, pin="1"),
        ),
    )
    net_cc2 = Net(
        name=cc_net2,
        connections=(
            NetConnection(ref=ref_conn, pin="B5"),
            NetConnection(ref=ref_cc2, pin="1"),
        ),
    )

    return SubcircuitResult(
        components=(conn_comp, r_cc1, r_cc2),
        nets=(net_vbus, net_gnd, net_dp, net_dm, net_cc1, net_cc2),
        description=(
            f"USB-C input: {ref_conn} with 5.1k CC pull-downs "
            f"{ref_cc1}/{ref_cc2} for power sink identification"
        ),
    )


# ---------------------------------------------------------------------------
# Voltage calculation utility (public helper)
# ---------------------------------------------------------------------------


def voltage_divider_vout(vin: float, r_top: float, r_bot: float) -> float:
    """Calculate the output voltage of a resistive divider.

    Args:
        vin: Input voltage in volts.
        r_top: Top (series) resistor value in ohms.
        r_bot: Bottom (shunt) resistor value in ohms.

    Returns:
        Output voltage at the divider mid-point in volts.

    Raises:
        ValueError: If both resistors sum to zero (degenerate divider).
    """
    total = r_top + r_bot
    if total == 0.0:
        raise ValueError("r_top + r_bot must be non-zero")
    return vin * r_bot / total


def led_limit_resistor(vcc_v: float, vf_v: float, target_ma: float) -> float:
    """Calculate the nominal LED current-limit resistor value.

    Args:
        vcc_v: Supply voltage in volts.
        vf_v: LED forward voltage in volts.
        target_ma: Target forward current in milliamps.

    Returns:
        Nominal resistance in ohms (before E24 rounding).

    Raises:
        ValueError: If *target_ma* is non-positive.
    """
    if target_ma <= 0:
        raise ValueError(f"target_ma must be positive, got {target_ma}")
    return (vcc_v - vf_v) / (target_ma / 1000.0)

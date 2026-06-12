"""Deterministic netlist lint — catches DOA-board defect classes.

Born from the 2026-06-12 functional review of nl-s-3c, where the
requirements netlist carried board-killing errors no gate looked at:

* The W5500 was modeled with a FICTITIOUS 18-pin map on an LQFP-48
  footprint — 30 pads (including power pins) floated, nets landed on
  wrong physical pins. -> :func:`check_pin_pad_coverage`.
* The 24V input TVS was wired anode-to-VIN / cathode-to-GND — a
  forward diode shorting the input. -> :func:`check_diode_polarity`.

Both are pure netlist+footprint facts, checkable without datasheets.
(The third defect class found that day — pin NAMES shifted against
the datasheet, ADS1115/AP6320x — is NOT decidable offline; that is
the pcb-pinout-verify datasheet flow's job.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.placement_v2.ir import Severity, Violation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kicad_pipeline.models.pcb import Footprint
    from kicad_pipeline.models.requirements import ProjectRequirements

#: A part whose declared pins cover less than this fraction of its
#: footprint's pads is a STUB pin map (W5500: 18/48 = 37%), not a part
#: with some unused IOs (ESP32 module: ~73% covered is normal).
MIN_PAD_COVERAGE = 0.5

#: Value/footprint tokens marking LEDs — anode-on-rail is NORMAL for an
#: indicator (rail -> LED -> resistor -> GND), never a reversed clamp.
_LED_TOKENS = ("led", "ws2812")


def _is_power_rail(net: str) -> bool:
    return net.startswith("+") or net in ("VIN", "VCC", "VDD")


def _is_gnd(net: str) -> bool:
    return "GND" in net.upper()


def check_pin_pad_coverage(
    requirements: ProjectRequirements,
    footprints: Mapping[str, Footprint],
) -> tuple[Violation, ...]:
    """Declared pins vs physical pads, per component.

    CRITICAL when a declared pin number has no pad (its net can never
    reach copper), and CRITICAL when coverage of the footprint's
    numbered pads falls below :data:`MIN_PAD_COVERAGE` — a stub pin
    map silently floats supply pins.
    """
    out: list[Violation] = []
    for comp in requirements.components:
        fp = footprints.get(comp.ref)
        if fp is None or not comp.pins:
            continue
        pad_numbers = {p.number for p in fp.pads if p.number}
        # Only NUMERIC pads count toward coverage: shield/EP pads
        # ("SH1", "EP") are legitimately unmodeled.
        numeric_pads = {p for p in pad_numbers if p.isdigit()}
        declared = {p.number for p in comp.pins}

        ghost = sorted(declared - pad_numbers)
        if ghost:
            out.append(Violation(
                constraint=f"pin_pad_coverage({comp.ref})",
                refs=(comp.ref,),
                severity=Severity.CRITICAL,
                measured=float(len(ghost)), limit=0.0,
                message=(
                    f"{comp.ref} ({comp.value}) declares pins with no pad on "
                    f"{fp.lib_id}: {', '.join(ghost[:8])} — their nets can "
                    f"never reach copper"
                ),
            ))
        if numeric_pads:
            coverage = len(declared & numeric_pads) / len(numeric_pads)
            if coverage < MIN_PAD_COVERAGE:
                out.append(Violation(
                    constraint=f"pin_pad_coverage({comp.ref})",
                    refs=(comp.ref,),
                    severity=Severity.CRITICAL,
                    measured=coverage, limit=MIN_PAD_COVERAGE,
                    message=(
                        f"{comp.ref} ({comp.value}) pin map covers only "
                        f"{len(declared & numeric_pads)}/{len(numeric_pads)} pads of "
                        f"{fp.lib_id} — a stub pin map floats unmodeled pads "
                        f"(supply pins included)"
                    ),
                ))
    return tuple(out)


def check_diode_polarity(
    requirements: ProjectRequirements,
) -> tuple[Violation, ...]:
    """Reversed protection diodes: anode on a power rail, cathode on GND.

    A unidirectional TVS/clamp across a positive rail must point
    cathode-to-rail; the reverse is a forward diode shorting the
    supply at ~0.7V. LEDs are excluded (anode-on-rail is their normal
    orientation, ballasted by a series resistor).
    """
    pin_names: dict[tuple[str, str], str] = {}
    for comp in requirements.components:
        for pin in comp.pins:
            pin_names[(comp.ref, pin.number)] = pin.name.upper()

    out: list[Violation] = []
    for comp in requirements.components:
        if not comp.ref.startswith("D"):
            continue
        ident = (comp.value + " " + comp.footprint).lower()
        if any(t in ident for t in _LED_TOKENS):
            continue
        anode_net = cathode_net = None
        for net in requirements.nets:
            for conn in net.connections:
                if conn.ref != comp.ref:
                    continue
                name = pin_names.get((comp.ref, conn.pin), "")
                if name in ("A", "ANODE", "+"):
                    anode_net = net.name
                elif name in ("K", "C", "CATHODE", "-"):
                    cathode_net = net.name
        if anode_net is None or cathode_net is None:
            continue
        if _is_power_rail(anode_net) and _is_gnd(cathode_net):
            out.append(Violation(
                constraint=f"diode_polarity({comp.ref})",
                refs=(comp.ref,),
                severity=Severity.CRITICAL,
                measured=1.0, limit=0.0,
                message=(
                    f"{comp.ref} ({comp.value}) is forward-biased across the "
                    f"supply: anode on {anode_net}, cathode on {cathode_net} "
                    f"— a clamp must point cathode-to-rail; as wired it "
                    f"shorts the input at ~0.7V"
                ),
            ))
    return tuple(out)


def lint_netlist(
    requirements: ProjectRequirements,
    footprints: Mapping[str, Footprint],
) -> tuple[Violation, ...]:
    """All netlist lint checks, in one call."""
    return (
        check_pin_pad_coverage(requirements, footprints)
        + check_diode_polarity(requirements)
    )


__all__ = [
    "MIN_PAD_COVERAGE",
    "check_diode_polarity",
    "check_pin_pad_coverage",
    "lint_netlist",
]

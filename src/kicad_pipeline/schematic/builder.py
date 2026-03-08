"""Top-level schematic builder orchestrator.

Combines component placement, wire routing, and S-expression serialisation
into a single pipeline entry point.  The primary public surface is:

* :func:`build_schematic` — assemble a :class:`~kicad_pipeline.models.schematic.Schematic`
  from :class:`~kicad_pipeline.models.requirements.ProjectRequirements`.
* :func:`schematic_to_sexp` — serialise a :class:`~kicad_pipeline.models.schematic.Schematic`
  to a KiCad S-expression tree.
* :func:`write_schematic` — write the S-expression tree to a ``.kicad_sch`` file.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    KICAD_GENERATOR,
    KICAD_SCH_VERSION,
    SCHEMATIC_PIN_LENGTH_MM,
)
from kicad_pipeline.exceptions import SchematicError
from kicad_pipeline.pcb.footprints import footprint_for_component
from kicad_pipeline.schematic.symbols import get_or_make_symbol

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import (
        Component,
        FeatureBlock,
        Net,
        NetConnection,
        ProjectRequirements,
    )
from kicad_pipeline.models.schematic import (
    FontEffect,
    GlobalLabel,
    Junction,
    Label,
    LibCircle,
    LibPin,
    LibPolyline,
    LibRectangle,
    LibSymbol,
    NoConnect,
    Point,
    PowerSymbol,
    Schematic,
    Stroke,
    StrokeType,
    SymbolInstance,
    TextProperty,
    Wire,
)
from kicad_pipeline.schematic.placement import layout_schematic
from kicad_pipeline.schematic.wiring import route_net
from kicad_pipeline.sexp.parser import parse_file
from kicad_pipeline.sexp.writer import SExpNode, write_file

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known power net names → KiCad power library IDs
# ---------------------------------------------------------------------------

_POWER_LIB_IDS: dict[str, str] = {
    "GND": "power:GND",
    "GNDD": "power:GNDD",
    "VBUS": "power:VBUS",
    "+5V": "power:+5V",
    "+3V3": "power:+3V3",
    "+3.3V": "power:+3.3V",
    "VCC": "power:VCC",
    "+1V8": "power:+1V8",
}

# Power net names whose symbols should be rendered pointing downward (GND family)
_GND_NETS: frozenset[str] = frozenset({"GND", "GNDD"})


# ---------------------------------------------------------------------------
# KiCad power library reader — extracts real symbols from the installed library
# ---------------------------------------------------------------------------

_KICAD_POWER_LIB_PATHS: tuple[str, ...] = (
    # macOS
    "/Applications/KiCad/KiCad.app/Contents/SharedSupport/symbols/power.kicad_sym",
    # Linux common locations
    "/usr/share/kicad/symbols/power.kicad_sym",
    "/usr/local/share/kicad/symbols/power.kicad_sym",
    # Flatpak
    "/var/lib/flatpak/app/org.kicad.KiCad/current/active/files/share/kicad/symbols/power.kicad_sym",
)

# Cache: maps symbol short name (e.g. "+5V") to its parsed SExpNode
_power_lib_cache: dict[str, SExpNode] = {}
_power_lib_loaded: bool = False


def _load_kicad_power_library() -> None:
    """Load and parse the KiCad power symbol library once.

    Populates :data:`_power_lib_cache` with a mapping from symbol short name
    (e.g. ``"+5V"``, ``"GND"``) to the raw ``SExpNode`` tree from the library.
    """
    global _power_lib_loaded
    if _power_lib_loaded:
        return
    _power_lib_loaded = True

    lib_path: Path | None = None
    for candidate in _KICAD_POWER_LIB_PATHS:
        p = Path(candidate)
        if p.is_file():
            lib_path = p
            break

    if lib_path is None:
        log.warning("KiCad power library not found; power symbols will use fallback definitions")
        return

    log.info("Loading KiCad power library from %s", lib_path)
    tree = parse_file(lib_path)

    # The library file is: (kicad_symbol_lib ... (symbol "NAME" ...) ...)
    if not isinstance(tree, list):
        return
    for child in tree:
        if isinstance(child, list) and len(child) >= 2 and child[0] == "symbol":
            sym_name = child[1]
            if isinstance(sym_name, str):
                _power_lib_cache[sym_name] = child


def _get_kicad_power_symbol(net_name: str) -> SExpNode | None:
    """Retrieve a power symbol definition from the KiCad library.

    Args:
        net_name: The power net name (e.g. ``"+5V"``, ``"GND"``).

    Returns:
        The raw ``SExpNode`` tree for the symbol, or ``None`` if not found.
    """
    _load_kicad_power_library()
    return _power_lib_cache.get(net_name)


def _power_lib_symbol_sexp(lib_id: str, net_name: str) -> SExpNode:
    """Return a power lib_symbol definition, preferring the real KiCad library.

    Reads the actual ``power.kicad_sym`` from the KiCad installation and
    extracts the exact symbol definition.  Falls back to a minimal built-in
    definition only if KiCad is not installed.

    Args:
        lib_id: Full library ID (e.g. ``power:GND``).
        net_name: The net name (e.g. ``GND``).

    Returns:
        ``SExpNode`` list for the ``(symbol ...)`` in ``lib_symbols``.
    """
    short = lib_id.split(":")[-1] if ":" in lib_id else lib_id

    # Try to get the ACTUAL KiCad symbol
    real_sym = _get_kicad_power_symbol(short)
    if real_sym is not None and isinstance(real_sym, list):
        # Clone the symbol tree and rename to use full lib_id (power:NAME)
        result = list(real_sym)
        result[1] = lib_id  # Replace short name with full lib_id
        return result

    # Fallback: minimal definition if KiCad library not available
    log.warning("Power symbol %r not found in KiCad library; using fallback", net_name)
    is_gnd = net_name in _GND_NETS

    _stroke_default: SExpNode = ["stroke", ["width", 0], ["type", "default"]]
    _fill_none: SExpNode = ["fill", ["type", "none"]]

    if is_gnd:
        unit_body: list[SExpNode] = [
            "symbol", f"{short}_0_1",
            [
                "polyline",
                ["pts", ["xy", 0, 0], ["xy", 0, -1.27], ["xy", 1.27, -1.27],
                 ["xy", 0, -2.54], ["xy", -1.27, -1.27], ["xy", 0, -1.27]],
                _stroke_default, _fill_none,
            ],
        ]
        pin_at: SExpNode = ["at", 0, 0, 270]
        ref_at: SExpNode = ["at", 0, -6.35, 0]
        val_at: SExpNode = ["at", 0, -3.81, 0]
    else:
        unit_body = [
            "symbol", f"{short}_0_1",
            ["polyline", ["pts", ["xy", -0.762, 1.27], ["xy", 0, 2.54]],
             _stroke_default, _fill_none],
            ["polyline", ["pts", ["xy", 0, 2.54], ["xy", 0.762, 1.27]],
             _stroke_default, _fill_none],
            ["polyline", ["pts", ["xy", 0, 0], ["xy", 0, 2.54]],
             _stroke_default, _fill_none],
        ]
        pin_at = ["at", 0, 0, 90]
        ref_at = ["at", 0, -3.81, 0]
        val_at = ["at", 0, 3.556, 0]

    return [
        "symbol", lib_id,
        ["power"],
        ["pin_numbers", ["hide", True]],
        ["pin_names", ["offset", 0], ["hide", True]],
        ["exclude_from_sim", False],
        ["in_bom", True],
        ["on_board", True],
        ["property", "Reference", "#PWR", ref_at,
         _effects_sexp(FontEffect(hidden=True))],
        ["property", "Value", net_name, val_at,
         _effects_sexp(FontEffect())],
        ["property", "Footprint", "", ["at", 0, 0, 0],
         _effects_sexp(FontEffect(hidden=True))],
        ["property", "Datasheet", "", ["at", 0, 0, 0],
         _effects_sexp(FontEffect(hidden=True))],
        unit_body,
        [
            "symbol", f"{short}_1_1",
            ["pin", "power_in", "line", pin_at, ["length", 0],
             ["name", "~", ["effects", ["font", ["size", 1.27, 1.27]]]],
             ["number", "1", ["effects", ["font", ["size", 1.27, 1.27]]]]],
        ],
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_uuid() -> str:
    """Return a fresh RFC-4122 UUID string.

    Returns:
        Hyphenated UUID string.
    """
    return str(uuid.uuid4())


def _component_lib_id(component: Component) -> str:
    """Derive the KiCad lib_id string for a component.

    Uses the component's footprint as a heuristic to select the most
    appropriate KiCad symbol library:

    * Resistors (``R_*``) → ``Device:R``
    * Capacitors (``C_*``) → ``Device:C``
    * LEDs (``LED_*``) → ``Device:LED``
    * Transistors (``SOT-23``) → ``Device:Q_NPN_BCE``
    * USB connectors → ``Connector:USB_C_Receptacle_USB2.0``
    * Everything else → ``kicad-ai:{component.value}``

    Args:
        component: The :class:`~kicad_pipeline.models.requirements.Component`
            to classify.

    Returns:
        A lib_id string suitable for use in a KiCad schematic.
    """
    fp = component.footprint
    if fp.startswith("R_"):
        return "Device:R"
    if fp.startswith("C_"):
        return "Device:C"
    if fp.startswith("LED_"):
        return "Device:LED"
    if fp in ("SOT-23", "SOT-23-3"):
        return "Device:Q_NPN_BCE"
    if fp in ("SOD-123", "SOD-123W"):
        return "Device:D"
    if "USB_C" in fp:
        return "Connector:USB_C_Receptacle_USB2.0"
    if "SOT-223" in fp:
        return "Device:Regulator_Linear"
    # Fallback: custom kicad-ai library symbol
    safe_value = component.value.replace(" ", "_").replace("/", "_")
    return f"kicad-ai:{safe_value}"


def _make_lib_symbol(lib_id: str, component: Component) -> LibSymbol:
    """Build a minimal :class:`LibSymbol` definition for *component*.

    The symbol body is a simple rectangle with one pin per
    :class:`~kicad_pipeline.models.requirements.Pin` in the component.  Pin
    positions are stacked vertically on the left side.

    Args:
        lib_id: KiCad library identifier string.
        component: Source component providing pin metadata.

    Returns:
        A :class:`LibSymbol` suitable for inclusion in the ``lib_symbols``
        section of a KiCad schematic.
    """
    pins: list[LibPin] = []
    pin_spacing = 2.54  # mm
    body_left = -5.08
    body_right = 5.08
    body_top = -pin_spacing * (len(component.pins) / 2.0)
    body_bottom = pin_spacing * (len(component.pins) / 2.0)

    for idx, pin in enumerate(component.pins):
        y_pos = body_top + idx * pin_spacing + pin_spacing / 2
        lib_pin = LibPin(
            number=pin.number,
            name=pin.name,
            pin_type=pin.pin_type.value,
            at=Point(x=body_left - SCHEMATIC_PIN_LENGTH_MM, y=y_pos),
            rotation=0.0,
            length=SCHEMATIC_PIN_LENGTH_MM,
        )
        pins.append(lib_pin)

    rect = LibRectangle(
        start=Point(x=body_left, y=body_top),
        end=Point(x=body_right, y=body_bottom),
        stroke=Stroke(),
        fill="background",
    )

    return LibSymbol(
        lib_id=lib_id,
        pins=tuple(pins),
        shapes=(rect,),
    )


def _make_symbol_instance(
    component: Component,
    lib_id: str,
    position: Point,
    lib_sym: LibSymbol | None = None,
    footprint_lib_id: str | None = None,
) -> SymbolInstance:
    """Create a placed :class:`SymbolInstance` for *component*.

    Args:
        component: Source component.
        lib_id: KiCad library identifier string.
        position: Placement position on the schematic canvas.
        lib_sym: Optional LibSymbol to compute body dimensions from.
        footprint_lib_id: Resolved PCB footprint library ID. Falls back to
            ``component.footprint`` when ``None``.

    Returns:
        A :class:`SymbolInstance` with a fresh UUID and text properties.
    """
    # Compute body_half from LibSymbol shapes if available
    body_half: float
    if lib_sym is not None and lib_sym.shapes:
        # Find the rectangle shape for body extent
        y_min = 0.0
        y_max = 0.0
        for shape in lib_sym.shapes:
            if isinstance(shape, LibRectangle):
                y_min = min(y_min, shape.start.y, shape.end.y)
                y_max = max(y_max, shape.start.y, shape.end.y)
        if y_max > y_min:
            body_half = max(abs(y_min), abs(y_max))
        else:
            # Fallback: use pin extents
            if lib_sym.pins:
                pin_ys = [p.at.y for p in lib_sym.pins]
                body_half = max(abs(min(pin_ys)), abs(max(pin_ys))) + 2.54
            else:
                body_half = 2.54
    else:
        pin_spacing = 2.54
        body_half = pin_spacing * (len(component.pins) / 2.0)
    label_clearance = 2.54
    ref_prop = TextProperty(
        text=component.ref,
        position=Point(x=position.x, y=position.y - body_half - label_clearance),
    )
    val_prop = TextProperty(
        text=component.value,
        position=Point(x=position.x, y=position.y + body_half + label_clearance),
    )
    resolved_fp = footprint_lib_id if footprint_lib_id is not None else component.footprint
    return SymbolInstance(
        lib_id=lib_id,
        ref=component.ref,
        value=component.value,
        footprint=resolved_fp,
        position=position,
        lcsc=component.lcsc,
        uuid=_new_uuid(),
        ref_property=ref_prop,
        value_property=val_prop,
    )


def _annotate_requirements(requirements: ProjectRequirements) -> ProjectRequirements:
    """Auto-annotate unannotated reference designators (R? -> R1, R2, ...).

    Scans all component refs for the ``?`` placeholder and assigns sequential
    numbers by prefix.  Updates refs in components, nets, and feature blocks
    consistently.

    Args:
        requirements: Original requirements (may have unannotated refs).

    Returns:
        New :class:`ProjectRequirements` with all refs annotated.  Returns the
        original unchanged if no ``?`` refs are found.
    """
    # Check if any refs need annotation
    needs_annotation = any("?" in c.ref for c in requirements.components)
    if not needs_annotation:
        return requirements

    # Build old_ref -> new_ref mapping (by list index to handle duplicate "R?" etc.)
    prefix_counters: dict[str, int] = {}
    # First pass: register existing annotated refs to avoid collisions
    for comp in requirements.components:
        ref = comp.ref
        if "?" not in ref:
            prefix = "".join(ch for ch in ref if ch.isalpha())
            num_str = ref[len(prefix):]
            if num_str.isdigit():
                prefix_counters[prefix] = max(prefix_counters.get(prefix, 0), int(num_str))

    # Second pass: assign numbers to unannotated refs
    annotated_refs: list[str] = []
    # Track per-old-ref indices to build a mapping for net/feature updates
    old_to_new: dict[str, list[str]] = {}  # "R?" -> ["R1", "R2", ...]
    for comp in requirements.components:
        ref = comp.ref
        if "?" in ref:
            prefix = ref.replace("?", "")
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            new_ref = f"{prefix}{count}"
            annotated_refs.append(new_ref)
            old_to_new.setdefault(ref, []).append(new_ref)
        else:
            annotated_refs.append(ref)

    # Create new components with annotated refs
    new_components = tuple(
        replace(comp, ref=annotated_refs[idx])
        for idx, comp in enumerate(requirements.components)
    )

    # For nets: build (old_ref, pin_number) -> queue of new_refs
    # When multiple components share the same old ref and pin number,
    # net connections consume from the queue in order.
    pin_ref_queues: dict[tuple[str, str], list[str]] = {}
    for idx, comp in enumerate(requirements.components):
        for pin in comp.pins:
            key = (comp.ref, pin.number)
            pin_ref_queues.setdefault(key, []).append(annotated_refs[idx])

    # Update net connections — pop from queue to disambiguate duplicates
    new_nets: list[Net] = []
    for net in requirements.nets:
        new_conns: list[NetConnection] = []
        for conn in net.connections:
            key = (conn.ref, conn.pin)
            queue = pin_ref_queues.get(key)
            new_ref = queue.pop(0) if queue else conn.ref
            new_conns.append(replace(conn, ref=new_ref))
        new_nets.append(replace(net, connections=tuple(new_conns)))

    # Update feature blocks — use queue to map duplicate refs
    feature_ref_queues: dict[str, list[str]] = {k: list(v) for k, v in old_to_new.items()}
    new_features: list[FeatureBlock] = []
    for fb in requirements.features:
        new_fb_refs: list[str] = []
        for ref in fb.components:
            fq = feature_ref_queues.get(ref)
            new_fb_refs.append(fq.pop(0) if fq else ref)
        new_features.append(replace(fb, components=tuple(new_fb_refs)))

    log.info(
        "auto_annotate: annotated %d refs",
        sum(1 for c in requirements.components if "?" in c.ref),
    )

    return replace(
        requirements,
        components=new_components,
        nets=tuple(new_nets),
        features=tuple(new_features),
    )


def _collect_power_nets(requirements: ProjectRequirements) -> list[str]:
    """Return a deduplicated list of power net names found in *requirements*.

    Searches both the explicit net list and the pin net assignments of every
    component.  Only nets whose names appear in :data:`_POWER_LIB_IDS` are
    returned.

    Args:
        requirements: Project requirements document.

    Returns:
        List of recognised power net names (no duplicates, stable order).
    """
    seen: set[str] = set()
    result: list[str] = []
    candidates: list[str] = [n.name for n in requirements.nets]
    for comp in requirements.components:
        for pin in comp.pins:
            if pin.net is not None:
                candidates.append(pin.net)
    for name in candidates:
        if name in _POWER_LIB_IDS and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _power_symbol_offset(
    side: str, stub: float, is_gnd: bool,
) -> tuple[float, float, float]:
    """Compute (dx, dy, rotation) for a power symbol relative to its pin.

    Args:
        side: Pin side on the component body.
        stub: Distance in mm from pin to power symbol.
        is_gnd: Whether this is a ground net.

    Returns:
        Tuple of (dx, dy, rotation_degrees).
    """
    if side == "top":
        return 0.0, -stub, (0.0 if not is_gnd else 180.0)
    if side == "bottom":
        return 0.0, stub, (180.0 if not is_gnd else 0.0)
    if side == "right":
        return stub, 0.0, (270.0 if not is_gnd else 90.0)
    # left (default)
    return -stub, 0.0, (90.0 if not is_gnd else 270.0)


def _make_power_symbols_at_pins(
    power_net_names: list[str],
    pin_positions: dict[tuple[str, str], Point],
    pin_sides: dict[tuple[str, str], str],
    requirements: ProjectRequirements,
) -> tuple[list[PowerSymbol], list[Wire], list[GlobalLabel], list[Junction]]:
    """Create :class:`PowerSymbol` instances at power pin locations.

    When multiple pins on the same component, same side, and same power net
    exist, they are consolidated into a single power symbol at their vertical
    (or horizontal) midpoint.  A bus wire connects all pins to the shared
    symbol, with junctions at each T-connection (except bus endpoints).

    Single-pin groups keep the original one-symbol-per-pin behaviour.

    Args:
        power_net_names: Recognised power net names.
        pin_positions: Mapping ``(ref, pin_number)`` to absolute position.
        pin_sides: Mapping ``(ref, pin_number)`` to side string.
        requirements: Project requirements (for net connection lists).

    Returns:
        Tuple of ``(power_symbols, wires, extra_global_labels, junctions)``.
    """
    symbols: list[PowerSymbol] = []
    wires: list[Wire] = []
    junctions: list[Junction] = []
    pwr_idx = 0
    power_set = set(power_net_names)
    stub = 7.62  # mm distance from pin to power symbol

    # Group pins by (ref, net_name, side) so duplicates can be consolidated
    _group_key = tuple[str, str, str]  # (ref, net_name, side)
    groups: dict[_group_key, list[tuple[str, Point]]] = {}

    for net in requirements.nets:
        if net.name not in power_set:
            continue
        for conn in net.connections:
            pin_key = (conn.ref, conn.pin)
            pin_pos = pin_positions.get(pin_key)
            if pin_pos is None:
                continue
            side = pin_sides.get(pin_key, "left")
            gk: _group_key = (conn.ref, net.name, side)
            groups.setdefault(gk, []).append((conn.pin, pin_pos))

    for (_ref, net_name, side), pin_list in groups.items():
        lib_id = _POWER_LIB_IDS.get(net_name, f"power:{net_name}")
        is_gnd = net_name in _GND_NETS
        dx, dy, rotation = _power_symbol_offset(side, stub, is_gnd)

        if len(pin_list) == 1:
            # Single pin — original behaviour: one stub wire + one symbol
            _pin_num, pin_pos = pin_list[0]
            sx, sy = pin_pos.x + dx, pin_pos.y + dy
            wires.append(Wire(
                start=pin_pos, end=Point(x=sx, y=sy),
                stroke=Stroke(), uuid=_new_uuid(),
            ))
            pwr_idx += 1
            symbols.append(PowerSymbol(
                lib_id=lib_id,
                position=Point(x=sx, y=sy),
                ref=f"#PWR0{pwr_idx:02d}",
                value=net_name,
                rotation=rotation,
                uuid=_new_uuid(),
            ))
            continue

        # Multiple pins — consolidate: one symbol at midpoint, bus wire
        # Sort pins by the axis perpendicular to the stub direction.
        is_vertical_stub = side in ("top", "bottom")
        if is_vertical_stub:
            # Pins arranged horizontally; bus is horizontal
            pin_list.sort(key=lambda p: p[1].x)
        else:
            # Pins arranged vertically; bus is vertical
            pin_list.sort(key=lambda p: p[1].y)

        # Compute bus line coordinate (offset from pins by stub distance)
        first_pos = pin_list[0][1]
        bus_fixed = (first_pos.y + dy) if is_vertical_stub else (first_pos.x + dx)

        # Compute midpoint along the bus for the power symbol
        if is_vertical_stub:
            coords = [p.x for _, p in pin_list]
            mid = (min(coords) + max(coords)) / 2.0
            sym_x, sym_y = mid, bus_fixed
        else:
            coords = [p.y for _, p in pin_list]
            mid = (min(coords) + max(coords)) / 2.0
            sym_x, sym_y = bus_fixed, mid

        pwr_idx += 1
        symbols.append(PowerSymbol(
            lib_id=lib_id,
            position=Point(x=sym_x, y=sym_y),
            ref=f"#PWR0{pwr_idx:02d}",
            value=net_name,
            rotation=rotation,
            uuid=_new_uuid(),
        ))

        # Draw bus wire spanning all pin stub endpoints
        if is_vertical_stub:
            bus_start = Point(x=pin_list[0][1].x, y=bus_fixed)
            bus_end = Point(x=pin_list[-1][1].x, y=bus_fixed)
        else:
            bus_start = Point(x=bus_fixed, y=pin_list[0][1].y)
            bus_end = Point(x=bus_fixed, y=pin_list[-1][1].y)

        wires.append(Wire(
            start=bus_start, end=bus_end,
            stroke=Stroke(), uuid=_new_uuid(),
        ))

        # Draw horizontal/vertical stub wires from each pin to the bus,
        # and add junctions at T-connections (all except first and last)
        for idx, (_pin_num, pin_pos) in enumerate(pin_list):
            if is_vertical_stub:
                stub_end = Point(x=pin_pos.x, y=bus_fixed)
            else:
                stub_end = Point(x=bus_fixed, y=pin_pos.y)
            wires.append(Wire(
                start=pin_pos, end=stub_end,
                stroke=Stroke(), uuid=_new_uuid(),
            ))
            # Add junction at T-connections (interior points on bus)
            if 0 < idx < len(pin_list) - 1:
                junctions.append(Junction(
                    position=stub_end,
                    uuid=_new_uuid(),
                ))

    return symbols, wires, [], junctions


def _refine_feature_map_by_connectivity(
    feature_map: dict[str, str],
    requirements: ProjectRequirements,
) -> dict[str, str]:
    """Move small components to the feature group of their neighbours.

    Only moves components with ≤4 pins when ALL their non-power net
    neighbours belong to a single different feature group.  This prevents
    accidental moves while still grouping sensor connectors with their
    voltage divider chains.

    Large components (>4 pins) are never moved — they anchor their zone.
    """
    power_nets = set(_POWER_LIB_IDS.keys())
    ref_pins = {c.ref: len(c.pins) for c in requirements.components}

    # Build adjacency: ref → set of connected refs (via non-power nets)
    neighbours: dict[str, set[str]] = {c.ref: set() for c in requirements.components}
    for net in requirements.nets:
        if net.name in power_nets:
            continue
        refs_on_net = {conn.ref for conn in net.connections}
        for r in refs_on_net:
            neighbours.setdefault(r, set()).update(refs_on_net - {r})

    refined = dict(feature_map)
    for ref, my_feature in feature_map.items():
        # Only consider small components (≤4 pins)
        if ref_pins.get(ref, 0) > 4:
            continue
        nbrs = neighbours.get(ref, set())
        if not nbrs:
            continue
        # Collect feature groups of all neighbours
        nbr_features = {feature_map.get(n, "Peripherals") for n in nbrs}
        # Only move if ALL neighbours are in exactly one OTHER feature group
        if len(nbr_features) == 1:
            target = next(iter(nbr_features))
            if target != my_feature:
                refined[ref] = target
                log.debug("refine_feature_map: %s moved %s -> %s", ref, my_feature, target)

    return refined


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_schematic(
    requirements: ProjectRequirements,
) -> Schematic:
    """Build a complete :class:`Schematic` from *requirements*.

    Steps:

    1. Generate a :class:`LibSymbol` definition for every component.
    2. Compute component positions using :func:`~.placement.layout_schematic`.
    3. Create :class:`SymbolInstance` objects for each placed component.
    4. Route wires for all nets using :func:`~.wiring.route_net`.
    5. Add :class:`PowerSymbol` instances for recognised power nets.
    6. Assemble and return the complete :class:`Schematic`.

    Args:
        requirements: Fully-populated project requirements document.

    Returns:
        A complete :class:`Schematic` ready for serialisation.

    Raises:
        :class:`~kicad_pipeline.exceptions.SchematicError`: If no components
            are provided or a critical build step fails.
    """
    if not requirements.components:
        raise SchematicError("Cannot build schematic: requirements has no components")

    # Auto-annotate unannotated refs (R? -> R1, C? -> C1, etc.)
    requirements = _annotate_requirements(requirements)

    log.info(
        "build_schematic: %d components, %d nets",
        len(requirements.components),
        len(requirements.nets),
    )

    # ------------------------------------------------------------------
    # Step 1: Derive feature map from FeatureBlocks, then refine by connectivity
    # ------------------------------------------------------------------
    feature_map: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            feature_map[ref] = fb.name

    # Refine feature map by connectivity: move small components (≤4 pins)
    # to the feature group where ALL their non-power neighbours live.
    # This groups sensor connectors (J2-J5) with their voltage dividers.
    feature_map = _refine_feature_map_by_connectivity(feature_map, requirements)

    # ------------------------------------------------------------------
    # Step 2: Compute positions (with pin-count-aware spacing)
    # ------------------------------------------------------------------
    all_refs = [c.ref for c in requirements.components]
    pin_count_map = {c.ref: len(c.pins) for c in requirements.components}

    # Auto-select page size: A3 only for very large designs
    # Count "active" pins (those on a net) rather than total pins to avoid
    # triggering on large connectors with many unused pass-through pins.
    active_pins = sum(
        1 for c in requirements.components for p in c.pins
        if p.net is not None
    )
    max_pins = max((len(c.pins) for c in requirements.components), default=0)
    paper = "A4"
    if len(requirements.components) > 15 or active_pins > 60 or max_pins >= 20:
        paper = "A3"
        log.info(
            "build_schematic: auto-selected A3 page (%d components, %d pins)",
            len(requirements.components),
            active_pins,
        )

    # Build adjacency map for connectivity-based sorting within zones
    adjacency: dict[str, set[str]] = {c.ref: set() for c in requirements.components}
    power_net_set_adj = set(_POWER_LIB_IDS.keys())
    for net in requirements.nets:
        if net.name in power_net_set_adj:
            continue
        refs_on_net = {conn.ref for conn in net.connections}
        for r in refs_on_net:
            adjacency.setdefault(r, set()).update(refs_on_net - {r})

    positions = layout_schematic(
        all_refs, feature_map, pin_count_map=pin_count_map, paper=paper,
        adjacency=adjacency,
    )

    # ------------------------------------------------------------------
    # Step 3: Build lib_symbols + symbol instances
    # ------------------------------------------------------------------
    lib_symbols_list: list[LibSymbol] = []
    symbols_list: list[SymbolInstance] = []
    seen_lib_ids: set[str] = set()
    lib_cache: dict[str, LibSymbol] = {}
    # Map from component ref to its LibSymbol for pin position lookup
    comp_lib_sym: dict[str, LibSymbol] = {}

    for comp in requirements.components:
        lib_sym = get_or_make_symbol(comp, lib_cache)
        lib_id = lib_sym.lib_id
        comp_lib_sym[comp.ref] = lib_sym
        if lib_id not in seen_lib_ids:
            lib_symbols_list.append(lib_sym)
            seen_lib_ids.add(lib_id)

        pos = positions.get(comp.ref, Point(x=0.0, y=0.0))
        # Resolve PCB footprint lib_id for the Footprint property
        try:
            pcb_fp = footprint_for_component(comp.ref, comp.value, comp.footprint, comp.lcsc)
            fp_lib_id = pcb_fp.lib_id
        except Exception:
            fp_lib_id = comp.footprint  # fallback to bare name
        inst = _make_symbol_instance(comp, lib_id, pos, lib_sym, footprint_lib_id=fp_lib_id)
        symbols_list.append(inst)

    # ------------------------------------------------------------------
    # Step 4: Build pin-position map for wire routing
    # ------------------------------------------------------------------
    # Use actual LibSymbol pin positions (multi-sided layout from symbols.py)
    pin_positions: dict[tuple[str, str], Point] = {}
    pin_sides: dict[tuple[str, str], str] = {}
    for comp in requirements.components:
        sym_pos = positions.get(comp.ref, Point(x=0.0, y=0.0))
        comp_sym = comp_lib_sym.get(comp.ref)
        if comp_sym is not None:
            for lib_pin in comp_sym.pins:
                pin_x = sym_pos.x + lib_pin.at.x
                # KiCad lib_symbol Y-axis: positive = up (mathematical)
                # Schematic Y-axis: positive = down (screen)
                # Negate Y to convert from lib space to schematic space
                pin_y = sym_pos.y - lib_pin.at.y
                pin_positions[(comp.ref, lib_pin.number)] = Point(x=pin_x, y=pin_y)
                # Determine side from lib pin rotation (KiCad convention):
                # 0°=left-side pin (extends RIGHT toward body)
                # 180°=right-side pin (extends LEFT toward body)
                # 270°=top pin (extends DOWN toward body)
                # 90°=bottom pin (extends UP toward body)
                rot = lib_pin.rotation % 360.0
                if abs(rot) < 1.0:
                    side = "left"
                elif abs(rot - 180.0) < 1.0:
                    side = "right"
                elif abs(rot - 270.0) < 1.0:
                    side = "top"
                elif abs(rot - 90.0) < 1.0:
                    side = "bottom"
                else:
                    side = "left"
                pin_sides[(comp.ref, lib_pin.number)] = side

    # ------------------------------------------------------------------
    # Step 5: Route nets (skip power nets — handled by power symbols)
    # ------------------------------------------------------------------
    all_wires: list[Wire] = []
    all_junctions: list[Junction] = []
    all_global_labels: list[GlobalLabel] = []
    all_local_labels: list[Label] = []

    power_net_set = set(_POWER_LIB_IDS.keys())
    for net in requirements.nets:
        if net.name in power_net_set:
            continue  # power nets use power symbols, not labels
        ws, js, gls, ls = route_net(
            net, pin_positions, use_global_labels=True, pin_sides=pin_sides,
        )
        all_wires.extend(ws)
        all_junctions.extend(js)
        all_global_labels.extend(gls)
        all_local_labels.extend(ls)

    # ------------------------------------------------------------------
    # Step 6: Power symbols at pin locations
    # ------------------------------------------------------------------
    power_net_names = _collect_power_nets(requirements)
    power_syms, power_wires, power_labels, power_junctions = _make_power_symbols_at_pins(
        power_net_names, pin_positions, pin_sides, requirements,
    )
    all_wires.extend(power_wires)
    all_global_labels.extend(power_labels)
    all_junctions.extend(power_junctions)

    # ------------------------------------------------------------------
    # Step 7: No-connect markers for pins with no net assignment
    # ------------------------------------------------------------------
    connected_pins: set[tuple[str, str]] = set()
    for net in requirements.nets:
        for conn in net.connections:
            connected_pins.add((conn.ref, conn.pin))

    no_connects: list[NoConnect] = []
    for comp in requirements.components:
        for pin in comp.pins:
            key = (comp.ref, pin.number)
            if key not in connected_pins and pin.pin_type.value != "no_connect":
                pin_pos = pin_positions.get(key)
                if pin_pos is not None:
                    no_connects.append(NoConnect(position=pin_pos, uuid=_new_uuid()))

    log.info(
        "build_schematic complete: %d symbols, %d wires, %d labels, %d power syms, %d no-connects",
        len(symbols_list),
        len(all_wires),
        len(all_global_labels) + len(all_local_labels),
        len(power_syms),
        len(no_connects),
    )

    return Schematic(
        lib_symbols=tuple(lib_symbols_list),
        symbols=tuple(symbols_list),
        power_symbols=tuple(power_syms),
        wires=tuple(all_wires),
        junctions=tuple(all_junctions),
        no_connects=tuple(no_connects),
        labels=tuple(all_local_labels),
        global_labels=tuple(all_global_labels),
        version=KICAD_SCH_VERSION,
        generator=KICAD_GENERATOR,
        paper=paper,
        title=requirements.project.name,
        date=datetime.date.today().isoformat(),
        revision=requirements.project.revision,
        company=requirements.project.author or "",
    )


# ---------------------------------------------------------------------------
# S-expression serialiser
# ---------------------------------------------------------------------------


def _effects_sexp(effects: FontEffect) -> SExpNode:
    """Serialise a :class:`FontEffect` to a KiCad ``(effects ...)`` node.

    Args:
        effects: Font effect settings.

    Returns:
        ``SExpNode`` list.
    """
    font: SExpNode = ["font", ["size", effects.size_x, effects.size_y]]
    node: list[SExpNode] = ["effects", font]
    if effects.justify:
        node.append(["justify", effects.justify])
    if effects.hidden:
        node.append(["hide", "yes"])
    return node


def _stroke_sexp(stroke: Stroke) -> SExpNode:
    """Serialise a :class:`Stroke` to a KiCad ``(stroke ...)`` node.

    Args:
        stroke: Stroke settings.

    Returns:
        ``SExpNode`` list.
    """
    return ["stroke", ["width", stroke.width], ["type", stroke.stroke_type.value]]


def _lib_pin_sexp(pin: LibPin) -> SExpNode:
    """Serialise a :class:`LibPin` to a KiCad ``(pin ...)`` node.

    Args:
        pin: Pin definition.

    Returns:
        ``SExpNode`` list.
    """
    name_effects = _effects_sexp(pin.name_effects)
    number_effects = _effects_sexp(pin.number_effects)
    return [
        "pin",
        pin.pin_type,
        "line",
        ["at", pin.at.x, pin.at.y, int(pin.rotation)],
        ["length", pin.length],
        ["name", pin.name, name_effects],
        ["number", pin.number, number_effects],
    ]


def _lib_symbol_sexp(sym: LibSymbol) -> SExpNode:
    """Serialise a :class:`LibSymbol` to a KiCad ``(symbol ...)`` node.

    Args:
        sym: Symbol definition.

    Returns:
        ``SExpNode`` list.
    """
    body: list[SExpNode] = ["symbol", sym.lib_id]

    # KiCad 9 required attributes
    body.append(["exclude_from_sim", False])
    body.append(["in_bom", True])
    body.append(["on_board", True])

    # KiCad 9 required properties on lib_symbols
    # Extract the symbol short name (after the colon in lib_id)
    short_name = sym.lib_id.split(":")[-1] if ":" in sym.lib_id else sym.lib_id
    for prop_name, prop_value, hidden in (
        ("Reference", short_name[0] if short_name else "U", False),
        ("Value", short_name, False),
        ("Footprint", "", True),
        ("Datasheet", "", True),
    ):
        prop_node: list[SExpNode] = [
            "property", prop_name, prop_value,
            ["at", 0, 0, 0],
            _effects_sexp(FontEffect(hidden=hidden)),
        ]
        body.append(prop_node)

    # Hide pin numbers — pin names provide identification
    body.append(["pin_numbers", ["hide", True]])

    # Unit body sub-symbol — use short name without lib prefix
    unit_body: list[SExpNode] = ["symbol", f"{short_name}_0_1"]
    for shape in sym.shapes:
        if isinstance(shape, LibRectangle):
            unit_body.append(
                [
                    "rectangle",
                    ["start", shape.start.x, shape.start.y],
                    ["end", shape.end.x, shape.end.y],
                    _stroke_sexp(shape.stroke),
                    ["fill", ["type", shape.fill]],
                ]
            )
        elif isinstance(shape, LibPolyline):
            pts: list[SExpNode] = ["pts"]
            for pt in shape.points:
                pts.append(["xy", pt.x, pt.y])
            unit_body.append(
                ["polyline", pts, _stroke_sexp(shape.stroke), ["fill", ["type", shape.fill]]]
            )
        elif isinstance(shape, LibCircle):
            unit_body.append(
                [
                    "circle",
                    ["center", shape.center.x, shape.center.y],
                    ["radius", shape.radius],
                    _stroke_sexp(shape.stroke),
                    ["fill", ["type", shape.fill]],
                ]
            )
    body.append(unit_body)

    # Pin sub-symbol — use short name without lib prefix
    pin_body: list[SExpNode] = ["symbol", f"{short_name}_1_1"]
    for pin in sym.pins:
        pin_body.append(_lib_pin_sexp(pin))
    body.append(pin_body)

    return body


def _symbol_instance_sexp(
    inst: SymbolInstance,
    project_name: str = "kicad-ai",
    root_uuid: str = "",
) -> SExpNode:
    """Serialise a :class:`SymbolInstance` to a KiCad ``(symbol ...)`` node.

    Args:
        inst: Placed symbol instance.
        project_name: Project name for the ``(instances ...)`` annotation block.
        root_uuid: Root schematic sheet UUID for the instances path.

    Returns:
        ``SExpNode`` list.
    """
    ref_at = inst.ref_property.position if inst.ref_property else inst.position
    val_at = inst.value_property.position if inst.value_property else inst.position
    ref_rot = inst.ref_property.rotation if inst.ref_property else 0.0
    val_rot = inst.value_property.rotation if inst.value_property else 0.0

    node: list[SExpNode] = [
        "symbol",
        ["lib_id", inst.lib_id],
        ["at", inst.position.x, inst.position.y, int(inst.rotation)],
        ["unit", inst.unit],
        ["exclude_from_sim", False],
        ["in_bom", inst.in_bom],
        ["on_board", inst.on_board],
        ["uuid", inst.uuid],
        [
            "property",
            "Reference",
            inst.ref,
            ["at", ref_at.x, ref_at.y, int(ref_rot)],
            _effects_sexp(FontEffect()),
        ],
        [
            "property",
            "Value",
            inst.value,
            ["at", val_at.x, val_at.y, int(val_rot)],
            _effects_sexp(FontEffect()),
        ],
        [
            "property",
            "Footprint",
            inst.footprint,
            ["at", inst.position.x, inst.position.y, 0],
            _effects_sexp(FontEffect(hidden=True)),
        ],
    ]
    if inst.lcsc is not None:
        node.append(
            [
                "property",
                "LCSC",
                inst.lcsc,
                ["at", inst.position.x, inst.position.y + 5.08, 0],
                _effects_sexp(FontEffect(hidden=True)),
            ]
        )
    # KiCad 9 requires (instances ...) inside each placed symbol for annotation.
    # Path must be "/{root_sheet_uuid}" for KiCad to resolve references.
    sheet_path = f"/{root_uuid}" if root_uuid else "/"
    node.append(
        [
            "instances",
            [
                "project",
                project_name,
                [
                    "path",
                    sheet_path,
                    ["reference", inst.ref],
                    ["unit", inst.unit],
                ],
            ],
        ]
    )
    return node


def _wire_sexp(wire: Wire) -> SExpNode:
    """Serialise a :class:`Wire` to a KiCad ``(wire ...)`` node.

    Args:
        wire: Wire segment.

    Returns:
        ``SExpNode`` list.
    """
    # KiCad 9 wire format uses (pts (xy ...) (xy ...)) and stroke type solid.
    stroke = Stroke(width=wire.stroke.width, stroke_type=StrokeType.SOLID)
    return [
        "wire",
        ["pts",
         ["xy", wire.start.x, wire.start.y],
         ["xy", wire.end.x, wire.end.y]],
        _stroke_sexp(stroke),
        ["uuid", wire.uuid],
    ]


def _junction_sexp(j: Junction) -> SExpNode:
    """Serialise a :class:`Junction` to a KiCad ``(junction ...)`` node.

    Args:
        j: Junction.

    Returns:
        ``SExpNode`` list.
    """
    return ["junction", ["at", j.position.x, j.position.y], ["diameter", j.diameter],
            ["uuid", j.uuid]]


def _label_sexp(label: Label) -> SExpNode:
    """Serialise a :class:`Label` to a KiCad ``(label ...)`` node.

    Args:
        label: Local net label.

    Returns:
        ``SExpNode`` list.
    """
    return [
        "label",
        label.text,
        ["at", label.position.x, label.position.y, int(label.rotation)],
        _effects_sexp(label.effects),
        ["uuid", label.uuid],
    ]


def _global_label_sexp(gl: GlobalLabel) -> SExpNode:
    """Serialise a :class:`GlobalLabel` to a KiCad ``(global_label ...)`` node.

    Args:
        gl: Global net label.

    Returns:
        ``SExpNode`` list.
    """
    return [
        "global_label",
        gl.text,
        ["shape", gl.shape],
        ["at", gl.position.x, gl.position.y, int(gl.rotation)],
        _effects_sexp(gl.effects),
        ["uuid", gl.uuid],
    ]


def _power_symbol_sexp(
    ps: PowerSymbol,
    project_name: str = "kicad-ai",
    root_uuid: str = "",
) -> SExpNode:
    """Serialise a :class:`PowerSymbol` to a KiCad ``(symbol ...)`` node.

    Args:
        ps: Power symbol instance.
        project_name: Project name for the ``(instances ...)`` annotation block.
        root_uuid: Root schematic sheet UUID for the instances path.

    Returns:
        ``SExpNode`` list.
    """
    sheet_path = f"/{root_uuid}" if root_uuid else "/"
    return [
        "symbol",
        ["lib_id", ps.lib_id],
        ["at", ps.position.x, ps.position.y, int(ps.rotation)],
        ["unit", 1],
        ["exclude_from_sim", False],
        ["in_bom", False],
        ["on_board", False],
        ["uuid", ps.uuid],
        [
            "property",
            "Reference",
            ps.ref,
            ["at", ps.position.x, ps.position.y - 2.54, 0],
            _effects_sexp(FontEffect(hidden=True)),
        ],
        [
            "property",
            "Value",
            ps.value,
            ["at", ps.position.x, ps.position.y + 2.54, 0],
            _effects_sexp(FontEffect()),
        ],
        [
            "instances",
            [
                "project",
                project_name,
                [
                    "path",
                    sheet_path,
                    ["reference", ps.ref],
                    ["unit", 1],
                ],
            ],
        ],
    ]


def schematic_to_sexp(
    schematic: Schematic,
    project_name: str = "kicad-ai",
) -> SExpNode:
    """Serialise a :class:`Schematic` to a KiCad S-expression tree.

    The output tree follows the KiCad 7+ ``.kicad_sch`` file format::

        (kicad_sch (version 20231120) (generator "kicad-ai-pipeline")
          (paper "A4")
          (lib_symbols ...)
          (symbol ...) ...
          (wire ...) ...
          (junction ...) ...
          (label ...) ...
          (global_label ...) ...
        )

    Args:
        schematic: The schematic to serialise.

    Returns:
        A nested :data:`~kicad_pipeline.sexp.writer.SExpNode` list representing
        the root ``(kicad_sch ...)`` expression.
    """
    root_uuid = str(uuid.uuid4())
    root: list[SExpNode] = [
        "kicad_sch",
        ["version", schematic.version],
        ["generator", schematic.generator],
        ["generator_version", schematic.generator_version],
        ["uuid", root_uuid],
        ["paper", schematic.paper],
    ]

    # Title block
    if schematic.title or schematic.date or schematic.revision or schematic.company:
        title_block: list[SExpNode] = ["title_block"]
        if schematic.title:
            title_block.append(["title", schematic.title])
        if schematic.date:
            title_block.append(["date", schematic.date])
        if schematic.revision:
            title_block.append(["rev", schematic.revision])
        if schematic.company:
            title_block.append(["company", schematic.company])
        root.append(title_block)

    # lib_symbols section
    lib_syms_node: list[SExpNode] = ["lib_symbols"]
    for lib_sym in schematic.lib_symbols:
        lib_syms_node.append(_lib_symbol_sexp(lib_sym))
    # Add lib_symbol definitions for power symbols
    seen_power_ids: set[str] = set()
    for ps in schematic.power_symbols:
        if ps.lib_id not in seen_power_ids:
            seen_power_ids.add(ps.lib_id)
            lib_syms_node.append(_power_lib_symbol_sexp(ps.lib_id, ps.value))
    root.append(lib_syms_node)

    # Symbol instances (regular + power)
    for inst in schematic.symbols:
        root.append(_symbol_instance_sexp(inst, project_name=project_name, root_uuid=root_uuid))
    for ps in schematic.power_symbols:
        root.append(_power_symbol_sexp(ps, project_name=project_name, root_uuid=root_uuid))

    # Wires
    for wire in schematic.wires:
        root.append(_wire_sexp(wire))

    # Junctions
    for j in schematic.junctions:
        root.append(_junction_sexp(j))

    # No-connects
    for nc in schematic.no_connects:
        root.append(["no_connect", ["at", nc.position.x, nc.position.y], ["uuid", nc.uuid]])

    # Local labels
    for label in schematic.labels:
        root.append(_label_sexp(label))

    # Global labels
    for gl in schematic.global_labels:
        root.append(_global_label_sexp(gl))

    # KiCad 9 canonical sheet_instances section (root sheet)
    root.append(["sheet_instances", ["path", "/", ["page", "1"]]])

    # KiCad 9 symbol_instances — maps each symbol UUID to its reference designator.
    # Path format: "/{root_sheet_uuid}/{symbol_uuid}" for KiCad 9.
    sym_instances: list[SExpNode] = ["symbol_instances"]
    for inst in schematic.symbols:
        sym_instances.append([
            "path", f"/{root_uuid}/{inst.uuid}",
            ["reference", inst.ref],
            ["unit", inst.unit],
        ])
    for ps in schematic.power_symbols:
        sym_instances.append([
            "path", f"/{root_uuid}/{ps.uuid}",
            ["reference", ps.ref],
            ["unit", 1],
        ])
    root.append(sym_instances)

    return root


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_schematic(schematic: Schematic, path: str | Path) -> None:
    """Serialise *schematic* and write it to a ``.kicad_sch`` file.

    Args:
        schematic: The schematic to write.
        path: Destination file path.  The parent directory must exist.

    Raises:
        :class:`~kicad_pipeline.exceptions.SchematicError`: If serialisation
            fails for any reason.
        :class:`OSError`: If the file cannot be written.
    """
    dest = Path(path)
    log.info("write_schematic → %s", dest)

    # Warn about unannotated ref designators (contain '?')
    for inst in schematic.symbols:
        if "?" in inst.ref:
            log.warning(
                "write_schematic: ref designator '%s' contains '?' — "
                "KiCad will show unannotated designators. "
                "Run annotation before writing.",
                inst.ref,
            )

    # Derive project name from filename stem so KiCad instance references match.
    proj_name = dest.stem
    try:
        sexp = schematic_to_sexp(schematic, project_name=proj_name)
        write_file(sexp, dest)
    except OSError:
        raise
    except Exception as exc:
        raise SchematicError(f"Failed to write schematic to {dest}: {exc}") from exc
    log.info("write_schematic: wrote %s", dest)

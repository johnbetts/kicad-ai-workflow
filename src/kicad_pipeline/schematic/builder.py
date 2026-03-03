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

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    KICAD_GENERATOR,
    KICAD_SCH_VERSION,
    SCHEMATIC_PIN_LENGTH_MM,
)
from kicad_pipeline.exceptions import SchematicError

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import Component, ProjectRequirements
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
    Point,
    PowerSymbol,
    Schematic,
    Stroke,
    SymbolInstance,
    TextProperty,
    Wire,
)
from kicad_pipeline.schematic.placement import layout_schematic
from kicad_pipeline.schematic.wiring import route_net
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
) -> SymbolInstance:
    """Create a placed :class:`SymbolInstance` for *component*.

    Args:
        component: Source component.
        lib_id: KiCad library identifier string.
        position: Placement position on the schematic canvas.

    Returns:
        A :class:`SymbolInstance` with a fresh UUID and text properties.
    """
    ref_prop = TextProperty(
        text=component.ref,
        position=Point(x=position.x, y=position.y - 2.54),
    )
    val_prop = TextProperty(
        text=component.value,
        position=Point(x=position.x, y=position.y + 2.54),
    )
    return SymbolInstance(
        lib_id=lib_id,
        ref=component.ref,
        value=component.value,
        footprint=component.footprint,
        position=position,
        lcsc=component.lcsc,
        uuid=_new_uuid(),
        ref_property=ref_prop,
        value_property=val_prop,
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


def _make_power_symbols(
    power_net_names: list[str],
    base_x: float = 10.0,
    base_y: float = 10.0,
) -> list[PowerSymbol]:
    """Create :class:`PowerSymbol` instances for the given net names.

    Symbols are spaced 20 mm apart horizontally.

    Args:
        power_net_names: List of power net names to instantiate.
        base_x: X origin for the first power symbol.
        base_y: Y origin for the first power symbol.

    Returns:
        List of :class:`PowerSymbol` objects.
    """
    symbols: list[PowerSymbol] = []
    for idx, name in enumerate(power_net_names):
        lib_id = _POWER_LIB_IDS.get(name, f"power:{name}")
        rotation = 270.0 if name in _GND_NETS else 0.0
        sym = PowerSymbol(
            lib_id=lib_id,
            position=Point(x=base_x + idx * 20.0, y=base_y),
            ref=f"#PWR0{idx + 1:02d}",
            value=name,
            rotation=rotation,
            uuid=_new_uuid(),
        )
        symbols.append(sym)
    return symbols


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

    log.info(
        "build_schematic: %d components, %d nets",
        len(requirements.components),
        len(requirements.nets),
    )

    # ------------------------------------------------------------------
    # Step 1: Derive feature map from FeatureBlocks
    # ------------------------------------------------------------------
    feature_map: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            feature_map[ref] = fb.name

    # ------------------------------------------------------------------
    # Step 2: Compute positions
    # ------------------------------------------------------------------
    all_refs = [c.ref for c in requirements.components]
    positions = layout_schematic(all_refs, feature_map)

    # ------------------------------------------------------------------
    # Step 3: Build lib_symbols + symbol instances
    # ------------------------------------------------------------------
    lib_symbols_list: list[LibSymbol] = []
    symbols_list: list[SymbolInstance] = []
    seen_lib_ids: set[str] = set()

    for comp in requirements.components:
        lib_id = _component_lib_id(comp)
        if lib_id not in seen_lib_ids:
            lib_sym = _make_lib_symbol(lib_id, comp)
            lib_symbols_list.append(lib_sym)
            seen_lib_ids.add(lib_id)

        pos = positions.get(comp.ref, Point(x=0.0, y=0.0))
        inst = _make_symbol_instance(comp, lib_id, pos)
        symbols_list.append(inst)

    # ------------------------------------------------------------------
    # Step 4: Build pin-position map for wire routing
    # ------------------------------------------------------------------
    # For simplicity: place each pin at the symbol origin offset by pin index
    pin_positions: dict[tuple[str, str], Point] = {}
    for comp in requirements.components:
        sym_pos = positions.get(comp.ref, Point(x=0.0, y=0.0))
        for idx, pin in enumerate(comp.pins):
            pin_x = sym_pos.x - SCHEMATIC_PIN_LENGTH_MM - 5.08
            pin_y = sym_pos.y + idx * 2.54
            pin_positions[(comp.ref, pin.number)] = Point(x=pin_x, y=pin_y)

    # ------------------------------------------------------------------
    # Step 5: Route nets
    # ------------------------------------------------------------------
    all_wires: list[Wire] = []
    all_junctions: list[Junction] = []
    all_global_labels: list[GlobalLabel] = []
    all_local_labels: list[Label] = []

    for net in requirements.nets:
        ws, js, gls, ls = route_net(net, pin_positions, use_global_labels=True)
        all_wires.extend(ws)
        all_junctions.extend(js)
        all_global_labels.extend(gls)
        all_local_labels.extend(ls)

    # ------------------------------------------------------------------
    # Step 6: Power symbols
    # ------------------------------------------------------------------
    power_net_names = _collect_power_nets(requirements)
    power_syms = _make_power_symbols(power_net_names)

    log.info(
        "build_schematic complete: %d symbols, %d wires, %d labels, %d power syms",
        len(symbols_list),
        len(all_wires),
        len(all_global_labels) + len(all_local_labels),
        len(power_syms),
    )

    return Schematic(
        lib_symbols=tuple(lib_symbols_list),
        symbols=tuple(symbols_list),
        power_symbols=tuple(power_syms),
        wires=tuple(all_wires),
        junctions=tuple(all_junctions),
        no_connects=(),
        labels=tuple(all_local_labels),
        global_labels=tuple(all_global_labels),
        version=KICAD_SCH_VERSION,
        generator=KICAD_GENERATOR,
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
    if sym.extends is not None:
        body.append(["extends", sym.extends])

    # Unit body sub-symbol
    unit_body: list[SExpNode] = ["symbol", f"{sym.lib_id}_0_1"]
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

    # Pin sub-symbol
    pin_body: list[SExpNode] = ["symbol", f"{sym.lib_id}_1_1"]
    for pin in sym.pins:
        pin_body.append(_lib_pin_sexp(pin))
    body.append(pin_body)

    return body


def _symbol_instance_sexp(inst: SymbolInstance) -> SExpNode:
    """Serialise a :class:`SymbolInstance` to a KiCad ``(symbol ...)`` node.

    Args:
        inst: Placed symbol instance.

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
    return node


def _wire_sexp(wire: Wire) -> SExpNode:
    """Serialise a :class:`Wire` to a KiCad ``(wire ...)`` node.

    Args:
        wire: Wire segment.

    Returns:
        ``SExpNode`` list.
    """
    return [
        "wire",
        ["start", wire.start.x, wire.start.y],
        ["end", wire.end.x, wire.end.y],
        _stroke_sexp(wire.stroke),
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


def _power_symbol_sexp(ps: PowerSymbol) -> SExpNode:
    """Serialise a :class:`PowerSymbol` to a KiCad ``(symbol ...)`` node.

    Args:
        ps: Power symbol instance.

    Returns:
        ``SExpNode`` list.
    """
    return [
        "symbol",
        ["lib_id", ps.lib_id],
        ["at", ps.position.x, ps.position.y, int(ps.rotation)],
        ["unit", 1],
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
    ]


def schematic_to_sexp(schematic: Schematic) -> SExpNode:
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
    root: list[SExpNode] = [
        "kicad_sch",
        ["version", schematic.version],
        ["generator", schematic.generator],
        ["paper", schematic.paper],
    ]

    # lib_symbols section
    lib_syms_node: list[SExpNode] = ["lib_symbols"]
    for lib_sym in schematic.lib_symbols:
        lib_syms_node.append(_lib_symbol_sexp(lib_sym))
    root.append(lib_syms_node)

    # Symbol instances (regular + power)
    for inst in schematic.symbols:
        root.append(_symbol_instance_sexp(inst))
    for ps in schematic.power_symbols:
        root.append(_power_symbol_sexp(ps))

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
    try:
        sexp = schematic_to_sexp(schematic)
        write_file(sexp, dest)
    except OSError:
        raise
    except Exception as exc:
        raise SchematicError(f"Failed to write schematic to {dest}: {exc}") from exc
    log.info("write_schematic: wrote %s", dest)

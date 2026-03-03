"""PCB builder orchestrator.

Combines placement, silkscreen, and S-expression serialisation into a single
pipeline entry point.  The primary public surface is:

* :func:`build_pcb` — assemble a :class:`~kicad_pipeline.models.pcb.PCBDesign`
  from :class:`~kicad_pipeline.models.requirements.ProjectRequirements`.
* :func:`pcb_to_sexp` — serialise a :class:`~kicad_pipeline.models.pcb.PCBDesign`
  to a KiCad S-expression tree.
* :func:`write_pcb` — write the S-expression tree to a ``.kicad_pcb`` file.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    KICAD_GENERATOR,
    KICAD_PCB_VERSION,
    LAYER_B_CU,
    LAYER_EDGE_CUTS,
    LAYER_F_CU,
    PCB_EDGE_CUTS_WIDTH_MM,
)
from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    FootprintText,
    Keepout,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
    ZoneFill,
    ZonePolygon,
)
from kicad_pipeline.pcb.placement import layout_pcb
from kicad_pipeline.pcb.silkscreen import add_silkscreen_to_footprint
from kicad_pipeline.sexp.writer import SExpNode, write_file

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import Component, ProjectRequirements

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Board defaults
# ---------------------------------------------------------------------------

_DEFAULT_BOARD_WIDTH_MM: float = 80.0
"""Default PCB width in mm (Hammond 1551K enclosure footprint)."""

_DEFAULT_BOARD_HEIGHT_MM: float = 40.0
"""Default PCB height in mm (Hammond 1551K enclosure footprint)."""

_MOUNTING_HOLE_INSET_MM: float = 3.5
"""Distance from board corner to mounting-hole centre in mm."""

_MOUNTING_HOLE_DIAMETER_MM: float = 3.2
"""Mounting hole drill diameter in mm (M3 screw)."""

_KEEPOUT_MARGIN_MM: float = 4.0
"""Radius of the keepout zone around each mounting hole in mm."""

_ANTENNA_KEEPOUT_WIDTH_MM: float = 15.0
"""Width of the no-copper keepout zone reserved for an ESP32 antenna in mm."""

_ANTENNA_KEEPOUT_HEIGHT_MM: float = 10.0
"""Height of the no-copper keepout zone reserved for an ESP32 antenna in mm."""

# Keywords that indicate the design contains an RF module requiring a keepout
_RF_KEYWORDS: frozenset[str] = frozenset({"esp32", "esp8266", "nrf", "cc3200", "rf"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_uuid() -> str:
    """Return a fresh RFC-4122 UUID string.

    Returns:
        Hyphenated UUID string.
    """
    return str(uuid.uuid4())


def _make_board_outline(width: float, height: float) -> BoardOutline:
    """Create a rectangular :class:`BoardOutline` for the given dimensions.

    Args:
        width: Board width in mm.
        height: Board height in mm.

    Returns:
        :class:`BoardOutline` with a closed rectangular polygon.
    """
    polygon = (
        Point(x=0.0, y=0.0),
        Point(x=width, y=0.0),
        Point(x=width, y=height),
        Point(x=0.0, y=height),
    )
    return BoardOutline(polygon=polygon, width=PCB_EDGE_CUTS_WIDTH_MM)


def _build_nets(requirements: ProjectRequirements) -> tuple[NetEntry, ...]:
    """Build the net list from *requirements*.

    Net 0 is always the unconnected sentinel ``""``; net 1 is always ``GND``.
    All other nets are derived from :attr:`ProjectRequirements.nets` in order.

    Args:
        requirements: Project requirements document.

    Returns:
        Tuple of :class:`NetEntry` objects starting with net 0 (empty) and
        net 1 (``GND``).
    """
    nets: list[NetEntry] = [NetEntry(number=0, name="")]
    seen: set[str] = {""}
    # GND always gets number 1
    nets.append(NetEntry(number=1, name="GND"))
    seen.add("GND")

    next_num = 2
    for net in requirements.nets:
        if net.name not in seen:
            nets.append(NetEntry(number=next_num, name=net.name))
            seen.add(net.name)
            next_num += 1

    return tuple(nets)


def _footprint_lib_id(component: Component) -> str:
    """Derive a KiCad footprint library identifier for *component*.

    Uses the component's ``footprint`` field when it contains a colon
    (already fully qualified, e.g. ``"R_SMD:R_0805_2012Metric"``).
    Otherwise wraps it as ``"kicad-ai:<footprint>"``.

    Args:
        component: The component to classify.

    Returns:
        Fully-qualified KiCad footprint ``lib_id`` string.
    """
    fp = component.footprint
    if ":" in fp:
        return fp
    return f"kicad-ai:{fp}"


def _make_footprint(
    component: Component,
    position: Point,
    net_lookup: dict[str, int],
) -> Footprint:
    """Build a minimal :class:`Footprint` for *component* at *position*.

    For each component pin that has a ``net`` attribute, a single SMD
    rectangular pad is created with the corresponding net number.

    Args:
        component: Source component.
        position: Board-coordinate placement position in mm.
        net_lookup: Mapping from net name to net number.

    Returns:
        A :class:`Footprint` with pads assigned to their respective nets.
    """
    lib_id = _footprint_lib_id(component)
    pads: list[Pad] = []

    for pin in component.pins:
        net_num: int | None = None
        net_name: str | None = None
        if pin.net is not None:
            net_num = net_lookup.get(pin.net)
            net_name = pin.net

        pad = Pad(
            number=pin.number,
            pad_type="smd",
            shape="rect",
            position=Point(x=0.0, y=0.0),  # relative to footprint origin
            size_x=1.5,
            size_y=1.5,
            layers=(LAYER_F_CU,),
            net_number=net_num,
            net_name=net_name,
            uuid=_new_uuid(),
        )
        pads.append(pad)

    fp = Footprint(
        lib_id=lib_id,
        ref=component.ref,
        value=component.value,
        position=position,
        rotation=0.0,
        layer=LAYER_F_CU,
        pads=tuple(pads),
        graphics=(),
        texts=(),
        lcsc=component.lcsc,
        uuid=_new_uuid(),
        attr="smd",
    )
    return fp


def _make_gnd_zones(
    board: BoardOutline,
    gnd_net_number: int,
) -> tuple[ZonePolygon, ZonePolygon]:
    """Create GND copper pours on both ``F.Cu`` and ``B.Cu``.

    Args:
        board: The board outline; its polygon is used as the zone boundary.
        gnd_net_number: Net number of the GND net.

    Returns:
        A pair ``(front_zone, back_zone)`` of :class:`ZonePolygon` objects.
    """
    front = ZonePolygon(
        net_number=gnd_net_number,
        net_name="GND",
        layer=LAYER_F_CU,
        name="GND_F",
        polygon=board.polygon,
        fill=ZoneFill.SOLID,
        uuid=_new_uuid(),
    )
    back = ZonePolygon(
        net_number=gnd_net_number,
        net_name="GND",
        layer=LAYER_B_CU,
        name="GND_B",
        polygon=board.polygon,
        fill=ZoneFill.SOLID,
        uuid=_new_uuid(),
    )
    return front, back


def _make_mounting_hole_keepouts(
    board_width: float,
    board_height: float,
    inset: float,
    radius: float,
) -> tuple[Keepout, ...]:
    """Create circular keepout zones at the four board corners.

    Each keepout is represented as a 12-point polygon approximating a circle.

    Args:
        board_width: Board width in mm.
        board_height: Board height in mm.
        inset: Distance from board corner to mounting hole centre in mm.
        radius: Radius of the keepout zone in mm.

    Returns:
        Tuple of four :class:`Keepout` objects, one per corner.
    """
    import math

    corners = [
        (inset, inset),
        (board_width - inset, inset),
        (board_width - inset, board_height - inset),
        (inset, board_height - inset),
    ]
    keepouts: list[Keepout] = []
    n_pts = 12
    for cx, cy in corners:
        pts = tuple(
            Point(
                x=cx + radius * math.cos(2.0 * math.pi * i / n_pts),
                y=cy + radius * math.sin(2.0 * math.pi * i / n_pts),
            )
            for i in range(n_pts)
        )
        keepouts.append(
            Keepout(
                polygon=pts,
                layers=(LAYER_F_CU, LAYER_B_CU),
                no_copper=True,
                no_vias=True,
                no_tracks=True,
                uuid=_new_uuid(),
            )
        )
    return tuple(keepouts)


def _make_antenna_keepout(
    board_width: float,
    width: float,
    height: float,
) -> Keepout:
    """Create a no-copper keepout zone for an RF antenna in the top-right corner.

    Args:
        board_width: Total board width in mm (used to position the keepout).
        width: Width of the antenna keepout zone in mm.
        height: Height of the antenna keepout zone in mm.

    Returns:
        A :class:`Keepout` covering the top-right area of the board.
    """
    x0 = board_width - width
    y0 = 0.0
    polygon = (
        Point(x=x0, y=y0),
        Point(x=board_width, y=y0),
        Point(x=board_width, y=height),
        Point(x=x0, y=height),
    )
    return Keepout(
        polygon=polygon,
        layers=(LAYER_F_CU, LAYER_B_CU),
        no_copper=True,
        no_vias=False,
        no_tracks=True,
        uuid=_new_uuid(),
    )


def _has_rf_module(requirements: ProjectRequirements) -> bool:
    """Return True if any component value suggests an RF / WiFi module.

    Args:
        requirements: Project requirements document.

    Returns:
        ``True`` when an RF-type component is detected.
    """
    for comp in requirements.components:
        val_lower = comp.value.lower()
        if any(kw in val_lower for kw in _RF_KEYWORDS):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pcb(
    requirements: ProjectRequirements,
    board_width_mm: float | None = None,
    board_height_mm: float | None = None,
) -> PCBDesign:
    """Build a complete :class:`PCBDesign` from *requirements*.

    Steps:

    1. Determine board dimensions from :attr:`~ProjectRequirements.mechanical`
       constraints or the supplied overrides (default 80 x 40 mm).
    2. Create the rectangular :class:`BoardOutline`.
    3. Build the :class:`NetEntry` list from requirements (GND always = net 1).
    4. Generate :class:`Footprint` objects for all components with net
       assignments applied to pads.
    5. Run zone-based PCB placement (:func:`~.placement.layout_pcb`) to assign
       board coordinates to each footprint.
    6. Add GND copper pours on ``F.Cu`` and ``B.Cu``.
    7. Add an antenna keepout in the top-right corner if an ESP32 / RF module
       is detected in the component list.
    8. Add mounting-hole keepout zones at the four board corners.
    9. Apply silkscreen labels to every footprint.
    10. Return the complete :class:`PCBDesign` (no tracks — added by autorouter).

    Args:
        requirements: Fully-populated project requirements document.
        board_width_mm: Override board width in mm.  If ``None``, the value
            from :attr:`~ProjectRequirements.mechanical` is used, or
            80 mm as the default.
        board_height_mm: Override board height in mm.  If ``None``, the value
            from :attr:`~ProjectRequirements.mechanical` is used, or
            40 mm as the default.

    Returns:
        A complete :class:`PCBDesign` ready for serialisation.

    Raises:
        PCBError: If the requirements contain no components, or if placement
            fails for any other reason.
    """
    if not requirements.components:
        raise PCBError("Cannot build PCB: requirements has no components")

    log.info(
        "build_pcb: %d components, %d nets",
        len(requirements.components),
        len(requirements.nets),
    )

    # ------------------------------------------------------------------
    # Step 1: Board dimensions
    # ------------------------------------------------------------------
    if board_width_mm is None:
        if requirements.mechanical is not None:
            board_width_mm = requirements.mechanical.board_width_mm
        else:
            board_width_mm = _DEFAULT_BOARD_WIDTH_MM

    if board_height_mm is None:
        if requirements.mechanical is not None:
            board_height_mm = requirements.mechanical.board_height_mm
        else:
            board_height_mm = _DEFAULT_BOARD_HEIGHT_MM

    log.info("build_pcb: board %.1f x %.1f mm", board_width_mm, board_height_mm)

    # ------------------------------------------------------------------
    # Step 2: Board outline
    # ------------------------------------------------------------------
    outline = _make_board_outline(board_width_mm, board_height_mm)

    # ------------------------------------------------------------------
    # Step 3: Nets
    # ------------------------------------------------------------------
    nets = _build_nets(requirements)
    net_lookup: dict[str, int] = {n.name: n.number for n in nets}

    # ------------------------------------------------------------------
    # Step 4: Footprints (without position — placement assigns positions)
    # ------------------------------------------------------------------
    pre_footprints: list[Footprint] = []
    for comp in requirements.components:
        fp = _make_footprint(comp, Point(x=0.0, y=0.0), net_lookup)
        pre_footprints.append(fp)

    # ------------------------------------------------------------------
    # Step 5: Layout placement
    # ------------------------------------------------------------------
    positions = layout_pcb(requirements, outline)

    # Apply positions to footprints
    footprints_with_pos: list[Footprint] = []
    for fp in pre_footprints:
        pos = positions.get(fp.ref, Point(x=0.0, y=0.0))
        fp_placed = Footprint(
            lib_id=fp.lib_id,
            ref=fp.ref,
            value=fp.value,
            position=pos,
            rotation=fp.rotation,
            layer=fp.layer,
            pads=fp.pads,
            graphics=fp.graphics,
            texts=fp.texts,
            lcsc=fp.lcsc,
            uuid=fp.uuid,
            attr=fp.attr,
        )
        footprints_with_pos.append(fp_placed)

    # ------------------------------------------------------------------
    # Step 6: GND pours
    # ------------------------------------------------------------------
    gnd_net_num = net_lookup.get("GND", 1)
    gnd_front, gnd_back = _make_gnd_zones(outline, gnd_net_num)
    zones: list[ZonePolygon] = [gnd_front, gnd_back]

    # ------------------------------------------------------------------
    # Step 7: Antenna keepout (ESP32 / RF modules)
    # ------------------------------------------------------------------
    keepouts: list[Keepout] = []
    if _has_rf_module(requirements):
        log.info("build_pcb: RF module detected — adding antenna keepout")
        antenna_ko = _make_antenna_keepout(
            board_width_mm,
            _ANTENNA_KEEPOUT_WIDTH_MM,
            _ANTENNA_KEEPOUT_HEIGHT_MM,
        )
        keepouts.append(antenna_ko)

    # ------------------------------------------------------------------
    # Step 8: Mounting-hole keepouts
    # ------------------------------------------------------------------
    corner_keepouts = _make_mounting_hole_keepouts(
        board_width_mm,
        board_height_mm,
        _MOUNTING_HOLE_INSET_MM,
        _KEEPOUT_MARGIN_MM,
    )
    keepouts.extend(corner_keepouts)

    # ------------------------------------------------------------------
    # Step 9: Silkscreen
    # ------------------------------------------------------------------
    final_footprints = [add_silkscreen_to_footprint(fp) for fp in footprints_with_pos]

    log.info(
        "build_pcb complete: %d footprints, %d nets, %d zones, %d keepouts",
        len(final_footprints),
        len(nets),
        len(zones),
        len(keepouts),
    )

    return PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=nets,
        footprints=tuple(final_footprints),
        tracks=(),
        vias=(),
        zones=tuple(zones),
        keepouts=tuple(keepouts),
        version=KICAD_PCB_VERSION,
        generator=KICAD_GENERATOR,
    )


# ---------------------------------------------------------------------------
# S-expression serialiser
# ---------------------------------------------------------------------------

# Standard KiCad layer table for a 2-layer board
_LAYER_TABLE: list[tuple[int, str, str]] = [
    (0, "F.Cu", "signal"),
    (31, "B.Cu", "signal"),
    (32, "B.Adhes", "user"),
    (33, "F.Adhes", "user"),
    (34, "B.Paste", "user"),
    (35, "F.Paste", "user"),
    (36, "B.SilkS", "user"),
    (37, "F.SilkS", "user"),
    (38, "B.Mask", "user"),
    (39, "F.Mask", "user"),
    (40, "Dwgs.User", "user"),
    (41, "Cmts.User", "user"),
    (44, "Edge.Cuts", "user"),
    (45, "Margin", "user"),
    (46, "B.CrtYd", "user"),
    (47, "F.CrtYd", "user"),
    (48, "B.Fab", "user"),
    (49, "F.Fab", "user"),
]


def _pad_sexp(pad: Pad) -> SExpNode:
    """Serialise a :class:`Pad` to a KiCad ``(pad ...)`` node.

    Args:
        pad: The pad to serialise.

    Returns:
        ``SExpNode`` list.
    """
    node: list[SExpNode] = [
        "pad",
        pad.number,
        pad.pad_type,
        pad.shape,
        ["at", pad.position.x, pad.position.y],
        ["size", pad.size_x, pad.size_y],
        ["layers", *pad.layers],
    ]
    if pad.net_number is not None and pad.net_name is not None:
        node.append(["net", pad.net_number, pad.net_name])
    if pad.uuid:
        node.append(["uuid", pad.uuid])
    return node


def _footprint_text_sexp(ft: FootprintText) -> SExpNode:
    """Serialise a :class:`FootprintText` to a KiCad ``(fp_text ...)`` node.

    Args:
        ft: The footprint text item to serialise.

    Returns:
        ``SExpNode`` list.
    """
    effects: list[SExpNode] = [
        "effects",
        ["font", ["size", ft.effects_size, ft.effects_size]],
    ]
    if ft.hidden:
        effects.append(["hide", "yes"])

    node: list[SExpNode] = [
        "fp_text",
        ft.text_type,
        ft.text,
        ["at", ft.position.x, ft.position.y],
        ["layer", ft.layer],
        effects,
    ]
    if ft.uuid:
        node.append(["uuid", ft.uuid])
    return node


def _footprint_sexp(fp: Footprint) -> SExpNode:
    """Serialise a :class:`Footprint` to a KiCad ``(footprint ...)`` node.

    Args:
        fp: The footprint to serialise.

    Returns:
        ``SExpNode`` list.
    """
    node: list[SExpNode] = [
        "footprint",
        fp.lib_id,
        ["layer", fp.layer],
        ["at", fp.position.x, fp.position.y, fp.rotation],
        ["attr", fp.attr],
    ]

    if fp.uuid:
        node.append(["uuid", fp.uuid])

    # Properties for ref and value
    node.append(
        [
            "property",
            "Reference",
            fp.ref,
            ["at", fp.position.x, fp.position.y - 1.5, 0],
            ["layer", "F.SilkS"],
            ["effects", ["font", ["size", 1.0, 1.0]]],
        ]
    )
    node.append(
        [
            "property",
            "Value",
            fp.value,
            ["at", fp.position.x, fp.position.y + 1.5, 0],
            ["layer", "F.Fab"],
            ["effects", ["font", ["size", 1.0, 1.0]], ["hide", "yes"]],
        ]
    )

    for text in fp.texts:
        node.append(_footprint_text_sexp(text))

    for pad in fp.pads:
        node.append(_pad_sexp(pad))

    return node


def _outline_sexp(outline: BoardOutline) -> list[SExpNode]:
    """Serialise a :class:`BoardOutline` to ``(gr_line ...)`` nodes.

    Generates one line segment per consecutive pair of polygon points,
    closing the polygon with a final segment from the last point back to the
    first.

    Args:
        outline: Board outline polygon.

    Returns:
        List of ``SExpNode`` lists, one per edge segment.
    """
    nodes: list[SExpNode] = []
    pts = outline.polygon
    n = len(pts)
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        nodes.append(
            [
                "gr_line",
                ["start", p0.x, p0.y],
                ["end", p1.x, p1.y],
                ["layer", LAYER_EDGE_CUTS],
                ["width", outline.width],
            ]
        )
    return nodes


def _zone_sexp(zone: ZonePolygon) -> SExpNode:
    """Serialise a :class:`ZonePolygon` to a KiCad ``(zone ...)`` node.

    Args:
        zone: Copper pour zone to serialise.

    Returns:
        ``SExpNode`` list.
    """
    pts_node: list[SExpNode] = ["pts"]
    for pt in zone.polygon:
        pts_node.append(["xy", pt.x, pt.y])

    node: list[SExpNode] = [
        "zone",
        ["net", zone.net_number],
        ["net_name", zone.net_name],
        ["layer", zone.layer],
        ["name", zone.name],
        ["hatch", "edge", 0.508],
        ["connect_pads", ["clearance", zone.min_thickness]],
        ["min_thickness", zone.min_thickness],
        [
            "fill",
            "yes",
            ["mode", zone.fill.value],
            ["thermal_gap", zone.thermal_relief_gap],
            ["thermal_bridge_width", zone.thermal_relief_bridge],
        ],
        ["polygon", pts_node],
    ]
    if zone.uuid:
        node.append(["uuid", zone.uuid])
    return node


def _keepout_sexp(keepout: Keepout) -> SExpNode:
    """Serialise a :class:`Keepout` to a KiCad ``(zone ...)`` keepout node.

    Args:
        keepout: Keepout zone to serialise.

    Returns:
        ``SExpNode`` list.
    """
    pts_node: list[SExpNode] = ["pts"]
    for pt in keepout.polygon:
        pts_node.append(["xy", pt.x, pt.y])

    rules: list[SExpNode] = ["keepout"]
    if keepout.no_copper:
        rules.append(["copper", "not_allowed"])
    if keepout.no_vias:
        rules.append(["vias", "not_allowed"])
    if keepout.no_tracks:
        rules.append(["tracks", "not_allowed"])

    node: list[SExpNode] = [
        "zone",
        ["net", 0],
        ["net_name", ""],
        ["layers", *keepout.layers],
        ["hatch", "edge", 0.508],
        rules,
        ["polygon", pts_node],
    ]
    if keepout.uuid:
        node.append(["uuid", keepout.uuid])
    return node


def pcb_to_sexp(design: PCBDesign) -> SExpNode:
    """Serialise a :class:`PCBDesign` to a KiCad S-expression tree.

    Output structure::

        (kicad_pcb (version 20231120) (generator "kicad-ai-pipeline")
          (general (thickness 1.6))
          (paper "A4")
          (layers
            (0 "F.Cu" signal)
            (31 "B.Cu" signal)
            ...
          )
          (setup (stackup ...) (pcbplotparams ...))
          (net 0 "")
          (net 1 "GND")
          ...
          (footprint "R_0805" (layer "F.Cu") (at x y rot) ...)
          (gr_line (start ...) (end ...) (layer "Edge.Cuts") (width 0.05))
          (zone (net 1) (net_name "GND") (layer "F.Cu") ...)
        )

    Args:
        design: The PCB design to serialise.

    Returns:
        A nested :data:`~kicad_pipeline.sexp.writer.SExpNode` list
        representing the root ``(kicad_pcb ...)`` expression.
    """
    root: list[SExpNode] = [
        "kicad_pcb",
        ["version", design.version],
        ["generator", design.generator],
        ["general", ["thickness", 1.6]],
        ["paper", "A4"],
    ]

    # Layers
    layers_node: list[SExpNode] = ["layers"]
    for layer_num, layer_name, layer_type in _LAYER_TABLE:
        layers_node.append([layer_num, layer_name, layer_type])
    root.append(layers_node)

    # Setup
    root.append(
        [
            "setup",
            ["pad_drill_adjust_back_out", "no"],
            [
                "pcbplotparams",
                ["layerselection", "0x00010fc_ffffffff"],
                ["outputdirectory", ""],
            ],
        ]
    )

    # Nets
    for net in design.nets:
        root.append(["net", net.number, net.name])

    # Footprints
    for fp in design.footprints:
        root.append(_footprint_sexp(fp))

    # Board outline
    for line in _outline_sexp(design.outline):
        root.append(line)

    # Copper zones
    for zone in design.zones:
        root.append(_zone_sexp(zone))

    # Keepout zones
    for keepout in design.keepouts:
        root.append(_keepout_sexp(keepout))

    return root


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_pcb(design: PCBDesign, path: str | Path) -> None:
    """Serialise *design* and write it to a ``.kicad_pcb`` file.

    Args:
        design: The PCB design to write.
        path: Destination file path.  The parent directory must exist.

    Raises:
        PCBError: If serialisation fails for any reason.
        OSError: If the file cannot be written.
    """
    dest = Path(path)
    log.info("write_pcb → %s", dest)
    try:
        sexp = pcb_to_sexp(design)
        write_file(sexp, dest)
    except OSError:
        raise
    except Exception as exc:
        raise PCBError(f"Failed to write PCB to {dest}: {exc}") from exc
    log.info("write_pcb: wrote %s", dest)

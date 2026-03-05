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

import datetime
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
    ZONE_CLEARANCE_DEFAULT_MM,
    ZONE_MIN_THICKNESS_MM,
)
from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    FootprintArc,
    FootprintCircle,
    FootprintLine,
    FootprintText,
    Keepout,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
    Track,
    Via,
    ZoneFill,
    ZonePolygon,
)
from kicad_pipeline.pcb.board_templates import get_template
from kicad_pipeline.pcb.footprints import (
    estimate_footprint_size,
    footprint_for_component,
    make_mounting_hole,
)
from kicad_pipeline.pcb.netclasses import classify_nets
from kicad_pipeline.pcb.placement import LayoutResult, layout_pcb
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

_KEEPOUT_MARGIN_MM: float = 3.0
"""Radius of the keepout zone around each mounting hole in mm.

Must be <= ``_MOUNTING_HOLE_INSET_MM`` (3.5 mm) to prevent keepout
circles from extending past the board edge.
"""

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


def _make_board_outline(
    width: float,
    height: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    corner_radius_mm: float = 0.0,
) -> BoardOutline:
    """Create a rectangular :class:`BoardOutline` for the given dimensions.

    Args:
        width: Board width in mm.
        height: Board height in mm.
        origin_x: X coordinate of the board origin in mm.
        origin_y: Y coordinate of the board origin in mm.
        corner_radius_mm: Corner rounding radius in mm (0 for sharp corners).

    Returns:
        :class:`BoardOutline` with a closed polygon (rounded if radius > 0).
    """
    import math as _m

    r = corner_radius_mm
    if r <= 0.0:
        # Sharp-cornered rectangle
        polygon = (
            Point(x=origin_x, y=origin_y),
            Point(x=origin_x + width, y=origin_y),
            Point(x=origin_x + width, y=origin_y + height),
            Point(x=origin_x, y=origin_y + height),
            Point(x=origin_x, y=origin_y),
        )
        return BoardOutline(polygon=polygon, width=PCB_EDGE_CUTS_WIDTH_MM)

    # Clamp radius to half the smaller dimension
    r = min(r, width / 2.0, height / 2.0)
    n_seg = 8  # arc segments per corner

    points: list[Point] = []
    # Corner centres and start angles (CW traversal)
    corners = [
        (origin_x + r, origin_y + r, _m.pi, _m.pi * 1.5),           # top-left
        (origin_x + width - r, origin_y + r, _m.pi * 1.5, 2 * _m.pi),  # top-right
        (origin_x + width - r, origin_y + height - r, 0.0, _m.pi * 0.5),  # bottom-right
        (origin_x + r, origin_y + height - r, _m.pi * 0.5, _m.pi),   # bottom-left
    ]
    for cx, cy, a_start, a_end in corners:
        for i in range(n_seg + 1):
            angle = a_start + (a_end - a_start) * i / n_seg
            points.append(Point(
                x=round(cx + r * _m.cos(angle), 6),
                y=round(cy + r * _m.sin(angle), 6),
            ))

    # Explicitly close polygon (last point == first point)
    if points:
        points.append(points[0])

    return BoardOutline(polygon=tuple(points), width=PCB_EDGE_CUTS_WIDTH_MM)


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


def _apply_nets_to_footprint(
    fp: Footprint,
    component: Component,
    net_lookup: dict[str, int],
) -> Footprint:
    """Copy net assignments onto matching pads of a footprint.

    ``footprint_for_component`` generates pads without net assignments.
    This helper copies the net number/name from the component's pin
    definitions onto matching pad numbers.

    Args:
        fp: Footprint from ``footprint_for_component`` (no nets on pads).
        component: Source component with pin-level net info.
        net_lookup: Mapping from net name to net number.

    Returns:
        A new :class:`Footprint` with nets assigned to pads.
    """
    pin_net_map: dict[str, tuple[int | None, str | None]] = {}
    for pin in component.pins:
        if pin.net is not None:
            net_num = net_lookup.get(pin.net)
            pin_net_map[pin.number] = (net_num, pin.net)

    new_pads: list[Pad] = []
    for pad in fp.pads:
        if pad.number in pin_net_map:
            net_num, net_name = pin_net_map[pad.number]
            pad = Pad(
                number=pad.number,
                pad_type=pad.pad_type,
                shape=pad.shape,
                position=pad.position,
                size_x=pad.size_x,
                size_y=pad.size_y,
                layers=pad.layers,
                net_number=net_num,
                net_name=net_name,
                drill_diameter=pad.drill_diameter,
                roundrect_ratio=pad.roundrect_ratio,
                uuid=pad.uuid or _new_uuid(),
            )
        new_pads.append(pad)

    return Footprint(
        lib_id=fp.lib_id,
        ref=fp.ref,
        value=fp.value,
        position=fp.position,
        rotation=fp.rotation,
        layer=fp.layer,
        pads=tuple(new_pads),
        graphics=fp.graphics,
        texts=fp.texts,
        lcsc=fp.lcsc or component.lcsc,
        uuid=fp.uuid or _new_uuid(),
        attr=fp.attr,
    )


def _make_gnd_zones(
    board: BoardOutline,
    gnd_net_number: int,
    clearance_mm: float = ZONE_CLEARANCE_DEFAULT_MM,
    strategy: str = "both",
) -> tuple[ZonePolygon, ...]:
    """Create GND copper pours on ``F.Cu`` and/or ``B.Cu``.

    Args:
        board: The board outline; its polygon is used as the zone boundary.
        gnd_net_number: Net number of the GND net.
        clearance_mm: Zone-to-pad/track clearance in mm.
        strategy: Ground plane strategy.  ``"both"`` (default) places GND
            pours on both layers.  ``"back_only"`` places GND only on B.Cu,
            leaving F.Cu free for routing (recommended for designs with
            >20 nets or analog signals).

    Returns:
        Tuple of :class:`ZonePolygon` objects (1 or 2 zones).
    """
    back = ZonePolygon(
        net_number=gnd_net_number,
        net_name="GND",
        layer=LAYER_B_CU,
        name="GND_B",
        polygon=board.polygon,
        min_thickness=ZONE_MIN_THICKNESS_MM,
        fill=ZoneFill.SOLID,
        clearance_mm=clearance_mm,
        uuid=_new_uuid(),
    )
    if strategy == "back_only":
        log.info("build_pcb: GND plane on B.Cu only (back_only strategy)")
        return (back,)
    front = ZonePolygon(
        net_number=gnd_net_number,
        net_name="GND",
        layer=LAYER_F_CU,
        name="GND_F",
        polygon=board.polygon,
        min_thickness=ZONE_MIN_THICKNESS_MM,
        fill=ZoneFill.SOLID,
        clearance_mm=clearance_mm,
        uuid=_new_uuid(),
    )
    return (front, back)


def _make_mounting_hole_keepouts(
    board_width: float,
    board_height: float,
    inset: float,
    radius: float,
    mounting_positions: tuple[tuple[float, float], ...] | None = None,
) -> tuple[Keepout, ...]:
    """Create circular keepout zones around mounting holes.

    When *mounting_positions* is provided, keepouts are placed at those exact
    positions. Otherwise, keepouts are placed at 4-corner fallback positions
    using the *inset* parameter.

    Each keepout is represented as a 12-point polygon approximating a circle.

    Args:
        board_width: Board width in mm.
        board_height: Board height in mm.
        inset: Distance from board corner to mounting hole centre in mm
            (used only for 4-corner fallback).
        radius: Radius of the keepout zone in mm.
        mounting_positions: Explicit mounting hole centres ``(x, y)`` in mm.
            When provided, overrides the 4-corner fallback.

    Returns:
        Tuple of :class:`Keepout` objects, one per mounting hole.
    """
    import math

    if mounting_positions is not None:
        corners = list(mounting_positions)
    else:
        corners = [
            (inset, inset),
            (board_width - inset, inset),
            (board_width - inset, board_height - inset),
            (inset, board_height - inset),
        ]
    keepouts: list[Keepout] = []
    n_pts = 12
    for cx, cy in corners:
        points: list[Point] = [
            Point(
                x=cx + radius * math.cos(2.0 * math.pi * i / n_pts),
                y=cy + radius * math.sin(2.0 * math.pi * i / n_pts),
            )
            for i in range(n_pts)
        ]
        # Do NOT explicitly close — KiCad auto-closes polygons.
        # Explicit closure creates a zero-length edge → "malformed" warning.
        pts = tuple(points)
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
    # Do NOT explicitly close — KiCad auto-closes polygons for keepouts.
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


_GND_VIA_SPACING_MM: float = 8.0
"""Grid spacing for GND stitching vias in mm."""

_GND_VIA_SIZE_MM: float = 0.8
"""Pad diameter for GND stitching vias in mm."""

_GND_VIA_DRILL_MM: float = 0.4
"""Drill diameter for GND stitching vias in mm."""

_GND_VIA_EDGE_MARGIN_MM: float = 1.5
"""Minimum distance from board edge for stitching vias in mm."""

_GND_VIA_FP_CLEARANCE_MM: float = 0.5
"""Clearance from footprint bounding boxes for stitching vias in mm."""


def _make_gnd_stitching_vias(
    board_width_mm: float,
    board_height_mm: float,
    gnd_net_num: int,
    footprints: tuple[Footprint, ...],
    keepouts: tuple[Keepout, ...],
) -> tuple[Via, ...]:
    """Generate a grid of GND stitching vias across the board.

    Stitching vias connect F.Cu and B.Cu GND pours, preventing the back
    copper plane from floating.  Vias are placed on a regular grid and
    skipped where they would overlap footprint bounding boxes or keepout
    zones.

    Args:
        board_width_mm: Board width in mm.
        board_height_mm: Board height in mm.
        gnd_net_num: Net number assigned to GND.
        footprints: All placed footprints (used for bounding-box exclusion).
        keepouts: Keepout zones to avoid.

    Returns:
        Tuple of :class:`Via` instances.
    """
    import math as _m

    margin = _GND_VIA_EDGE_MARGIN_MM
    spacing = _GND_VIA_SPACING_MM
    clr = _GND_VIA_FP_CLEARANCE_MM

    # Pre-compute footprint bounding boxes (centre +/- half-size + clearance)
    fp_boxes: list[tuple[float, float, float, float]] = []
    for fp in footprints:
        # Estimate size from pads + graphics extent
        pad_xs = [fp.position.x + p.position.x for p in fp.pads] if fp.pads else [fp.position.x]
        pad_ys = [fp.position.y + p.position.y for p in fp.pads] if fp.pads else [fp.position.y]
        half_sx = [p.size_x / 2.0 for p in fp.pads] if fp.pads else [0.0]
        half_sy = [p.size_y / 2.0 for p in fp.pads] if fp.pads else [0.0]
        min_x = min(px - hs for px, hs in zip(pad_xs, half_sx, strict=True)) - clr
        max_x = max(px + hs for px, hs in zip(pad_xs, half_sx, strict=True)) + clr
        min_y = min(py - hs for py, hs in zip(pad_ys, half_sy, strict=True)) - clr
        max_y = max(py + hs for py, hs in zip(pad_ys, half_sy, strict=True)) + clr
        fp_boxes.append((min_x, min_y, max_x, max_y))

    # Pre-compute keepout bounding boxes
    ko_boxes: list[tuple[float, float, float, float]] = []
    for ko in keepouts:
        if ko.no_vias or ko.no_copper:
            xs = [p.x for p in ko.polygon]
            ys = [p.y for p in ko.polygon]
            ko_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    def _blocked(x: float, y: float) -> bool:
        """Return True if (x, y) is inside a footprint or keepout."""
        for bx0, by0, bx1, by1 in fp_boxes:
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                return True
        return any(
            bx0 <= x <= bx1 and by0 <= y <= by1
            for bx0, by0, bx1, by1 in ko_boxes
        )

    # Generate grid
    cols = _m.floor((board_width_mm - 2 * margin) / spacing) + 1
    rows = _m.floor((board_height_mm - 2 * margin) / spacing) + 1
    x_start = margin + (board_width_mm - 2 * margin - (cols - 1) * spacing) / 2.0
    y_start = margin + (board_height_mm - 2 * margin - (rows - 1) * spacing) / 2.0

    vias: list[Via] = []
    for r in range(rows):
        for c in range(cols):
            vx = round(x_start + c * spacing, 3)
            vy = round(y_start + r * spacing, 3)
            if not _blocked(vx, vy):
                vias.append(Via(
                    position=Point(vx, vy),
                    drill=_GND_VIA_DRILL_MM,
                    size=_GND_VIA_SIZE_MM,
                    layers=(LAYER_F_CU, LAYER_B_CU),
                    net_number=gnd_net_num,
                    uuid=_new_uuid(),
                ))

    log.info("build_pcb: generated %d GND stitching vias", len(vias))
    return tuple(vias)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pcb(
    requirements: ProjectRequirements,
    board_width_mm: float | None = None,
    board_height_mm: float | None = None,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    board_template: str | None = None,
    auto_route: bool = True,
) -> PCBDesign:
    """Build a complete :class:`PCBDesign` from *requirements*.

    Steps:

    1. Determine board dimensions from *board_template*, mechanical
       constraints, or the supplied overrides (default 80 x 40 mm).
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
        origin_x: X coordinate of the board origin in mm (default 0.0).
        origin_y: Y coordinate of the board origin in mm (default 0.0).
        board_template: Optional board template name (e.g. ``"RPI_HAT"``).
            When provided, uses template dimensions and fixed component
            positions.

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
    # Step 0: Board template (if specified)
    # ------------------------------------------------------------------
    fixed_positions: dict[str, tuple[float, float, float]] | None = None
    corner_radius_mm: float = 0.0
    template_mounting_positions: tuple[tuple[float, float], ...] | None = None
    template_mounting_diameter: float | None = None
    tmpl_obj: object | None = None
    if board_template is not None:
        tmpl = get_template(board_template)
        tmpl_obj = tmpl
        log.info("build_pcb: using board template '%s'", tmpl.name)
        if board_width_mm is None:
            board_width_mm = tmpl.board_width_mm
        if board_height_mm is None:
            board_height_mm = tmpl.board_height_mm
        corner_radius_mm = tmpl.corner_radius_mm
        # Extract mounting hole positions from template
        if tmpl.mounting_holes:
            template_mounting_positions = tuple(
                (h.x_mm, h.y_mm) for h in tmpl.mounting_holes
            )
            template_mounting_diameter = tmpl.mounting_holes[0].diameter_mm
        # Extract fixed component positions from template
        if tmpl.fixed_components:
            fixed_positions = {}
            for fc in tmpl.fixed_components:
                # Match template ref_pattern against actual component refs
                for comp in requirements.components:
                    if comp.ref == fc.ref_pattern:
                        fixed_positions[comp.ref] = (fc.x_mm, fc.y_mm, fc.rotation)
                        break

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
    outline = _make_board_outline(
        board_width_mm, board_height_mm, origin_x, origin_y,
        corner_radius_mm=corner_radius_mm,
    )

    # ------------------------------------------------------------------
    # Step 3: Nets
    # ------------------------------------------------------------------
    nets = _build_nets(requirements)
    net_lookup: dict[str, int] = {n.name: n.number for n in nets}

    # ------------------------------------------------------------------
    # Step 4: Footprints (without position — placement assigns positions)
    # Uses footprint_for_component from footprints.py for proper geometry,
    # then applies net assignments from component pins.
    # ------------------------------------------------------------------
    pre_footprints: list[Footprint] = []
    for comp in requirements.components:
        fp = footprint_for_component(comp.ref, comp.value, comp.footprint, comp.lcsc)
        fp = _apply_nets_to_footprint(fp, comp, net_lookup)
        pre_footprints.append(fp)

    # ------------------------------------------------------------------
    # Step 4b: Compute footprint sizes and auto-size board if needed
    # ------------------------------------------------------------------
    fp_sizes: dict[str, tuple[float, float]] = {}
    total_area = 0.0
    for comp in requirements.components:
        sz = estimate_footprint_size(comp.footprint)
        fp_sizes[comp.ref] = sz
        total_area += sz[0] * sz[1]

    # Only auto-size when no template constrains dimensions
    if tmpl_obj is None:
        import math as _math

        # Estimate minimum board area as 3x total footprint area
        min_board_area = total_area * 3.0
        # Maintain ~2:1 aspect ratio: w * h = area, h = w/2 → w = sqrt(2*area)
        min_width = _math.sqrt(min_board_area * 2.0)
        min_height = min_width / 2.0
        new_width = max(board_width_mm, min_width)
        new_height = max(board_height_mm, min_height)
        # Ensure largest footprint fits with at least 10mm margin on each side
        max_fp_w = max((s[0] for s in fp_sizes.values()), default=0.0)
        max_fp_h = max((s[1] for s in fp_sizes.values()), default=0.0)
        new_width = max(new_width, max_fp_w + 20.0)
        new_height = max(new_height, max_fp_h + 20.0)
        # Cap aspect ratio to 2.5:1 max
        if new_width > 2.5 * new_height:
            new_height = new_width / 2.0
        elif new_height > 2.5 * new_width:
            new_width = new_height / 2.0

        if new_width > board_width_mm or new_height > board_height_mm:
            board_width_mm = new_width
            board_height_mm = new_height
            outline = _make_board_outline(
                board_width_mm, board_height_mm, origin_x, origin_y,
                corner_radius_mm=corner_radius_mm,
            )
            log.info(
                "build_pcb: auto-sized board to %.1f x %.1f mm",
                board_width_mm,
                board_height_mm,
            )

    # ------------------------------------------------------------------
    # Step 4c: Create keepouts BEFORE placement so solver can avoid them
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

    # Mounting-hole keepouts
    mount_positions: tuple[tuple[float, float], ...] | None = template_mounting_positions
    mount_radius = _KEEPOUT_MARGIN_MM
    if (
        mount_positions is None
        and requirements.mechanical is not None
        and requirements.mechanical.mounting_hole_positions
    ):
        mount_positions = requirements.mechanical.mounting_hole_positions
    if template_mounting_diameter is not None:
        mount_radius = template_mounting_diameter / 2.0 + 1.0
    elif requirements.mechanical is not None:
        mount_radius = requirements.mechanical.mounting_hole_diameter_mm / 2.0 + 1.0

    corner_keepouts = _make_mounting_hole_keepouts(
        board_width_mm,
        board_height_mm,
        _MOUNTING_HOLE_INSET_MM,
        mount_radius,
        mounting_positions=mount_positions,
    )
    keepouts.extend(corner_keepouts)

    # ------------------------------------------------------------------
    # Step 5: Layout placement (with keepouts available to solver)
    # ------------------------------------------------------------------
    layout_result: LayoutResult = layout_pcb(
        requirements, outline, footprint_sizes=fp_sizes,
        fixed_positions=fixed_positions,
        board_template=tmpl_obj,
        keepouts=tuple(keepouts),
    )

    # Apply positions and rotations to footprints
    footprints_with_pos: list[Footprint] = []
    for fp in pre_footprints:
        pos = layout_result.positions.get(fp.ref, Point(x=0.0, y=0.0))
        rot = layout_result.rotations.get(fp.ref, fp.rotation)
        fp_placed = Footprint(
            lib_id=fp.lib_id,
            ref=fp.ref,
            value=fp.value,
            position=pos,
            rotation=rot,
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
    # Step 5b: Net classification
    # ------------------------------------------------------------------
    netclasses = classify_nets(nets)

    # Zone clearance uses the safe default — KiCad enforces per-netclass
    # clearance on tracks separately, so zones should not inherit the max.
    zone_clearance = ZONE_CLEARANCE_DEFAULT_MM

    # ------------------------------------------------------------------
    # Step 6: GND pours
    # ------------------------------------------------------------------
    gnd_net_num = net_lookup.get("GND", 1)
    # Use "both" — F.Cu pour connects SMD GND pads, B.Cu pour provides
    # ground plane.  KiCad automatically keeps clearance from signal tracks.
    gnd_strategy = "both"
    gnd_zones = _make_gnd_zones(outline, gnd_net_num, zone_clearance, strategy=gnd_strategy)
    zones: list[ZonePolygon] = list(gnd_zones)

    # ------------------------------------------------------------------
    # Step 7-8: Keepouts already created in step 4c (before placement)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 9: Silkscreen
    # ------------------------------------------------------------------
    final_footprints = [add_silkscreen_to_footprint(fp) for fp in footprints_with_pos]

    # ------------------------------------------------------------------
    # Step 9b: Mounting hole footprints (NPTH, no net)
    # ------------------------------------------------------------------
    if template_mounting_positions and template_mounting_diameter is not None:
        for idx, (mx, my) in enumerate(template_mounting_positions, start=1):
            mh_ref = f"H{idx}"
            mh_fp = make_mounting_hole(mh_ref, drill_diameter=template_mounting_diameter)
            # Place at the template-defined position
            mh_fp = Footprint(
                lib_id=mh_fp.lib_id,
                ref=mh_fp.ref,
                value=mh_fp.value,
                position=Point(mx, my),
                rotation=0.0,
                layer=mh_fp.layer,
                pads=mh_fp.pads,
                graphics=mh_fp.graphics,
                texts=mh_fp.texts,
                uuid=mh_fp.uuid,
                attr=mh_fp.attr,
            )
            final_footprints.append(mh_fp)
        log.info(
            "build_pcb: added %d mounting hole footprints",
            len(template_mounting_positions),
        )

    # ------------------------------------------------------------------
    # Step 10: Autoroute (when enabled)
    # ------------------------------------------------------------------
    all_tracks: tuple[Track, ...] = ()
    all_vias: tuple[Via, ...] = ()
    if auto_route:
        from kicad_pipeline.pcb.netclasses import net_clearance_map, net_width_map
        from kicad_pipeline.pcb.netlist import build_netlist
        from kicad_pipeline.routing.grid_router import (
            collect_tracks,
            collect_vias,
            route_all_nets,
        )

        netlist = build_netlist(requirements)
        widths = net_width_map(netclasses)
        clearances = net_clearance_map(netclasses)
        route_results = route_all_nets(
            netlist, final_footprints,
            board_width_mm, board_height_mm,
            grid_step_mm=0.25,
            net_widths=widths,
            net_clearances=clearances,
            keepouts=tuple(keepouts),
        )
        all_tracks = collect_tracks(route_results)
        all_vias = collect_vias(route_results)
        routed = sum(1 for r in route_results if r.routed)
        unrouted = sum(1 for r in route_results if not r.routed)
        log.info(
            "build_pcb: autoroute complete — %d tracks, %d routed, %d unrouted",
            len(all_tracks), routed, unrouted,
        )

        # Note: GND pads on F.Cu connect to the B.Cu GND pour through
        # the zone fill (applied when opening in KiCad).  THT pads already
        # have plated holes.  SMD GND pads may show as "unconnected" in
        # DRC until zones are filled.

    # ------------------------------------------------------------------
    # Step 10b: GND stitching vias (connect F.Cu ↔ B.Cu ground planes)
    # ------------------------------------------------------------------
    gnd_vias = _make_gnd_stitching_vias(
        board_width_mm, board_height_mm, gnd_net_num,
        tuple(final_footprints), tuple(keepouts),
    )
    all_vias = all_vias + gnd_vias

    log.info(
        "build_pcb complete: %d footprints, %d nets, %d zones, %d keepouts, "
        "%d tracks, %d vias",
        len(final_footprints),
        len(nets),
        len(zones),
        len(keepouts),
        len(all_tracks),
        len(all_vias),
    )

    return PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=nets,
        footprints=tuple(final_footprints),
        tracks=all_tracks,
        vias=all_vias,
        zones=tuple(zones),
        keepouts=tuple(keepouts),
        netclasses=netclasses,
        version=KICAD_PCB_VERSION,
        generator=KICAD_GENERATOR,
        title=requirements.project.name,
        date=datetime.date.today().isoformat(),
        revision=requirements.project.revision,
        company=requirements.project.author or "",
    )


# ---------------------------------------------------------------------------
# S-expression serialiser
# ---------------------------------------------------------------------------

# Standard KiCad 9 layer table for a 2-layer board.
# KiCad 9 renumbered all layers; some have canonical aliases.
_LAYER_TABLE: list[tuple[int, str, str] | tuple[int, str, str, str]] = [
    (0, "F.Cu", "signal"),
    (2, "B.Cu", "signal"),
    (9, "F.Adhes", "user", "F.Adhesive"),
    (11, "B.Adhes", "user", "B.Adhesive"),
    (13, "F.Paste", "user"),
    (15, "B.Paste", "user"),
    (5, "F.SilkS", "user", "F.Silkscreen"),
    (7, "B.SilkS", "user", "B.Silkscreen"),
    (1, "F.Mask", "user"),
    (3, "B.Mask", "user"),
    (17, "Dwgs.User", "user", "User.Drawings"),
    (19, "Cmts.User", "user", "User.Comments"),
    (25, "Edge.Cuts", "user"),
    (27, "Margin", "user"),
    (31, "F.CrtYd", "user", "F.Courtyard"),
    (29, "B.CrtYd", "user", "B.Courtyard"),
    (33, "F.Fab", "user", "F.Fabrication"),
    (35, "B.Fab", "user", "B.Fabrication"),
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
    if pad.drill_diameter is not None and pad.drill_diameter > 0:
        node.append(["drill", pad.drill_diameter])
    if pad.roundrect_ratio is not None:
        node.append(["roundrect_rratio", pad.roundrect_ratio])
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

    # Properties for ref and value — relative to footprint origin
    node.append(
        [
            "property",
            "Reference",
            fp.ref,
            ["at", 0, -2.5, 0],
            ["layer", "F.SilkS"],
            ["effects", ["font", ["size", 1.0, 1.0]]],
        ]
    )
    node.append(
        [
            "property",
            "Value",
            fp.value,
            ["at", 0, 2.5, 0],
            ["layer", "F.Fab"],
            ["effects", ["font", ["size", 1.0, 1.0]], ["hide", "yes"]],
        ]
    )

    # KiCad 9: fp_text replaced by property entries (already emitted above).
    # Only emit fp_text for custom user text, not reference/value.
    for text in fp.texts:
        if text.text_type not in ("reference", "value"):
            node.append(_footprint_text_sexp(text))

    # Footprint graphics (courtyard, silkscreen, fab outlines)
    for graphic in fp.graphics:
        if isinstance(graphic, FootprintLine):
            g_line: list[SExpNode] = [
                "fp_line",
                ["start", graphic.start.x, graphic.start.y],
                ["end", graphic.end.x, graphic.end.y],
                ["layer", graphic.layer],
                ["width", graphic.width],
            ]
            if graphic.uuid:
                g_line.append(["uuid", graphic.uuid])
            node.append(g_line)
        elif isinstance(graphic, FootprintArc):
            g_arc: list[SExpNode] = [
                "fp_arc",
                ["start", graphic.start.x, graphic.start.y],
                ["mid", graphic.mid.x, graphic.mid.y],
                ["end", graphic.end.x, graphic.end.y],
                ["layer", graphic.layer],
                ["width", graphic.width],
            ]
            if graphic.uuid:
                g_arc.append(["uuid", graphic.uuid])
            node.append(g_arc)
        elif isinstance(graphic, FootprintCircle):
            g_circ: list[SExpNode] = [
                "fp_circle",
                ["center", graphic.center.x, graphic.center.y],
                ["end", graphic.end.x, graphic.end.y],
                ["layer", graphic.layer],
                ["width", graphic.width],
            ]
            if graphic.uuid:
                g_circ.append(["uuid", graphic.uuid])
            node.append(g_circ)

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
    # If polygon is explicitly closed (last == first), don't wrap around
    is_closed = (
        n > 2
        and abs(pts[0].x - pts[-1].x) < 1e-6
        and abs(pts[0].y - pts[-1].y) < 1e-6
    )
    edge_count = n - 1 if is_closed else n
    for i in range(edge_count):
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
    ]
    if zone.uuid:
        node.append(["uuid", zone.uuid])
    node.extend([
        ["hatch", "edge", 0.508],
        ["connect_pads", "yes", ["clearance", zone.clearance_mm]],
        ["min_thickness", zone.min_thickness],
        ["filled_areas_thickness", False],
        [
            "fill",
            ["thermal_gap", zone.thermal_relief_gap],
            ["thermal_bridge_width", zone.thermal_relief_bridge],
        ],
        ["polygon", pts_node],
    ])
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

    # KiCad 9 keepout format requires all five rule entries.
    rules: list[SExpNode] = [
        "keepout",
        ["copperpour", "not_allowed" if keepout.no_copper else "allowed"],
        ["footprints", "allowed"],
        ["pads", "allowed"],
        ["tracks", "not_allowed" if keepout.no_tracks else "allowed"],
        ["vias", "not_allowed" if keepout.no_vias else "allowed"],
    ]

    node: list[SExpNode] = [
        "zone",
        ["net", 0],
        ["net_name", ""],
        ["layers", *keepout.layers],
    ]
    if keepout.uuid:
        node.append(["uuid", keepout.uuid])
    node.extend([
        ["hatch", "edge", 0.508],
        rules,
        ["polygon", pts_node],
    ])
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
        ["generator_version", design.generator_version],
        ["general", ["thickness", 1.6], ["legacy_teardrops", False]],
        ["paper", "A4"],
    ]

    # Title block
    if design.title or design.date or design.revision or design.company:
        title_block: list[SExpNode] = ["title_block"]
        if design.title:
            title_block.append(["title", design.title])
        if design.date:
            title_block.append(["date", design.date])
        if design.revision:
            title_block.append(["rev", design.revision])
        if design.company:
            title_block.append(["company", design.company])
        root.append(title_block)

    # Layers
    layers_node: list[SExpNode] = ["layers"]
    for layer_entry in _LAYER_TABLE:
        layer_node: list[SExpNode] = [layer_entry[0], layer_entry[1], layer_entry[2]]
        if len(layer_entry) > 3:
            layer_node.append(layer_entry[3])
        layers_node.append(layer_node)
    root.append(layers_node)

    # Setup
    root.append(
        [
            "setup",
            ["pad_to_mask_clearance", 0],
            ["allow_soldermask_bridges_in_footprints", False],
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

    # Tracks (segments)
    for track in design.tracks:
        seg: list[SExpNode] = [
            "segment",
            ["start", track.start.x, track.start.y],
            ["end", track.end.x, track.end.y],
            ["width", track.width],
            ["layer", track.layer],
            ["net", track.net_number],
        ]
        if track.uuid:
            seg.append(["uuid", track.uuid])
        root.append(seg)

    # Vias
    for via in design.vias:
        via_node: list[SExpNode] = [
            "via",
            ["at", via.position.x, via.position.y],
            ["size", via.size],
            ["drill", via.drill],
            ["layers", *via.layers],
            ["net", via.net_number],
        ]
        if via.uuid:
            via_node.append(["uuid", via.uuid])
        root.append(via_node)

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

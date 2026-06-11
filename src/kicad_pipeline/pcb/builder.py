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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    KICAD_GENERATOR,
    KICAD_PCB_VERSION,
    LAYER_B_CU,
    LAYER_EDGE_CUTS,
    LAYER_F_CU,
    ZONE_CLEARANCE_DEFAULT_MM,
)
from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    FootprintArc,
    FootprintCircle,
    FootprintKeepout,
    FootprintLine,
    FootprintText,
    Keepout,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
    Track,
    Via,
    ZonePolygon,
)
from kicad_pipeline.pcb.board_templates import get_template
from kicad_pipeline.pcb.footprints import (
    compute_footprint_bbox,
    estimate_footprint_size,
    footprint_for_component,
    make_mounting_hole,
)
from kicad_pipeline.pcb.keepout_builder import (
    ANTENNA_KEEPOUT_HEIGHT_MM as _ANTENNA_KEEPOUT_HEIGHT_MM,
)
from kicad_pipeline.pcb.keepout_builder import (
    ANTENNA_KEEPOUT_WIDTH_MM as _ANTENNA_KEEPOUT_WIDTH_MM,
)
from kicad_pipeline.pcb.keepout_builder import (
    KEEPOUT_MARGIN_MM as _KEEPOUT_MARGIN_MM,
)
from kicad_pipeline.pcb.keepout_builder import (
    MOUNTING_HOLE_DIAMETER_MM as _MOUNTING_HOLE_DIAMETER_MM,
)
from kicad_pipeline.pcb.keepout_builder import (
    MOUNTING_HOLE_INSET_MM as _MOUNTING_HOLE_INSET_MM,
)
from kicad_pipeline.pcb.keepout_builder import (
    RF_KEYWORDS as _RF_KEYWORDS,
)
from kicad_pipeline.pcb.keepout_builder import (
    has_rf_module as _has_rf_module,
)
from kicad_pipeline.pcb.keepout_builder import (
    make_antenna_keepout as _make_antenna_keepout,
)
from kicad_pipeline.pcb.keepout_builder import (
    make_mounting_hole_keepouts as _make_mounting_hole_keepouts,
)
from kicad_pipeline.pcb.keepout_builder import (
    make_rf_module_body_keepout as _make_rf_module_body_keepout,
)
from kicad_pipeline.pcb.netclasses import classify_nets
from kicad_pipeline.pcb.outline_builder import make_board_outline as _make_board_outline
from kicad_pipeline.pcb.pin_map import origin_to_centroid
from kicad_pipeline.pcb.placement import LayoutResult, layout_pcb, place_groups_off_board
from kicad_pipeline.pcb.silkscreen import (
    add_silkscreen_to_footprint,
)
from kicad_pipeline.pcb.silkscreen import (
    clamp_silk_to_board as _clamp_silk_to_board,
)
from kicad_pipeline.pcb.silkscreen import (
    resolve_silk_collisions as _resolve_silk_collisions,
)
from kicad_pipeline.pcb.zone_builder import (
    make_gnd_stitching_vias as _make_gnd_stitching_vias,
)
from kicad_pipeline.pcb.zone_builder import (
    make_gnd_zones as _make_gnd_zones,
)
from kicad_pipeline.sexp.writer import SExpNode, write_file

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import Component, ProjectRequirements
    from kicad_pipeline.pcb.board_templates import BoardTemplate

log = logging.getLogger(__name__)

# Track board-size warnings to avoid repeating the same message every build_pcb() call
_board_size_warned: set[tuple[int, int, int]] = set()

# Module-level storage for preserved edge cuts (board slots/cutouts)
# populated by build_pcb() and consumed by pcb_to_sexp().
_preserved_edge_cuts: list[tuple[Point, Point, float]] = []

# ---------------------------------------------------------------------------
# Board defaults
# ---------------------------------------------------------------------------

_DEFAULT_BOARD_WIDTH_MM: float = 80.0
"""Default PCB width in mm (Hammond 1551K enclosure footprint)."""

_DEFAULT_BOARD_HEIGHT_MM: float = 40.0
"""Default PCB height in mm (Hammond 1551K enclosure footprint)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_uuid() -> str:
    """Return a fresh RFC-4122 UUID string.

    Returns:
        Hyphenated UUID string.
    """
    return str(uuid.uuid4())




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


def _footprint_lib_id(component: Component, project_name: str | None = None) -> str:
    """Derive a KiCad footprint library identifier for *component*.

    When *project_name* is provided, all footprints use the project-local
    library prefix ``{project_name}:{footprint_name}``.  Otherwise falls
    back to the legacy behaviour (``kicad-ai:`` prefix for bare names).

    Args:
        component: The component to classify.
        project_name: When set, use as the library prefix for all footprints.

    Returns:
        Fully-qualified KiCad footprint ``lib_id`` string.
    """
    from kicad_pipeline.pcb.footprint_library import footprint_name_from_lib_id

    if project_name is not None:
        fp_name = footprint_name_from_lib_id(component.footprint)
        return f"{project_name}:{fp_name}"
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
        models=fp.models, datasheet=fp.datasheet, description=fp.description,
        fp_zones=fp.fp_zones,
    )




# ---------------------------------------------------------------------------
# build_pcb helper context
# ---------------------------------------------------------------------------


@dataclass
class _BuildContext:
    """Mutable state container threaded through build_pcb helper functions."""

    board_width_mm: float
    board_height_mm: float
    origin_x: float
    origin_y: float
    corner_radius_mm: float
    layer_count: int
    project_name: str | None
    outline: BoardOutline
    nets: tuple[NetEntry, ...]
    net_lookup: dict[str, int]
    keepouts: list[Keepout]
    zones: list[ZonePolygon]
    fixed_positions: dict[str, tuple[float, float, float]] | None
    layer_overrides: dict[str, str]
    template_mounting_positions: tuple[tuple[float, float], ...] | None
    template_mounting_diameter: float | None
    tmpl_obj: BoardTemplate | None
    preserved_ref_text_positions: dict[str, tuple[float, float, float]]
    has_rf: bool
    rf_pos: tuple[float, float, float] | None
    fp_sizes: dict[str, tuple[float, float]]
    fp_bboxes: dict[str, object]
    #: True when the board size came from the caller or requirements;
    #: False when it is a heuristic estimate (auto-size). Placement v2
    #: treats an ESTIMATE as advisory and shrink-to-fits instead — a
    #: heuristic cap once missed a feasible floorplan by 0.03mm.
    explicit_dimensions: bool = True


# ---------------------------------------------------------------------------
# build_pcb extracted helpers
# ---------------------------------------------------------------------------


def _resolve_board_template(
    requirements: ProjectRequirements,
    board_template: str | None,
    board_width_mm: float | None,
    board_height_mm: float | None,
) -> tuple[
    str | None,
    float | None,
    float | None,
    float,
    dict[str, tuple[float, float, float]] | None,
    dict[str, str],
    tuple[tuple[float, float], ...] | None,
    float | None,
    BoardTemplate | None,
]:
    """Resolve board template, extracting dimensions and fixed positions.

    Returns:
        (board_template, board_width_mm, board_height_mm, corner_radius_mm,
         fixed_positions, layer_overrides, template_mounting_positions,
         template_mounting_diameter, tmpl_obj)
    """
    fixed_positions: dict[str, tuple[float, float, float]] | None = None
    layer_overrides: dict[str, str] = {}
    corner_radius_mm: float = 0.0
    template_mounting_positions: tuple[tuple[float, float], ...] | None = None
    template_mounting_diameter: float | None = None
    tmpl_obj: BoardTemplate | None = None

    if board_template is None and requirements.mechanical is not None:
        from kicad_pipeline.pcb.board_templates import detect_template
        auto_tmpl = detect_template(requirements.mechanical)
        if auto_tmpl is not None:
            board_template = auto_tmpl.name
            log.info("build_pcb: auto-detected board template '%s'", board_template)

    if board_template is not None:
        tmpl = get_template(board_template)
        tmpl_obj = tmpl
        log.info("build_pcb: using board template '%s'", tmpl.name)
        if board_width_mm is None:
            board_width_mm = tmpl.board_width_mm
        if board_height_mm is None:
            board_height_mm = tmpl.board_height_mm
        corner_radius_mm = tmpl.corner_radius_mm
        if tmpl.mounting_holes:
            template_mounting_positions = tuple(
                (h.x_mm, h.y_mm) for h in tmpl.mounting_holes
            )
            template_mounting_diameter = tmpl.mounting_holes[0].diameter_mm
        if tmpl.fixed_components:
            fixed_positions = _match_template_fixed_components(
                tmpl, requirements,
            )
            for fc in tmpl.fixed_components:
                matched = _find_template_match(fc, requirements)
                if matched is not None and fc.layer != "F.Cu":
                    layer_overrides[matched] = fc.layer

    return (
        board_template, board_width_mm, board_height_mm, corner_radius_mm,
        fixed_positions, layer_overrides, template_mounting_positions,
        template_mounting_diameter, tmpl_obj,
    )


def _match_template_fixed_components(
    tmpl: BoardTemplate,
    requirements: ProjectRequirements,
) -> dict[str, tuple[float, float, float]]:
    """Match template fixed components to requirement components."""
    fixed_positions: dict[str, tuple[float, float, float]] = {}
    for fc in tmpl.fixed_components:
        matched_ref = _find_template_match(fc, requirements)
        if matched_ref is not None:
            fixed_positions[matched_ref] = (fc.x_mm, fc.y_mm, fc.rotation)
            log.info(
                "build_pcb: template fixed %s at (%.1f, %.1f) layer=%s",
                matched_ref, fc.x_mm, fc.y_mm, fc.layer,
            )
    return fixed_positions


def _find_template_match(
    fc: object,
    requirements: ProjectRequirements,
) -> str | None:
    """Find a component ref matching a template fixed component."""
    is_gpio = "GPIO" in fc.description.upper()  # type: ignore[union-attr]
    for comp in requirements.components:
        if comp.ref == fc.ref_pattern:  # type: ignore[union-attr]
            if is_gpio and len(comp.pins) < 10:
                continue
            return comp.ref
    if is_gpio:
        for comp in requirements.components:
            fp_upper = comp.footprint.upper()
            if "02X20" in fp_upper or "2X20" in fp_upper:
                return comp.ref
    return None


def _resolve_preserved_positions(
    preserve_from: str | Path | object | None,
    preserve_ref_text: bool,
    requirements: ProjectRequirements,
    fixed_positions: dict[str, tuple[float, float, float]] | None,
) -> tuple[
    dict[str, tuple[float, float, float]] | None,
    dict[str, tuple[float, float, float]],
]:
    """Extract positions from an existing PCB or IPC connection.

    Returns:
        (fixed_positions, preserved_ref_text_positions)
    """
    preserved_ref_text_positions: dict[str, tuple[float, float, float]] = {}
    if preserve_from is None:
        return fixed_positions, preserved_ref_text_positions

    from kicad_pipeline.pcb.position_extractor import (
        positions_from_source,
        ref_text_positions_from_source,
    )

    existing = positions_from_source(preserve_from)
    current_refs = {c.ref for c in requirements.components}
    if fixed_positions is None:
        fixed_positions = {}
    for ref, pos in existing.items():
        if ref in current_refs:
            fixed_positions[ref] = pos
    if preserve_ref_text:
        preserved_ref_text_positions = ref_text_positions_from_source(preserve_from)
    log.info(
        "build_pcb: preserved %d/%d positions, %d ref text positions",
        len(fixed_positions), len(existing), len(preserved_ref_text_positions),
    )
    return fixed_positions, preserved_ref_text_positions


def _resolve_board_dimensions(
    board_width_mm: float | None,
    board_height_mm: float | None,
    requirements: ProjectRequirements,
) -> tuple[float, float, bool]:
    """Determine board width/height from requirements or defaults.

    Returns:
        (board_width_mm, board_height_mm, explicit_dimensions)
    """
    explicit_dimensions = (
        (board_width_mm is not None and board_height_mm is not None)
        or requirements.mechanical is not None
    )
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
    return board_width_mm, board_height_mm, explicit_dimensions


def _auto_size_board(
    board_width_mm: float,
    board_height_mm: float,
    origin_x: float,
    origin_y: float,
    corner_radius_mm: float,
    fp_sizes: dict[str, tuple[float, float]],
    total_area: float,
) -> tuple[float, float, BoardOutline]:
    """Auto-size the board — grow if too small, shrink if too big.

    Target utilization: 30-60% of board area used by components.
    A 60x40mm board for 15 small components is wasteful and produces
    scattered, unprofessional layouts.

    Returns:
        (board_width_mm, board_height_mm, outline)
    """
    import math as _math

    # Account for mounting holes: 4 corners need ~4mm inset each = 8mm on each axis
    mh_inset = 4.0  # mounting hole center inset from edge
    mh_clearance = 5.0  # clearance around each mounting hole
    mh_reserved_per_axis = 2 * (mh_inset + mh_clearance)  # ~18mm reserved

    # Account for connector body overhang: terminal blocks extend ~5mm beyond pads
    conn_body_margin = 8.0  # extra margin for THT connector bodies

    min_board_area = total_area * 3.0  # ~33% utilization target
    max_board_area = total_area * 5.0  # ~20% utilization floor
    min_width = _math.sqrt(min_board_area * 2.0)
    min_height = min_width / 2.0
    max_fp_w = max((s[0] for s in fp_sizes.values()), default=0.0)
    max_fp_h = max((s[1] for s in fp_sizes.values()), default=0.0)

    # Has THT connectors? They need extra width for body overhang
    has_tht_connector = any(
        "terminal" in k.lower() or "conn" in k.lower() or "pinheader" in k.lower()
        for k in fp_sizes
    )

    # Absolute minimums: mounting holes + largest component + margins
    # A board smaller than 50x30mm can't fit mounting holes + connectors + ICs
    abs_min_w = max(50.0, max_fp_w + mh_reserved_per_axis + conn_body_margin)
    abs_min_h = max(30.0, max_fp_h + mh_reserved_per_axis)
    if has_tht_connector:
        abs_min_w = max(abs_min_w, 55.0)  # THT connectors need board edge access + body clearance

    # Grow if too small
    new_width = max(board_width_mm, min_width, abs_min_w)
    new_height = max(board_height_mm, min_height, abs_min_h)

    # Shrink if too big — board area > 5x component area is wasteful
    current_area = new_width * new_height
    if current_area > max_board_area and total_area > 10.0:
        # Shrink toward ~3.5x component area, but respect absolute minimums
        target_area = total_area * 3.5
        # Ensure target area isn't below absolute minimum
        target_area = max(target_area, abs_min_w * abs_min_h)
        aspect = new_width / new_height if new_height > 0 else 1.5
        shrunk_height = _math.sqrt(target_area / aspect)
        shrunk_width = shrunk_height * aspect
        # Enforce absolute minimums
        shrunk_width = max(shrunk_width, abs_min_w)
        shrunk_height = max(shrunk_height, abs_min_h)
        if shrunk_width < new_width or shrunk_height < new_height:
            new_width = min(new_width, shrunk_width)
            new_height = min(new_height, shrunk_height)
            log.info(
                "build_pcb: auto-shrunk board to %.1f x %.1f mm "
                "(component area %.0f mm², was %.0f x %.0f)",
                new_width, new_height, total_area,
                board_width_mm, board_height_mm,
            )

    # Aspect ratio constraint
    if new_width > 2.5 * new_height:
        new_height = new_width / 2.0
    elif new_height > 2.5 * new_width:
        new_width = new_height / 2.0

    if abs(new_width - board_width_mm) > 0.5 or abs(new_height - board_height_mm) > 0.5:
        board_width_mm = new_width
        board_height_mm = new_height
        log.info(
            "build_pcb: auto-sized board to %.1f x %.1f mm",
            board_width_mm, board_height_mm,
        )

    outline = _make_board_outline(
        board_width_mm, board_height_mm, origin_x, origin_y,
        corner_radius_mm=corner_radius_mm,
    )
    return board_width_mm, board_height_mm, outline


def _warn_board_size(
    board_width_mm: float,
    board_height_mm: float,
    total_area: float,
) -> None:
    """Emit a warning if board area is too small for component footprints."""
    board_area = board_width_mm * board_height_mm
    min_area = total_area * 3.0
    if total_area > 0.0 and board_area < min_area:
        import math as _math
        suggested_w = _math.sqrt(min_area * (board_width_mm / board_height_mm))
        suggested_h = min_area / suggested_w
        _size_key = (round(board_width_mm), round(board_height_mm), round(total_area))
        if _size_key not in _board_size_warned:
            _board_size_warned.add(_size_key)
            log.warning(
                "build_pcb: board %.0fx%.0fmm (%.0f mm^2) may be too small for "
                "%.0f mm^2 of footprints (suggest %.0fx%.0fmm)",
                board_width_mm, board_height_mm, board_area,
                total_area, suggested_w, suggested_h,
            )


def _build_pre_footprints(
    requirements: ProjectRequirements,
    net_lookup: dict[str, int],
    layer_overrides: dict[str, str],
    project_name: str | None,
) -> list[Footprint]:
    """Create footprints for all components (without placement positions)."""
    pre_footprints: list[Footprint] = []
    for comp in requirements.components:
        comp_layer = layer_overrides.get(comp.ref, LAYER_F_CU)
        fp = footprint_for_component(
            comp.ref, comp.value, comp.footprint, comp.lcsc, layer=comp_layer,
            pins=comp.pins,
        )
        fp = _apply_nets_to_footprint(fp, comp, net_lookup)
        custom_props: list[tuple[str, str]] = []
        if comp.placement_group:
            custom_props.append(("PlacementGroup", comp.placement_group))
        if comp.placement_near:
            custom_props.append(("PlacementNear", comp.placement_near))
        if comp.placement_order is not None:
            custom_props.append(("PlacementOrder", str(comp.placement_order)))
        if comp.placement_near_max_mm is not None:
            custom_props.append(("PlacementNearMaxMM", str(comp.placement_near_max_mm)))
        if comp.datasheet or comp.description or custom_props:
            fp = Footprint(
                lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                position=fp.position, rotation=fp.rotation, layer=fp.layer,
                pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
                lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=fp.models,
                datasheet=comp.datasheet, description=comp.description,
                fp_zones=fp.fp_zones,
                custom_properties=tuple(custom_props),
            )
        if project_name is not None:
            new_lib_id = _footprint_lib_id(comp, project_name=project_name)
            fp = Footprint(
                lib_id=new_lib_id, ref=fp.ref, value=fp.value,
                position=fp.position, rotation=fp.rotation, layer=fp.layer,
                pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
                lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr, models=fp.models,
                datasheet=fp.datasheet, description=fp.description,
                fp_zones=fp.fp_zones,
                custom_properties=fp.custom_properties,
            )
        pre_footprints.append(fp)
    return pre_footprints


def _build_pre_placement_keepouts(
    ctx: _BuildContext,
    requirements: ProjectRequirements,
) -> None:
    """Create keepout zones before placement (RF, mounting holes)."""
    if ctx.has_rf and ctx.fixed_positions:
        for comp in requirements.components:
            val_lower = comp.value.lower()
            if any(kw in val_lower for kw in _RF_KEYWORDS):
                if comp.ref in ctx.fixed_positions:
                    px, py, pr = ctx.fixed_positions[comp.ref]
                    ctx.rf_pos = (px, py, pr)
                    log.info(
                        "build_pcb: anchoring antenna keepout to %s at "
                        "(%.1f, %.1f, rot=%.0f)",
                        comp.ref, px, py, pr,
                    )
                break
    if ctx.has_rf and ctx.rf_pos is not None:
        antenna_ko = _make_antenna_keepout(
            ctx.board_width_mm,
            _ANTENNA_KEEPOUT_WIDTH_MM,
            _ANTENNA_KEEPOUT_HEIGHT_MM,
            rf_position=ctx.rf_pos,
            layer_count=ctx.layer_count,
            board_height=ctx.board_height_mm,
        )
        if antenna_ko is not None:
            ctx.keepouts.append(antenna_ko)
        body_ko = _make_rf_module_body_keepout(
            ctx.rf_pos,
            layer_count=ctx.layer_count,
            board_width=ctx.board_width_mm,
            board_height=ctx.board_height_mm,
        )
        if body_ko is not None:
            log.info("build_pcb: adding RF module body keepout on inner layers")
            ctx.keepouts.append(body_ko)
    elif ctx.has_rf:
        log.info(
            "build_pcb: RF module detected but position unknown — "
            "antenna keepout deferred to post-placement (KI-019)"
        )

    # Mounting-hole keepouts
    mount_positions = ctx.template_mounting_positions
    mount_radius = _KEEPOUT_MARGIN_MM
    if (
        mount_positions is None
        and requirements.mechanical is not None
        and requirements.mechanical.mounting_hole_positions
    ):
        mount_positions = requirements.mechanical.mounting_hole_positions
    if ctx.template_mounting_diameter is not None:
        mount_radius = ctx.template_mounting_diameter / 2.0 + 1.0
    elif requirements.mechanical is not None:
        mount_radius = requirements.mechanical.mounting_hole_diameter_mm / 2.0 + 1.0

    corner_keepouts = _make_mounting_hole_keepouts(
        ctx.board_width_mm,
        ctx.board_height_mm,
        _MOUNTING_HOLE_INSET_MM,
        mount_radius,
        mounting_positions=mount_positions,
    )
    ctx.keepouts.extend(corner_keepouts)


def _run_placement_v2(
    ctx: _BuildContext,
    requirements: ProjectRequirements,
    pre_footprints: list[Footprint],
    v2_ledger_path: Path | None = None,
    v2_feedback_locks_path: Path | None = None,
) -> LayoutResult:
    """Placement engine v2: cells/contracts/proofs (placement_mode="v2").

    Halts the build with the violated constraints when any v2 stage
    fails — there is no degraded output. When *v2_ledger_path* is set,
    every v2 stage appends its proof record there (certify, cells,
    floorplan) so downstream gates extend the same evidence trail.
    """
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    from kicad_pipeline.placement_v2.pipeline import run_placement_v2

    part_rules = _Path(__file__).resolve().parents[3] / "data" / "part_rules.json"
    # Mounting-hole corners are NOT reserved: the hole placer shifts
    # holes along the edge when a corner is occupied (collision check
    # below), which beats starving edge-snapped groups of corner space.
    # Auto-ESTIMATED dims are advisory only: v2 shrink-to-fits and the
    # build adopts the packed size (a 0.03mm-too-small estimate once
    # halted a feasible power-chain floorplan).
    explicit = ctx.explicit_dimensions
    result = run_placement_v2(
        requirements,
        {fp.ref: fp for fp in pre_footprints},
        board_width_mm=ctx.board_width_mm if explicit else None,
        board_height_mm=ctx.board_height_mm if explicit else None,
        part_rules_path=part_rules if part_rules.exists() else None,
        feedback_locks_path=v2_feedback_locks_path,
        ledger_path=v2_ledger_path,
        timestamp=datetime.now(timezone.utc).isoformat() if v2_ledger_path else "",
    )
    if not result.ok:
        details = "; ".join(v.message for v in result.violations[:10])
        raise PCBError(
            f"placement v2 halted at stage {result.halted_stage!r}: {details}"
        )
    if not explicit and result.board_width > 0.0:
        ctx.board_width_mm = result.board_width
        ctx.board_height_mm = result.board_height
        ctx.outline = _make_board_outline(
            ctx.board_width_mm, ctx.board_height_mm,
            ctx.origin_x, ctx.origin_y,
            corner_radius_mm=ctx.corner_radius_mm,
        )
        # Pre-placement keepouts (mounting-hole corners) were computed
        # for the stale estimate — regenerate for the adopted outline.
        ctx.keepouts.clear()
        _build_pre_placement_keepouts(ctx, requirements)
        log.info(
            "build_pcb: adopted v2 shrink-to-fit board %.1f x %.1f mm",
            ctx.board_width_mm, ctx.board_height_mm,
        )
    return LayoutResult(
        positions=result.positions_dict(),
        rotations=result.rotations_dict(),
    )


def _run_placement(
    ctx: _BuildContext,
    requirements: ProjectRequirements,
    pre_footprints: list[Footprint],
    placement_mode: str,
    v2_ledger_path: Path | None = None,
    v2_feedback_locks_path: Path | None = None,
) -> list[Footprint]:
    """Run placement and apply positions/rotations to footprints."""
    layout_result: LayoutResult
    if placement_mode == "v2":
        layout_result = _run_placement_v2(
            ctx, requirements, pre_footprints, v2_ledger_path,
            v2_feedback_locks_path,
        )
    elif placement_mode == "grouped":
        layout_result = place_groups_off_board(
            footprints=tuple(pre_footprints),
            features=requirements.features,
            requirements=requirements,
            board_height_mm=ctx.board_height_mm,
            footprint_sizes=ctx.fp_sizes,
            fixed_positions=ctx.fixed_positions,
            board_width_mm=ctx.board_width_mm,
        )
    else:
        layout_result = layout_pcb(
            requirements, ctx.outline, footprint_sizes=ctx.fp_sizes,
            fixed_positions=ctx.fixed_positions,
            board_template=ctx.tmpl_obj,
            keepouts=tuple(ctx.keepouts),
            footprint_bboxes=ctx.fp_bboxes,
        )

    if layout_result.layers:
        for ref, lyr in layout_result.layers.items():
            if ref not in ctx.layer_overrides:
                ctx.layer_overrides[ref] = lyr

    footprints_with_pos: list[Footprint] = []
    for fp in pre_footprints:
        pos = layout_result.positions.get(fp.ref, Point(x=0.0, y=0.0))
        rot = layout_result.rotations.get(fp.ref, fp.rotation)
        fp_placed = Footprint(
            lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
            position=pos, rotation=rot, layer=fp.layer,
            pads=fp.pads, graphics=fp.graphics, texts=fp.texts,
            lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
            models=fp.models, datasheet=fp.datasheet,
            description=fp.description, fp_zones=fp.fp_zones,
            custom_properties=fp.custom_properties,
        )
        footprints_with_pos.append(fp_placed)
    return footprints_with_pos


def _create_post_placement_keepouts(
    ctx: _BuildContext,
    footprints_with_pos: list[Footprint],
) -> None:
    """Create antenna keepout from actual placed position (KI-019)."""
    if not (ctx.has_rf and ctx.rf_pos is None):
        return
    for fp in footprints_with_pos:
        val_lower = fp.value.lower() if fp.value else ""
        if any(kw in val_lower for kw in _RF_KEYWORDS):
            ctx.rf_pos = (fp.position.x, fp.position.y, fp.rotation)
            log.info(
                "build_pcb: creating post-placement antenna keepout for %s "
                "at (%.1f, %.1f, rot=%.0f)",
                fp.ref, fp.position.x, fp.position.y, fp.rotation,
            )
            antenna_ko = _make_antenna_keepout(
                ctx.board_width_mm,
                _ANTENNA_KEEPOUT_WIDTH_MM,
                _ANTENNA_KEEPOUT_HEIGHT_MM,
                rf_position=ctx.rf_pos,
                layer_count=ctx.layer_count,
                board_height=ctx.board_height_mm,
            )
            if antenna_ko is not None:
                ctx.keepouts.append(antenna_ko)
            body_ko = _make_rf_module_body_keepout(
                ctx.rf_pos,
                layer_count=ctx.layer_count,
                board_width=ctx.board_width_mm,
                board_height=ctx.board_height_mm,
            )
            if body_ko is not None:
                ctx.keepouts.append(body_ko)
            break


def _build_gnd_zones(
    ctx: _BuildContext,
    skip_inner_zones: bool,
) -> None:
    """Create GND copper pour zones and inner-layer zones."""
    gnd_net_num = ctx.net_lookup.get("GND", 1)
    zone_clearance = ZONE_CLEARANCE_DEFAULT_MM
    gnd_strategy = "both"
    gnd_zones = _make_gnd_zones(
        ctx.outline, gnd_net_num, zone_clearance, strategy=gnd_strategy,
    )
    ctx.zones.extend(gnd_zones)

    if ctx.layer_count >= 4 and not skip_inner_zones:
        from kicad_pipeline.pcb.zones import make_gnd_pour, make_power_pour

        in1_gnd = make_gnd_pour(
            ctx.outline, net_number=gnd_net_num, net_name="GND",
            layer="In1.Cu",
        )
        ctx.zones.append(in1_gnd)
        log.info("build_pcb: added In1.Cu GND plane zone")

        power5v_num = ctx.net_lookup.get("+5V")
        if power5v_num is not None:
            in2_5v = make_power_pour(
                ctx.outline, net_number=power5v_num, net_name="+5V",
                layer="In2.Cu",
            )
            ctx.zones.append(in2_5v)
            log.info("build_pcb: added In2.Cu +5V power plane zone")
    elif skip_inner_zones:
        log.info("build_pcb: skipping inner-layer zone generation (user-managed)")


def _apply_silkscreen_pass(
    ctx: _BuildContext,
    footprints_with_pos: list[Footprint],
) -> list[Footprint]:
    """Add silkscreen labels, clamp to board, resolve collisions."""
    final_footprints = []
    for fp in footprints_with_pos:
        fp_with_silk = add_silkscreen_to_footprint(fp)
        if (fp.position.x >= ctx.origin_x
                and fp.position.x <= ctx.origin_x + ctx.board_width_mm
                and fp.position.y >= ctx.origin_y
                and fp.position.y <= ctx.origin_y + ctx.board_height_mm):
            fp_with_silk = _clamp_silk_to_board(
                fp_with_silk,
                ctx.origin_x, ctx.origin_y,
                ctx.board_width_mm, ctx.board_height_mm,
            )
        final_footprints.append(fp_with_silk)

    final_footprints = _resolve_silk_collisions(final_footprints)

    if ctx.preserved_ref_text_positions:
        final_footprints = _restore_ref_text_positions(
            final_footprints, ctx.preserved_ref_text_positions,
        )

    return final_footprints


def _restore_ref_text_positions(
    footprints: list[Footprint],
    ref_text_positions: dict[str, tuple[float, float, float]],
) -> list[Footprint]:
    """Restore reference text positions from a preserved layout."""
    restored: list[Footprint] = []
    for fp in footprints:
        if fp.ref in ref_text_positions:
            tx, ty, trot = ref_text_positions[fp.ref]
            new_texts: list[FootprintText] = []
            for t in fp.texts:
                if t.text_type == "reference":
                    new_texts.append(FootprintText(
                        text_type=t.text_type,
                        text=t.text,
                        position=Point(x=tx, y=ty),
                        layer=t.layer,
                        rotation=trot,
                        effects_size=t.effects_size,
                        hidden=t.hidden,
                        uuid=t.uuid,
                    ))
                else:
                    new_texts.append(t)
            restored.append(Footprint(
                lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
                position=fp.position, rotation=fp.rotation, layer=fp.layer,
                pads=fp.pads, graphics=fp.graphics, texts=tuple(new_texts),
                lcsc=fp.lcsc, uuid=fp.uuid, attr=fp.attr,
                models=fp.models, datasheet=fp.datasheet,
                description=fp.description,
            ))
        else:
            restored.append(fp)
    log.info(
        "build_pcb: restored %d ref text positions from preserved layout",
        len(ref_text_positions),
    )
    return restored


def _mounting_hole_collides_with_footprints(
    mx: float,
    my: float,
    mh_radius: float,
    existing_fps: list[Footprint],
    fp_sizes: dict[str, tuple[float, float]],
) -> bool:
    """Check if a mounting hole at (mx, my) overlaps any existing footprint.

    Uses axis-aligned bounding box collision with a clearance gap.
    The mounting hole is modelled as a square with side = 2 * (mh_radius + 1.0)
    to include the keepout ring around the drill.
    """
    gap = 0.5  # mm clearance between mounting hole keepout and component courtyard
    mh_half = mh_radius + 1.0 + gap  # drill radius + annular ring + gap
    for fp in existing_fps:
        if fp.ref.startswith("H"):
            # Other mounting holes are obstacles too: H3's shift search
            # once walked the whole edge and stopped 0.9mm from H2
            # because placed holes were skipped (courtyards overlapped).
            w, h = (2.0 * (mh_radius + 1.0), 2.0 * (mh_radius + 1.0))
            if (mx - mh_half < fp.position.x + w / 2.0
                    and mx + mh_half > fp.position.x - w / 2.0
                    and my - mh_half < fp.position.y + h / 2.0
                    and my + mh_half > fp.position.y - h / 2.0):
                return True
            continue
        w, h = fp_sizes.get(fp.ref, (2.0, 2.0))
        # Courtyard-based size when graphics exist: fp_sizes is pad-
        # derived and under-counts THT connector BODIES — screw
        # terminals ended up inside mounting-hole courtyards on the
        # analog/power corners (Gate C 2026-06-11 standing reminder).
        from kicad_pipeline.placement_v2.footprint_geom import courtyard_halfdims
        try:
            hw_c, hh_c = courtyard_halfdims(fp)
            w, h = max(w, 2.0 * hw_c), max(h, 2.0 * hh_c)
        except (ValueError, ZeroDivisionError):
            pass
        # Account for rotation
        rot = fp.rotation % 360.0
        if 45.0 < rot < 135.0 or 225.0 < rot < 315.0:
            w, h = h, w
        half_w = w / 2.0
        half_h = h / 2.0
        # Center the AABB on the pad CENTROID, not the KiCad origin:
        # THT connectors keep their origin at pin 1, which shifted the
        # box and let holes overlap the far side of the body (J4/H3).
        fcx, fcy = origin_to_centroid(
            fp, fp.position.x, fp.position.y, fp.rotation,
        )
        # AABB overlap check
        if (mx - mh_half < fcx + half_w
                and mx + mh_half > fcx - half_w
                and my - mh_half < fcy + half_h
                and my + mh_half > fcy - half_h):
            return True
    return False


def _add_mounting_hole_footprints(
    ctx: _BuildContext,
    final_footprints: list[Footprint],
    corner_keepouts: list[Keepout],
    requirements: ProjectRequirements,
) -> None:
    """Add NPTH mounting hole footprints to the board.

    When a default corner position collides with an already-placed component,
    the mounting hole is shifted along the board edge to the nearest
    collision-free position.
    """
    mh_positions = ctx.template_mounting_positions
    mh_diameter = ctx.template_mounting_diameter
    if mh_positions is None and requirements.mechanical is not None:
        if requirements.mechanical.mounting_hole_positions:
            mh_positions = requirements.mechanical.mounting_hole_positions
        if mh_diameter is None:
            mh_diameter = requirements.mechanical.mounting_hole_diameter_mm
    if mh_positions is None and corner_keepouts:
        inset = _MOUNTING_HOLE_INSET_MM
        mh_positions = (
            (inset, inset),
            (ctx.board_width_mm - inset, inset),
            (ctx.board_width_mm - inset, ctx.board_height_mm - inset),
            (inset, ctx.board_height_mm - inset),
        )
    if mh_diameter is None:
        mh_diameter = _MOUNTING_HOLE_DIAMETER_MM

    if mh_positions:
        mh_radius = mh_diameter / 2.0
        for idx, (mx, my) in enumerate(mh_positions, start=1):
            mh_ref = f"H{idx}"

            # Check collision and shift if needed
            if _mounting_hole_collides_with_footprints(
                mx, my, mh_radius, final_footprints, ctx.fp_sizes,
            ):
                orig_x, orig_y = mx, my
                # Try shifting along the nearest edge (inward along Y or X)
                inset = _MOUNTING_HOLE_INSET_MM
                bh = ctx.board_height_mm
                # Determine which corner this is and shift direction
                is_top = my < bh / 2.0
                # Shift along Y edge (move away from corner)
                shift_step = 3.0  # mm per step
                for step in range(1, 10):
                    new_y = my + (shift_step * step * (1.0 if is_top else -1.0))
                    # Keep within board bounds
                    new_y = max(inset, min(new_y, bh - inset))
                    if not _mounting_hole_collides_with_footprints(
                        mx, new_y, mh_radius, final_footprints, ctx.fp_sizes,
                    ):
                        my = new_y
                        log.info(
                            "build_pcb: shifted %s from (%.1f, %.1f) to "
                            "(%.1f, %.1f) to avoid component collision",
                            mh_ref, orig_x, orig_y, mx, my,
                        )
                        break
                else:
                    log.warning(
                        "build_pcb: no collision-free position for %s near "
                        "(%.1f, %.1f) — SKIPPING the hole (an overlapping "
                        "hole is worse than a missing one)",
                        mh_ref, orig_x, orig_y,
                    )
                    continue

            mh_fp = make_mounting_hole(mh_ref, drill_diameter=mh_diameter)
            mh_fp = Footprint(
                lib_id=mh_fp.lib_id, ref=mh_fp.ref, value=mh_fp.value,
                position=Point(mx, my), rotation=0.0, layer=mh_fp.layer,
                pads=mh_fp.pads, graphics=mh_fp.graphics, texts=mh_fp.texts,
                uuid=mh_fp.uuid, attr=mh_fp.attr,
            )
            final_footprints.append(mh_fp)
        log.info(
            "build_pcb: added %d mounting hole footprints",
            len(mh_positions),
        )

    # Keepout rings must follow the holes ACTUALLY placed: shifted or
    # skipped holes left stale rings at the nominal corners — screw
    # terminals rendered sitting on hole-shaped keepouts where no hole
    # exists (human finding 2026-06-11, power_chain J1/J2 corners).
    placed_holes = tuple(
        (fp.position.x, fp.position.y)
        for fp in final_footprints
        if fp.ref.startswith("H") and fp.pads
        and fp.pads[0].pad_type == "np_thru_hole"
    )
    ctx.keepouts[:] = [
        k for k in ctx.keepouts if getattr(k, "tag", "") != "mounting_hole"
    ]
    if placed_holes:
        ctx.keepouts.extend(_make_mounting_hole_keepouts(
            ctx.board_width_mm, ctx.board_height_mm, _MOUNTING_HOLE_INSET_MM,
            mh_diameter / 2.0 + 1.0,
            mounting_positions=placed_holes,
        ))


def _run_autoroute_step(
    ctx: _BuildContext,
    requirements: ProjectRequirements,
    final_footprints: list[Footprint],
    auto_route: bool,
    netclasses: tuple[object, ...],
) -> tuple[tuple[Track, ...], tuple[Via, ...], bool]:
    """Run autorouting (FreeRouting then grid router fallback).

    Returns:
        (all_tracks, all_vias, freerouting_used)
    """
    all_tracks: tuple[Track, ...] = ()
    all_vias: tuple[Via, ...] = ()
    freerouting_used = False

    if not auto_route:
        return all_tracks, all_vias, freerouting_used

    from kicad_pipeline.routing.freerouting import (
        find_freerouting_jar,
        route_with_freerouting,
        ses_to_tracks,
        ses_to_vias,
    )

    jar_path = find_freerouting_jar()
    if jar_path is not None:
        log.info("build_pcb: FreeRouting JAR found at %s", jar_path)
        from kicad_pipeline.pcb.netlist import assign_net_numbers_to_footprints, build_netlist
        netlist = build_netlist(requirements)
        pre_route_fps = assign_net_numbers_to_footprints(
            list(final_footprints), netlist,
        )
        design_rules = DesignRules(layer_count=ctx.layer_count)
        pre_route_design = PCBDesign(
            outline=ctx.outline,
            design_rules=design_rules,
            nets=ctx.nets,
            footprints=tuple(pre_route_fps),
            tracks=(), vias=(), zones=(),
            keepouts=tuple(ctx.keepouts),
            netclasses=netclasses,
        )
        import tempfile
        dsn_dir = tempfile.mkdtemp(prefix="kicad_freeroute_")
        dsn_path = Path(dsn_dir) / "design.dsn"
        from kicad_pipeline.routing.dsn_export import write_dsn
        write_dsn(pre_route_design, dsn_path)
        log.info("build_pcb: exported DSN to %s", dsn_path)

        fr_result = route_with_freerouting(
            str(dsn_path), jar_path=jar_path, timeout_seconds=300,
        )
        if fr_result.success and fr_result.ses_file is not None:
            ses_content = Path(fr_result.ses_file).read_text(encoding="utf-8")
            all_tracks = ses_to_tracks(ses_content, pre_route_design)
            all_vias = ses_to_vias(ses_content, pre_route_design)
            freerouting_used = True
            log.info(
                "build_pcb: FreeRouting complete — %d tracks, %d vias",
                len(all_tracks), len(all_vias),
            )
        else:
            log.warning(
                "build_pcb: FreeRouting failed (%s), falling back to grid router",
                fr_result.error,
            )
    else:
        log.info("build_pcb: FreeRouting JAR not found, using grid router")

    if not freerouting_used:
        all_tracks, all_vias = _run_grid_router(
            ctx, requirements, final_footprints, netclasses,
        )

    return all_tracks, all_vias, freerouting_used


def _run_grid_router(
    ctx: _BuildContext,
    requirements: ProjectRequirements,
    final_footprints: list[Footprint],
    netclasses: tuple[object, ...],
) -> tuple[tuple[Track, ...], tuple[Via, ...]]:
    """Run the grid router fallback."""
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
        ctx.board_width_mm, ctx.board_height_mm,
        grid_step_mm=0.25,
        net_widths=widths,
        net_clearances=clearances,
        keepouts=tuple(ctx.keepouts),
        corner_radius_mm=ctx.corner_radius_mm,
    )
    all_tracks = collect_tracks(route_results, routed_only=False)
    all_vias = collect_vias(route_results)
    routed = sum(1 for r in route_results if r.routed)
    unrouted = sum(1 for r in route_results if not r.routed)
    log.info(
        "build_pcb: autoroute complete — %d tracks, %d routed, %d unrouted",
        len(all_tracks), routed, unrouted,
    )

    from kicad_pipeline.routing.metrics import compute_board_metrics
    metrics = compute_board_metrics(route_results, final_footprints)
    log.info(
        "build_pcb: routing %.1fmm total (%.2fx ideal), %d vias, %d/%d nets",
        metrics.total_track_length_mm,
        metrics.overall_length_ratio,
        metrics.total_vias,
        metrics.nets_routed,
        metrics.nets_routed + metrics.nets_failed,
    )

    return all_tracks, all_vias


def _add_rf_via_fence(
    ctx: _BuildContext,
    requirements: ProjectRequirements,
    final_footprints: list[Footprint],
    all_vias: tuple[Via, ...],
) -> tuple[Via, ...]:
    """Add RF via fence if an RF module is present.

    Currently disabled — the antenna keepout zone provides sufficient
    isolation.  Via fences are added manually during routing if needed.
    """
    # Via fence generation disabled: keepout zone is sufficient for
    # pre-routing placement.  Vias should be added during routing
    # when the ground plane topology is known.
    return all_vias


def _add_gnd_stitching_vias(
    ctx: _BuildContext,
    final_footprints: list[Footprint],
    all_vias: tuple[Via, ...],
    all_tracks: tuple[Track, ...],
    auto_route: bool,
    freerouting_used: bool,
) -> tuple[Via, ...]:
    """Add GND stitching vias when using grid router."""
    if auto_route and not freerouting_used:
        gnd_net_num = ctx.net_lookup.get("GND", 1)
        stitch_vias = _make_gnd_stitching_vias(
            ctx.outline, gnd_net_num, tuple(final_footprints),
            all_vias, all_tracks,
            keepout_zones=tuple(ctx.keepouts),
        )
        if stitch_vias:
            all_vias = all_vias + stitch_vias
            log.info("build_pcb: added %d GND stitching vias", len(stitch_vias))
    return all_vias


def _preserve_user_routing(
    ctx: _BuildContext,
    preserve_from: str | Path | object | None,
    preserve_routing: bool,
    pcb_file_path: str | Path | None,
    all_tracks: tuple[Track, ...],
    all_vias: tuple[Via, ...],
) -> tuple[tuple[Track, ...], tuple[Via, ...]]:
    """Preserve routing from an existing PCB (tracks, vias, zones, edge cuts)."""
    if not (preserve_routing and preserve_from is not None):
        return all_tracks, all_vias

    from kicad_pipeline.pcb.position_extractor import (
        remap_routing,
        routing_from_source,
    )

    _pcb_path = pcb_file_path
    if _pcb_path is None and isinstance(preserve_from, str | Path):
        _pcb_path = preserve_from

    preserved = routing_from_source(preserve_from, pcb_file_path=_pcb_path)
    if preserved is None or not (
        preserved.tracks or preserved.vias or preserved.zones
        or preserved.edge_cuts
    ):
        return all_tracks, all_vias

    new_net_map: dict[str, int] = {}
    for net_entry in ctx.nets:
        new_net_map[net_entry.name] = net_entry.number

    remapped_tracks, remapped_vias, remapped_zones = remap_routing(
        preserved, new_net_map,
    )
    all_tracks = all_tracks + remapped_tracks
    all_vias = all_vias + remapped_vias
    ctx.zones.extend(remapped_zones)

    if preserved.keepouts:
        log.info(
            "build_pcb: preserving %d user keepout zones",
            len(preserved.keepouts),
        )
        ctx.keepouts.extend(preserved.keepouts)

    if preserved.edge_cuts:
        log.info(
            "build_pcb: preserving %d user edge cut segments",
            len(preserved.edge_cuts),
        )
        _preserved_edge_cuts.clear()
        _preserved_edge_cuts.extend(preserved.edge_cuts)

    return all_tracks, all_vias


# ---------------------------------------------------------------------------
# build_pcb helpers
# ---------------------------------------------------------------------------


def _compute_footprint_sizes(
    requirements: ProjectRequirements,
) -> tuple[dict[str, tuple[float, float]], float]:
    """Compute footprint sizes and total component area."""
    fp_sizes: dict[str, tuple[float, float]] = {}
    total_area = 0.0
    for comp in requirements.components:
        sz = estimate_footprint_size(comp.footprint)
        fp_sizes[comp.ref] = sz
        total_area += sz[0] * sz[1]
    return fp_sizes, total_area


def _compute_footprint_bboxes(
    pre_footprints: list[Footprint],
) -> dict[str, object]:
    """Compute bounding boxes for all pre-placed footprints."""
    fp_bboxes: dict[str, object] = {}
    for fp in pre_footprints:
        fp_bboxes[fp.ref] = compute_footprint_bbox(fp)
    return fp_bboxes


def _assemble_pcb_design(
    ctx: _BuildContext,
    requirements: ProjectRequirements,
    final_footprints: list[Footprint],
    nets: tuple[NetEntry, ...],
    netclasses: tuple[object, ...],
    all_tracks: tuple[Track, ...],
    all_vias: tuple[Via, ...],
) -> PCBDesign:
    """Assign net numbers, generate DRC exclusions, and build the final design."""
    from kicad_pipeline.pcb.netlist import assign_net_numbers_to_footprints, build_netlist

    netlist = build_netlist(requirements)
    final_footprints = assign_net_numbers_to_footprints(final_footprints, netlist)

    drc_exclusions = _generate_ic_drc_exclusions(final_footprints)
    if drc_exclusions:
        log.info(
            "build_pcb: generated %d intra-footprint DRC exclusions",
            len(drc_exclusions),
        )

    design_rules = DesignRules(layer_count=ctx.layer_count)

    log.info(
        "build_pcb complete: %d footprints, %d nets, %d zones, %d keepouts, "
        "%d tracks, %d vias",
        len(final_footprints), len(nets), len(ctx.zones), len(ctx.keepouts),
        len(all_tracks), len(all_vias),
    )

    return PCBDesign(
        outline=ctx.outline,
        design_rules=design_rules,
        nets=nets,
        footprints=tuple(final_footprints),
        tracks=all_tracks,
        vias=all_vias,
        zones=tuple(ctx.zones),
        keepouts=tuple(ctx.keepouts),
        netclasses=netclasses,
        drc_exclusions=drc_exclusions,
        version=KICAD_PCB_VERSION,
        generator=KICAD_GENERATOR,
        title=requirements.project.name,
        date=datetime.date.today().isoformat(),
        revision=requirements.project.revision,
        company=requirements.project.author or "",
    )


def _setup_board(
    requirements: ProjectRequirements,
    board_template: str | None,
    board_width_mm: float | None,
    board_height_mm: float | None,
    origin_x: float,
    origin_y: float,
    layer_count: int,
    project_name: str | None,
    preserve_from: str | Path | object | None,
    preserve_ref_text: bool,
) -> tuple[_BuildContext, tuple[NetEntry, ...], list[Footprint]]:
    """Resolve template, dimensions, nets, and footprints; build context."""
    (
        board_template, board_width_mm, board_height_mm, corner_radius_mm,
        fixed_positions, layer_overrides, template_mounting_positions,
        template_mounting_diameter, tmpl_obj,
    ) = _resolve_board_template(
        requirements, board_template, board_width_mm, board_height_mm,
    )

    fixed_positions, preserved_ref_text_positions = _resolve_preserved_positions(
        preserve_from, preserve_ref_text, requirements, fixed_positions,
    )

    board_width_mm, board_height_mm, _explicit_dimensions = _resolve_board_dimensions(
        board_width_mm, board_height_mm, requirements,
    )

    outline = _make_board_outline(
        board_width_mm, board_height_mm, origin_x, origin_y,
        corner_radius_mm=corner_radius_mm,
    )

    nets = _build_nets(requirements)
    net_lookup: dict[str, int] = {n.name: n.number for n in nets}

    pre_footprints = _build_pre_footprints(
        requirements, net_lookup, layer_overrides, project_name,
    )

    fp_sizes, total_area = _compute_footprint_sizes(requirements)

    if tmpl_obj is None and not _explicit_dimensions:
        board_width_mm, board_height_mm, outline = _auto_size_board(
            board_width_mm, board_height_mm, origin_x, origin_y,
            corner_radius_mm, fp_sizes, total_area,
        )

    _warn_board_size(board_width_mm, board_height_mm, total_area)

    fp_bboxes = _compute_footprint_bboxes(pre_footprints)

    # Upgrade fp_sizes with actual pad-extent-based sizes when available.
    # The initial fp_sizes from estimate_footprint_size() can be significantly
    # smaller than the real JLCPCB footprints, causing the solver to undercount
    # collisions.  Here we replace estimates with bbox-derived sizes + 1mm
    # courtyard margin (matching review_agent._fp_size_dict convention).
    courtyard_margin_mm = 1.0
    for ref, bbox in fp_bboxes.items():
        from kicad_pipeline.models.pcb import FootprintBBox as _FpBBox
        if isinstance(bbox, _FpBBox):
            bbox_w = bbox.max_x - bbox.min_x + courtyard_margin_mm
            bbox_h = bbox.max_y - bbox.min_y + courtyard_margin_mm
            est_w, est_h = fp_sizes.get(ref, (0.0, 0.0))
            # Use the larger of estimate vs actual (never shrink)
            if bbox_w * bbox_h > est_w * est_h:
                fp_sizes[ref] = (bbox_w, bbox_h)

    ctx = _BuildContext(
        board_width_mm=board_width_mm,
        board_height_mm=board_height_mm,
        origin_x=origin_x,
        origin_y=origin_y,
        corner_radius_mm=corner_radius_mm,
        layer_count=layer_count,
        project_name=project_name,
        outline=outline,
        nets=nets,
        net_lookup=net_lookup,
        keepouts=[],
        zones=[],
        fixed_positions=fixed_positions,
        layer_overrides=layer_overrides,
        template_mounting_positions=template_mounting_positions,
        template_mounting_diameter=template_mounting_diameter,
        tmpl_obj=tmpl_obj,
        preserved_ref_text_positions=preserved_ref_text_positions,
        has_rf=_has_rf_module(requirements),
        rf_pos=None,
        fp_sizes=fp_sizes,
        fp_bboxes=fp_bboxes,
        explicit_dimensions=tmpl_obj is not None or _explicit_dimensions,
    )

    return ctx, nets, pre_footprints


def _post_placement_assembly(
    ctx: _BuildContext,
    requirements: ProjectRequirements,
    footprints_with_pos: list[Footprint],
    nets: tuple[NetEntry, ...],
    corner_keepouts: list[object],
    auto_route: bool,
    preserve_from: str | Path | object | None,
    preserve_routing: bool,
    pcb_file_path: str | Path | None,
    skip_inner_zones: bool,
) -> PCBDesign:
    """Run post-placement steps: keepouts, zones, silkscreen, routing, assembly."""
    _create_post_placement_keepouts(ctx, footprints_with_pos)

    # Stamp group metadata from FeatureBlock requirements onto footprints.
    group_map: dict[str, str] = {}
    for feat in requirements.features:
        for ref in feat.components:
            group_map[ref] = feat.name
    footprints_with_pos = [
        replace(fp, group=group_map[fp.ref]) if fp.ref in group_map and not fp.group else fp
        for fp in footprints_with_pos
    ]

    netclasses = classify_nets(nets)
    _build_gnd_zones(ctx, skip_inner_zones)

    final_footprints = _apply_silkscreen_pass(ctx, footprints_with_pos)
    _add_mounting_hole_footprints(ctx, final_footprints, corner_keepouts, requirements)

    all_tracks, all_vias, freerouting_used = _run_autoroute_step(
        ctx, requirements, final_footprints, auto_route, netclasses,
    )

    all_vias = _add_rf_via_fence(ctx, requirements, final_footprints, all_vias)
    all_vias = _add_gnd_stitching_vias(
        ctx, final_footprints, all_vias, all_tracks, auto_route, freerouting_used,
    )
    all_tracks, all_vias = _preserve_user_routing(
        ctx, preserve_from, preserve_routing, pcb_file_path,
        all_tracks, all_vias,
    )

    return _assemble_pcb_design(
        ctx, requirements, final_footprints, nets, netclasses,
        all_tracks, all_vias,
    )


# ---------------------------------------------------------------------------
# Component registry gate
# ---------------------------------------------------------------------------


def _check_component_registry(requirements: ProjectRequirements) -> None:
    """Warn about components not in the registry or with failed verification.

    Logs warnings for unverified/missing components and raises :class:`PCBError`
    for components with ``verification_status == "failed"``.
    """
    try:
        from kicad_pipeline.validation.component_registry import ComponentRegistry
        registry = ComponentRegistry()
    except Exception:
        log.debug("Component registry not available — skipping gate")
        return

    failed: list[str] = []
    unverified: list[str] = []
    missing: list[str] = []

    for comp in requirements.components:
        spec = registry.get(comp.footprint)
        if spec is None:
            missing.append(f"{comp.ref} ({comp.footprint})")
        elif spec.verification_status == "failed":
            failed.append(f"{comp.ref} ({comp.footprint})")
        elif spec.verification_status == "unverified":
            unverified.append(f"{comp.ref} ({comp.footprint})")

    if missing:
        log.warning(
            "Component registry: %d component(s) not registered — "
            "run /verify-components to add and verify: %s",
            len(missing), ", ".join(missing),
        )
    if unverified:
        log.warning(
            "Component registry: %d component(s) unverified — "
            "run /verify-components to verify: %s",
            len(unverified), ", ".join(unverified),
        )
    if failed:
        raise PCBError(
            f"Component registry: {len(failed)} component(s) have FAILED "
            f"verification — fix before building: {', '.join(failed)}"
        )


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
    preserve_from: str | Path | object | None = None,
    placement_mode: str = "solver",
    layer_count: int = 2,
    preserve_ref_text: bool = True,
    preserve_routing: bool = True,
    pcb_file_path: str | Path | None = None,
    skip_inner_zones: bool = False,
    project_name: str | None = None,
    v2_ledger_path: str | Path | None = None,
    v2_feedback_locks_path: str | Path | None = None,
) -> PCBDesign:
    """Build a complete :class:`PCBDesign` from *requirements*.

    Resolves board dimensions, generates footprints, runs placement,
    adds zones/keepouts/silkscreen, routes traces, and returns the
    assembled design. ``v2_ledger_path`` (placement_mode="v2" only)
    appends each v2 stage's proof record to a build ledger.

    Raises:
        PCBError: If the requirements contain no components.
    """
    if not requirements.components:
        raise PCBError("Cannot build PCB: requirements has no components")

    # -- Component registry gate: warn/block on unverified components ------
    _check_component_registry(requirements)

    _preserved_edge_cuts.clear()

    log.info(
        "build_pcb: %d components, %d nets",
        len(requirements.components),
        len(requirements.nets),
    )

    ctx, nets, pre_footprints = _setup_board(
        requirements, board_template, board_width_mm, board_height_mm,
        origin_x, origin_y, layer_count, project_name,
        preserve_from, preserve_ref_text,
    )

    _build_pre_placement_keepouts(ctx, requirements)

    footprints_with_pos = _run_placement(
        ctx, requirements, pre_footprints, placement_mode,
        v2_ledger_path=Path(v2_ledger_path) if v2_ledger_path is not None else None,
        v2_feedback_locks_path=(
            Path(v2_feedback_locks_path)
            if v2_feedback_locks_path is not None else None
        ),
    )
    # Captured AFTER placement: v2 may adopt a shrink-to-fit board size
    # and regenerate the mounting-hole corner keepouts for it.
    corner_keepouts = list(ctx.keepouts)

    return _post_placement_assembly(
        ctx, requirements, footprints_with_pos, nets,
        corner_keepouts, auto_route, preserve_from, preserve_routing,
        pcb_file_path, skip_inner_zones,
    )


def _generate_ic_drc_exclusions(
    footprints: list[Footprint],
) -> tuple[str, ...]:
    """Generate DRC exclusion strings for dense IC intra-footprint clearance.

    Fine-pitch ICs (MSOP, TSSOP, QFN, etc.) have pads closer together than
    typical clearance rules.  Escape routes from these pads inevitably pass
    within the clearance zone of adjacent pads.  KiCad flags these as
    violations but they are expected — the pad spacing is fixed by the
    package geometry.

    Returns:
        Tuple of KiCad DRC exclusion strings in ``"clearance|ref|pad|ref|pad"``
        format for all adjacent pad pairs on dense IC footprints.
    """
    exclusions: list[str] = []
    for fp in footprints:
        if len(fp.pads) < 6:
            continue
        # Compute minimum pad spacing
        positions = sorted(
            (p.position.x, p.position.y, p.number) for p in fp.pads
        )
        min_spacing = 999.0
        for i in range(len(positions) - 1):
            dx = abs(positions[i + 1][0] - positions[i][0])
            dy = abs(positions[i + 1][1] - positions[i][1])
            d = (dx * dx + dy * dy) ** 0.5
            if d > 0.01:
                min_spacing = min(min_spacing, d)
        if min_spacing >= 1.0:
            continue
        # Dense IC — generate exclusions for all adjacent pad pairs
        # that are within 2x the minimum spacing
        threshold = min_spacing * 2.5
        for i, pad_a in enumerate(fp.pads):
            for pad_b in fp.pads[i + 1 :]:
                dx = abs(pad_a.position.x - pad_b.position.x)
                dy = abs(pad_a.position.y - pad_b.position.y)
                d = (dx * dx + dy * dy) ** 0.5
                if d < threshold:
                    exclusions.append(
                        f"clearance|{fp.ref}|{pad_a.number}"
                        f"|{fp.ref}|{pad_b.number}"
                    )
    return tuple(exclusions)


# ---------------------------------------------------------------------------
# S-expression serialiser
# ---------------------------------------------------------------------------

def _build_layer_table(
    layer_count: int = 2,
) -> list[tuple[int, str, str] | tuple[int, str, str, str]]:
    """Build the layer table for the given copper layer count.

    For 2-layer boards, returns the standard F.Cu/B.Cu table.
    For 4-layer boards, adds In1.Cu (GND plane) and In2.Cu (power plane).

    Args:
        layer_count: Number of copper layers (2 or 4).

    Returns:
        Layer definition list suitable for KiCad S-expression output.
    """
    table: list[tuple[int, str, str] | tuple[int, str, str, str]] = [
        (0, "F.Cu", "signal"),
    ]
    if layer_count >= 4:
        table.append((4, "In1.Cu", "power"))
        table.append((6, "In2.Cu", "power"))
    table.extend([
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
    ])
    return table


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


def _hidden_property_sexp(
    name: str,
    value: str,
    fab_layer: str,
) -> list[SExpNode]:
    """Build a hidden ``(property ...)`` node on the fab layer."""
    return [
        "property", name, value,
        ["at", 0, 0, 0], ["layer", fab_layer],
        ["effects", ["font", ["size", 1.0, 1.0]], ["hide", "yes"]],
    ]


def _fp_standard_properties(fp: Footprint) -> list[list[SExpNode]]:
    """Build Reference, Value, Footprint, Datasheet, Description properties."""
    fab = "B.Fab" if fp.layer == LAYER_B_CU else "F.Fab"

    # Reference property — always centered on footprint body (at 0 0).
    # Footprint text entries may carry offsets for standalone .kicad_mod files,
    # but in PCB files the reference label belongs at the footprint origin.
    ref_text = next((t for t in fp.texts if t.text_type == "reference"), None)
    ref_x = 0.0
    ref_y = 0.0
    default_silk = "B.SilkS" if fp.layer == LAYER_B_CU else "F.SilkS"
    ref_layer = ref_text.layer if ref_text else default_silk
    ref_size = ref_text.effects_size if ref_text else 1.0
    ref_hidden = ref_text.hidden if ref_text else False
    ref_rotation = ref_text.rotation if ref_text else 0.0
    ref_effects: list[SExpNode] = [
        "effects", ["font", ["size", ref_size, ref_size]],
    ]
    if ref_hidden:
        ref_effects.append(["hide", "yes"])

    val_x = 0.0
    val_y = 0.0

    props: list[list[SExpNode]] = [
        [
            "property", "Reference", fp.ref,
            ["at", ref_x, ref_y, ref_rotation],
            ["layer", ref_layer],
            ref_effects,
        ],
        [
            "property", "Value", fp.value,
            ["at", val_x, val_y, 0],
            ["layer", fab],
            ["effects", ["font", ["size", 1.0, 1.0]], ["hide", "yes"]],
        ],
        _hidden_property_sexp("Footprint", fp.lib_id, fab),
        _hidden_property_sexp("Datasheet", fp.datasheet or "", fab),
        _hidden_property_sexp("Description", fp.description or "", fab),
    ]
    return props


def _fp_optional_properties(fp: Footprint) -> list[list[SExpNode]]:
    """Build LCSC, MPN, Manufacturer, and custom properties if present."""
    fab = "B.Fab" if fp.layer == LAYER_B_CU else "F.Fab"
    props: list[list[SExpNode]] = []
    if fp.lcsc:
        props.append(_hidden_property_sexp("LCSC", fp.lcsc, fab))
    if fp.mpn:
        props.append(_hidden_property_sexp("MPN", fp.mpn, fab))
    if fp.manufacturer:
        props.append(_hidden_property_sexp("Manufacturer", fp.manufacturer, fab))
    # Emit group/subgroup metadata as KiCad properties.
    if fp.group:
        props.append(_hidden_property_sexp("Group", fp.group, fab))
    if fp.subgroup:
        props.append(_hidden_property_sexp("Subgroup", fp.subgroup, fab))
    # Emit custom properties (placement constraints, etc.)
    for prop_name, prop_value in fp.custom_properties:
        props.append(_hidden_property_sexp(prop_name, prop_value, fab))
    return props


def _fp_graphic_sexp(graphic: FootprintLine | FootprintArc | FootprintCircle) -> list[SExpNode]:
    """Serialise a single footprint graphic element."""
    if isinstance(graphic, FootprintLine):
        g: list[SExpNode] = [
            "fp_line",
            ["start", graphic.start.x, graphic.start.y],
            ["end", graphic.end.x, graphic.end.y],
            ["layer", graphic.layer],
            ["width", graphic.width],
        ]
    elif isinstance(graphic, FootprintArc):
        g = [
            "fp_arc",
            ["start", graphic.start.x, graphic.start.y],
            ["mid", graphic.mid.x, graphic.mid.y],
            ["end", graphic.end.x, graphic.end.y],
            ["layer", graphic.layer],
            ["width", graphic.width],
        ]
    else:  # FootprintCircle
        g = [
            "fp_circle",
            ["center", graphic.center.x, graphic.center.y],
            ["end", graphic.end.x, graphic.end.y],
            ["layer", graphic.layer],
            ["width", graphic.width],
        ]
    if graphic.uuid:
        g.append(["uuid", graphic.uuid])
    return g


def _fp_keepout_sexp(keepout: FootprintKeepout) -> SExpNode:
    """Serialise a :class:`FootprintKeepout` to a KiCad ``(zone ...)`` node.

    Footprint-level keepout zones are emitted at board level using the same
    ``(zone ...)`` syntax as board-level keepouts (see :func:`_keepout_sexp`).

    Args:
        keepout: Footprint keepout zone to serialise.

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

    # Must match _keepout_sexp() ordering exactly: net, net_name, layers,
    # uuid, hatch, keepout rules, polygon.
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
        ["attr", *fp.attr.split()],
    ]

    if fp.uuid:
        node.append(["uuid", fp.uuid])

    # Standard and optional properties
    for prop in _fp_standard_properties(fp):
        node.append(prop)
    for prop in _fp_optional_properties(fp):
        node.append(prop)

    # Custom user text (not reference/value — those are properties now)
    for text in fp.texts:
        if text.text_type not in ("reference", "value"):
            node.append(_footprint_text_sexp(text))

    # Footprint graphics (courtyard, silkscreen, fab outlines)
    for graphic in fp.graphics:
        if isinstance(graphic, FootprintLine | FootprintArc | FootprintCircle):
            node.append(_fp_graphic_sexp(graphic))

    for pad in fp.pads:
        node.append(_pad_sexp(pad))

    # Footprint-level keepout zones are emitted at BOARD level in pcb_to_sexp(),
    # not inside the footprint node. KiCad does not support (zone ...) inside
    # (footprint ...) blocks.

    # 3D model references
    for model in fp.models:
        node.append([
            "model",
            model.path,
            ["offset", ["xyz", model.offset[0], model.offset[1], model.offset[2]]],
            ["scale", ["xyz", model.scale[0], model.scale[1], model.scale[2]]],
            ["rotate", ["xyz", model.rotate[0], model.rotate[1], model.rotate[2]]],
        ])

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
    if zone.priority > 0:
        node.append(["priority", zone.priority])
    # KiCad 9: fill node needs "yes" marker when zone has fill data
    fill_node: list[SExpNode] = ["fill"]
    if zone.filled_polygons:
        fill_node.append("yes")
    fill_node.extend([
        ["thermal_gap", zone.thermal_relief_gap],
        ["thermal_bridge_width", zone.thermal_relief_bridge],
    ])

    node.extend([
        ["hatch", "edge", 0.508],
        ["connect_pads", ["clearance", zone.clearance_mm]],
        ["min_thickness", zone.min_thickness],
        ["filled_areas_thickness", False],
        fill_node,
        ["polygon", pts_node],
    ])

    # Emit filled_polygon entries for pre-computed zone fill
    for fp_pts in zone.filled_polygons:
        fp_node: list[SExpNode] = ["pts"]
        for pt in fp_pts:
            fp_node.append(["xy", pt.x, pt.y])
        node.append(["filled_polygon", ["layer", zone.layer], fp_node])

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


def _pcb_title_block_sexp(design: PCBDesign) -> list[SExpNode] | None:
    """Build the PCB ``(title_block ...)`` node, or ``None`` if empty."""
    if not (design.title or design.date or design.revision or design.company):
        return None
    tb: list[SExpNode] = ["title_block"]
    if design.title:
        tb.append(["title", design.title])
    if design.date:
        tb.append(["date", design.date])
    if design.revision:
        tb.append(["rev", design.revision])
    if design.company:
        tb.append(["company", design.company])
    return tb


def _track_sexp(track: Track) -> list[SExpNode]:
    """Serialise a :class:`Track` to a ``(segment ...)`` node."""
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
    return seg


def _via_sexp(via: Via) -> list[SExpNode]:
    """Serialise a :class:`Via` to a ``(via ...)`` node."""
    node: list[SExpNode] = [
        "via",
        ["at", via.position.x, via.position.y],
        ["size", via.size],
        ["drill", via.drill],
        ["layers", *via.layers],
        ["net", via.net_number],
    ]
    if via.uuid:
        node.append(["uuid", via.uuid])
    return node


def _pcb_sexp_header(design: PCBDesign) -> list[SExpNode]:
    root: list[SExpNode] = [
        "kicad_pcb",
        ["version", design.version],
        ["generator", design.generator],
        ["generator_version", design.generator_version],
        ["general", ["thickness", 1.6], ["legacy_teardrops", False]],
        ["paper", "A4"],
    ]
    tb = _pcb_title_block_sexp(design)
    if tb is not None:
        root.append(tb)
    layers_node: list[SExpNode] = ["layers"]
    for layer_entry in _build_layer_table(design.design_rules.layer_count):
        layer_node: list[SExpNode] = [layer_entry[0], layer_entry[1], layer_entry[2]]
        if len(layer_entry) > 3:
            layer_node.append(layer_entry[3])
        layers_node.append(layer_node)
    root.append(layers_node)
    root.append([
        "setup",
        ["pad_to_mask_clearance", 0],
        ["allow_soldermask_bridges_in_footprints", False],
        ["pcbplotparams", ["layerselection", "0x00010fc_ffffffff"], ["outputdirectory", ""]],
    ])
    return root


def _pcb_sexp_fp_keepouts(design: PCBDesign) -> list[SExpNode]:
    import math as _math
    from dataclasses import replace as _replace
    result: list[SExpNode] = []
    for fp in design.footprints:
        for fz in fp.fp_zones:
            rot_rad = _math.radians(fp.rotation)
            cos_r, sin_r = _math.cos(rot_rad), _math.sin(rot_rad)
            board_pts: list[Point] = []
            for pt in fz.polygon:
                bx = fp.position.x + pt.x * cos_r - pt.y * sin_r
                by = fp.position.y + pt.x * sin_r + pt.y * cos_r
                board_pts.append(Point(x=round(bx, 4), y=round(by, 4)))
            board_fz = _replace(fz, polygon=tuple(board_pts))
            result.append(_fp_keepout_sexp(board_fz))
    return result


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
    root = _pcb_sexp_header(design)

    for net in design.nets:
        root.append(["net", net.number, net.name])
    for fp in design.footprints:
        root.append(_footprint_sexp(fp))
    for line in _outline_sexp(design.outline):
        root.append(line)

    for start, end, width in _preserved_edge_cuts:
        root.append([
            "gr_line",
            ["start", start.x, start.y], ["end", end.x, end.y],
            ["layer", LAYER_EDGE_CUTS], ["width", width],
        ])

    for zone in design.zones:
        root.append(_zone_sexp(zone))
    for keepout in design.keepouts:
        root.append(_keepout_sexp(keepout))

    for node in _pcb_sexp_fp_keepouts(design):
        root.append(node)

    for track in design.tracks:
        root.append(_track_sexp(track))
    for via in design.vias:
        root.append(_via_sexp(via))

    return root


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _sync_footprint_library(design: PCBDesign, project_dir: Path) -> None:
    """Auto-generate a project-local ``.pretty`` library from PCB footprints.

    Inspects the ``lib_id`` of each footprint to discover the project name
    prefix (``{project}:{name}``).  When a consistent prefix is found, writes
    one ``.kicad_mod`` per unique footprint and a matching ``fp-lib-table``.

    This ensures KiCad's "Update PCB from Schematic" can always resolve every
    footprint reference in the schematic.
    """
    # Discover the project name from lib_id prefixes — use the most common
    # prefix (mounting holes, KiCad standard libs may use a different one).
    from collections import Counter

    from kicad_pipeline.pcb.footprint_library import (
        footprint_name_from_lib_id,
        footprint_to_kicad_mod,
        write_fp_lib_table,
    )

    prefix_counts: Counter[str] = Counter()
    for fp in design.footprints:
        if ":" in fp.lib_id:
            prefix_counts[fp.lib_id.split(":")[0]] += 1

    if not prefix_counts:
        return

    project_name = prefix_counts.most_common(1)[0][0]

    # Build .pretty directory
    pretty_dir = project_dir / f"{project_name}.pretty"
    pretty_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    prefix_colon = f"{project_name}:"
    for fp in design.footprints:
        if not fp.lib_id.startswith(prefix_colon):
            continue
        fp_name = footprint_name_from_lib_id(fp.lib_id)
        if fp_name in seen:
            continue
        seen.add(fp_name)

        fp_sexp = _footprint_sexp(fp)
        assert isinstance(fp_sexp, list)
        kicad_mod = footprint_to_kicad_mod(fp_sexp, fp_name)
        mod_path = pretty_dir / f"{fp_name}.kicad_mod"
        mod_path.write_text(kicad_mod, encoding="utf-8")

    # Write fp-lib-table
    write_fp_lib_table(project_dir, project_name)

    log.info(
        "_sync_footprint_library: %d footprints in %s, fp-lib-table written",
        len(seen),
        pretty_dir,
    )


def write_pcb(
    design: PCBDesign,
    path: str | Path,
    *,
    fill_zones: bool = True,
    ipc_connection: object | None = None,
) -> None:
    """Serialise *design* and write it to a ``.kicad_pcb`` file.

    Args:
        design: The PCB design to write.
        path: Destination file path.  The parent directory must exist.
        fill_zones: If True, attempt to fill zones after writing.
        ipc_connection: Optional :class:`~kicad_pipeline.ipc.connection.KiCadConnection`.
            When provided, uses IPC to push the file and refill zones in the
            running KiCad instance.  Falls back to the subprocess approach
            if IPC zone fill fails.

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

    # Auto-generate project-local footprint library so "Update PCB from
    # Schematic" works in KiCad.  Detect the project name from the lib_id
    # prefix (all footprints use "{project_name}:{fp_name}" when
    # project_name was passed to build_pcb).
    _sync_footprint_library(design, dest.parent)

    # Patch PCB with schematic symbol paths so KiCad can correlate
    # footprints to symbols during "Update PCB from Schematic".
    sch_path = dest.with_suffix(".kicad_sch")
    if sch_path.exists():
        from kicad_pipeline.pcb.footprint_library import sync_pcb_to_schematic
        sync_pcb_to_schematic(dest, sch_path)

    if fill_zones:
        if ipc_connection is not None:
            _fill_zones_ipc(dest, ipc_connection)
        else:
            _fill_zones(dest)


def _fill_zones_ipc(pcb_path: Path, ipc_connection: object) -> None:
    """Fill copper zones using KiCad's IPC API, falling back to subprocess."""
    try:
        from kicad_pipeline.ipc.board_ops import push_pcb_to_kicad, refill_zones
        from kicad_pipeline.ipc.connection import KiCadConnection

        if not isinstance(ipc_connection, KiCadConnection):
            log.warning("ipc_connection is not a KiCadConnection; falling back to subprocess")
            _fill_zones(pcb_path)
            return

        push_pcb_to_kicad(pcb_path, ipc_connection)
        refill_zones(ipc_connection)
        log.info("Zone fill complete via IPC (%s)", pcb_path)
    except Exception as exc:
        log.warning(
            "IPC zone fill failed (%s), falling back to subprocess: %s",
            pcb_path, exc,
        )
        _fill_zones(pcb_path)


def _fill_zones(pcb_path: Path) -> None:
    """Fill copper zones using KiCad's Python API (``pcbnew``).

    KiCad 9's ``kicad-cli`` does not expose a ``fill-zones`` subcommand,
    but the bundled Python interpreter includes the ``pcbnew`` module.
    This function shells out to KiCad's Python to load the board,
    fill all zones, and save the result.

    Falls back to a log warning if KiCad's Python is not available.
    """
    import shutil
    import subprocess

    # Locate KiCad's bundled Python (macOS path)
    kicad_python = shutil.which("python3", path="/Applications/KiCad/KiCad.app"
                                "/Contents/Frameworks/Python.framework"
                                "/Versions/Current/bin")
    if kicad_python is None:
        log.warning(
            "Zone fill requires KiCad's bundled Python (pcbnew); "
            "CLI DRC will report GND stitching vias as dangling until "
            "zones are filled. (%s)", pcb_path,
        )
        return

    script = (
        "import pcbnew, sys\n"
        f"board = pcbnew.LoadBoard({str(pcb_path)!r})\n"
        "filler = pcbnew.ZONE_FILLER(board)\n"
        "zones = board.Zones()\n"
        "if zones.size() == 0:\n"
        "    sys.exit(0)\n"
        "filler.Fill(zones)\n"
        f"pcbnew.SaveBoard({str(pcb_path)!r}, board)\n"
    )
    try:
        result = subprocess.run(
            [kicad_python, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            log.info("Zone fill complete via pcbnew (%s)", pcb_path)
        else:
            log.warning(
                "Zone fill failed (exit %d): %s",
                result.returncode,
                result.stderr.strip()[:200],
            )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.warning("Zone fill unavailable: %s", exc)

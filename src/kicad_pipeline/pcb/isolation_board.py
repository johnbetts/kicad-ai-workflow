"""Build minimal single-component PCBs for isolation testing.

Generates a tiny board containing exactly one component so that
``kicad-image-gen`` can render it for visual verification without the
noise of a full design.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from kicad_pipeline.constants import LAYER_F_CU
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    PCBDesign,
    Point,
)
from kicad_pipeline.models.requirements import (
    Net,
    NetConnection,
    Pin,
    PinType,
)

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.validation.component_registry import ComponentSpec, PinSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pin type mapping: PinSpec.pin_type string -> PinType enum
# ---------------------------------------------------------------------------

_PIN_TYPE_MAP: dict[str, PinType] = {
    "input": PinType.INPUT,
    "output": PinType.OUTPUT,
    "power_in": PinType.POWER_IN,
    "power_out": PinType.POWER_OUT,
    "passive": PinType.PASSIVE,
    "bidirectional": PinType.BIDIRECTIONAL,
    "open_collector": PinType.OPEN_COLLECTOR,
    "no_connect": PinType.NO_CONNECT,
}

_DEFAULT_PIN_TYPE: PinType = PinType.PASSIVE

# ---------------------------------------------------------------------------
# Minimum board margin around the component body (mm)
# ---------------------------------------------------------------------------

_MARGIN_MM: float = 20.0
_MIN_BOARD_MM: float = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pin_spec_to_pin(pin_spec: PinSpec) -> Pin:
    """Convert a registry :class:`PinSpec` to a requirements :class:`Pin`."""
    pin_type = _PIN_TYPE_MAP.get(pin_spec.pin_type, _DEFAULT_PIN_TYPE)
    return Pin(
        number=pin_spec.number,
        name=pin_spec.name,
        pin_type=pin_type,
    )


def _board_size_from_footprint(
    fp: Footprint,
    body_w: float | None,
    body_h: float | None,
) -> tuple[float, float, float, float]:
    """Compute board dimensions and component placement from actual pad extent.

    Returns ``(board_w, board_h, place_x, place_y)`` where the component
    should be placed at ``(place_x, place_y)`` so all pads and body fit
    within the board with margin.
    """
    if not fp.pads:
        bw = body_w if body_w is not None else 5.0
        bh = body_h if body_h is not None else 5.0
        board_w = max(bw + _MARGIN_MM, _MIN_BOARD_MM)
        board_h = max(bh + _MARGIN_MM, _MIN_BOARD_MM)
        return board_w, board_h, board_w / 2.0, board_h / 2.0

    # Compute pad extent relative to footprint origin
    xs = [p.position.x for p in fp.pads]
    ys = [p.position.y for p in fp.pads]
    pad_x_min, pad_x_max = min(xs), max(xs)
    pad_y_min, pad_y_max = min(ys), max(ys)

    # Pad-based extent (add pad size margin)
    extent_w = (pad_x_max - pad_x_min) + 4.0  # ~2mm pad radius each side
    extent_h = (pad_y_max - pad_y_min) + 4.0

    # Also consider body size if larger
    bw = body_w if body_w is not None else extent_w
    bh = body_h if body_h is not None else extent_h
    extent_w = max(extent_w, bw)
    extent_h = max(extent_h, bh)

    board_w = max(extent_w + _MARGIN_MM, _MIN_BOARD_MM)
    board_h = max(extent_h + _MARGIN_MM, _MIN_BOARD_MM)

    # Place component so pad centroid is at board center
    pad_cx = (pad_x_min + pad_x_max) / 2.0
    pad_cy = (pad_y_min + pad_y_max) / 2.0
    place_x = board_w / 2.0 - pad_cx
    place_y = board_h / 2.0 - pad_cy

    return board_w, board_h, place_x, place_y


def _build_gnd_net(
    ref: str,
    pins: tuple[Pin, ...],
) -> tuple[Net, ...]:
    """Create a minimal GND net connecting to any power/ground pins.

    If no power pins exist the net is still created with zero connections
    so the board has at least a GND entry.
    """
    gnd_connections: list[NetConnection] = []
    for pin in pins:
        if pin.pin_type in (PinType.POWER_IN, PinType.POWER_OUT):
            name_lower = pin.name.lower()
            if "gnd" in name_lower or "vss" in name_lower or "ground" in name_lower:
                gnd_connections.append(NetConnection(ref=ref, pin=pin.number))

    return (Net(name="GND", connections=tuple(gnd_connections)),)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_isolation_footprint(
    spec: ComponentSpec,
    *,
    use_jlcpcb: bool = True,
) -> Footprint:
    """Generate a footprint for the given component spec.

    Calls :func:`footprint_for_component` with the spec's ref, value,
    footprint_id, and pins (converted to :class:`Pin` instances).

    When ``use_jlcpcb=True`` (default) and the spec has an ``lcsc``
    number, the JLCPCB cached footprint is used — this matches what
    real boards produce.  Set ``use_jlcpcb=False`` to force the
    parametric path (used for computing ``kicad_ref_pad1``).
    """
    from kicad_pipeline.pcb.footprints import footprint_for_component

    pins = tuple(_pin_spec_to_pin(ps) for ps in spec.pins)
    lcsc = spec.lcsc if use_jlcpcb else None

    logger.info(
        "Building isolation footprint for %s (%s, %s, lcsc=%s)",
        spec.component_id,
        spec.ref,
        spec.footprint_id,
        lcsc,
    )

    return footprint_for_component(
        ref=spec.ref,
        value=spec.value,
        footprint_id=spec.footprint_id,
        lcsc=lcsc,
        layer=LAYER_F_CU,
        pins=pins,
    )


def build_isolation_board(
    spec: ComponentSpec,
    output_dir: Path,
) -> Path:
    """Build a minimal PCB with one component and write to disk.

    Constructs a tiny board outline around the single component, generates
    the footprint, places it at the board center, and writes a valid
    ``.kicad_pcb`` file that ``kicad-image-gen`` can render.

    Returns:
        Path to the generated ``.kicad_pcb`` file.
    """
    from kicad_pipeline.pcb.builder import write_pcb

    pins = tuple(_pin_spec_to_pin(ps) for ps in spec.pins)

    # -- Generate footprint and compute board size from actual extent -------
    fp = build_isolation_footprint(spec)
    board_w, board_h, place_x, place_y = _board_size_from_footprint(
        fp, spec.body_width_mm, spec.body_height_mm,
    )
    fp = replace(fp, position=Point(x=place_x, y=place_y))

    # -- Board outline (rectangular) ----------------------------------------
    outline = BoardOutline(
        polygon=(
            Point(x=0.0, y=0.0),
            Point(x=board_w, y=0.0),
            Point(x=board_w, y=board_h),
            Point(x=0.0, y=board_h),
        ),
    )

    # -- Minimal net list (unconnected + GND) -------------------------------
    nets_list: list[NetEntry] = [NetEntry(number=0, name="")]
    gnd_nets = _build_gnd_net(spec.ref, pins)
    if gnd_nets:
        nets_list.append(NetEntry(number=1, name="GND"))

    # -- Assemble PCBDesign -------------------------------------------------
    design = PCBDesign(
        outline=outline,
        design_rules=DesignRules(),
        nets=tuple(nets_list),
        footprints=(fp,),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
        title=f"Isolation test: {spec.component_id}",
        revision="v0.1",
    )

    # -- Write to disk ------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = spec.component_id.replace("/", "_").replace(" ", "_")
    project_name = f"isolation_{safe_id}"
    pcb_path = output_dir / f"{project_name}.kicad_pcb"

    logger.info("Writing isolation board to %s", pcb_path)
    write_pcb(design, pcb_path, fill_zones=False)

    # -- Write schematic + project file so KiCad can open the full project ---
    _write_isolation_project(spec, pins, output_dir, project_name)

    return pcb_path


def _write_isolation_project(
    spec: ComponentSpec,
    pins: tuple[Pin, ...],
    output_dir: Path,
    project_name: str,
) -> None:
    """Write .kicad_sch and .kicad_pro alongside the .kicad_pcb.

    Creates a minimal single-component schematic and a KiCad project file
    so the isolation board can be opened as a full KiCad project.
    """
    from kicad_pipeline.models.requirements import (
        Component,
        FeatureBlock,
        MechanicalConstraints,
        Net,
        NetConnection,
        ProjectInfo,
        ProjectRequirements,
    )
    from kicad_pipeline.project_file import write_project_file

    try:
        from kicad_pipeline.schematic.builder import build_schematic, write_schematic
    except Exception:
        logger.debug("Schematic builder not available — skipping .kicad_sch")
        return

    # Build minimal requirements for the single component
    comp = Component(
        ref=spec.ref,
        value=spec.value,
        footprint=spec.footprint_id,
        lcsc=spec.lcsc,
        pins=pins,
    )
    # Simple GND net connecting any power/ground pins
    gnd_connections: list[NetConnection] = []
    for pin in pins:
        if pin.pin_type in (PinType.POWER_IN, PinType.POWER_OUT):
            name_lower = pin.name.lower()
            if "gnd" in name_lower or "vss" in name_lower:
                gnd_connections.append(NetConnection(ref=spec.ref, pin=pin.number))

    nets = (Net(name="GND", connections=tuple(gnd_connections)),) if gnd_connections else ()

    requirements = ProjectRequirements(
        project=ProjectInfo(
            name=project_name,
            description=f"Isolation test: {spec.component_id}",
        ),
        components=(comp,),
        nets=nets,
        features=(
            FeatureBlock(
                name="test",
                description=f"Isolation test for {spec.component_id}",
                components=(spec.ref,),
                nets=(),
                subcircuits=(),
            ),
        ),
        mechanical=MechanicalConstraints(
            board_width_mm=40.0,
            board_height_mm=30.0,
        ),
    )

    try:
        sch = build_schematic(requirements, compact=True, project_name=project_name)
        sch_path = output_dir / f"{project_name}.kicad_sch"
        write_schematic(sch, sch_path, project_name=project_name)
        logger.info("Schematic written: %s", sch_path)
    except Exception:
        logger.debug("Schematic generation failed — skipping", exc_info=True)

    try:
        write_project_file(project_name, output_dir)
        logger.info("Project file written: %s/%s.kicad_pro", output_dir, project_name)
    except Exception:
        logger.debug("Project file generation failed — skipping", exc_info=True)

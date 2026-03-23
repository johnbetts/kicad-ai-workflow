"""Shared test helpers for kicad_pipeline tests.

Provides common factory functions for PCBDesign, ProjectRequirements,
BoardOutline, Footprint, Pad, and Component objects used across many
test modules.  Import these instead of duplicating helpers in each file.
"""

from __future__ import annotations

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)

# ---------------------------------------------------------------------------
# Board outline
# ---------------------------------------------------------------------------


def make_board_outline(
    w: float = 80.0,
    h: float = 40.0,
) -> BoardOutline:
    """Rectangular board outline with configurable dimensions."""
    return BoardOutline(
        polygon=(
            Point(x=0.0, y=0.0),
            Point(x=w, y=0.0),
            Point(x=w, y=h),
            Point(x=0.0, y=h),
            Point(x=0.0, y=0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Pad / Footprint
# ---------------------------------------------------------------------------


def make_pad(
    number: str = "1",
    x: float = 0.0,
    y: float = 0.0,
    net_number: int = 0,
    net_name: str = "",
    size_x: float = 1.0,
    size_y: float = 1.0,
) -> Pad:
    """Minimal SMD pad with configurable position and net."""
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x=x, y=y),
        size_x=size_x,
        size_y=size_y,
        layers=("F.Cu",),
        net_number=net_number,
        net_name=net_name,
    )


def make_footprint(
    ref: str,
    x: float = 0.0,
    y: float = 0.0,
    lib_id: str = "R:R_0805",
    value: str = "10k",
    rotation: float = 0.0,
    pads: tuple[Pad, ...] | None = None,
) -> Footprint:
    """Minimal footprint with 2-pad default."""
    if pads is None:
        pads = (make_pad("1", -0.5, 0.0), make_pad("2", 0.5, 0.0))
    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(x=x, y=y),
        rotation=rotation,
        pads=pads,
    )


# ---------------------------------------------------------------------------
# PCBDesign
# ---------------------------------------------------------------------------


def make_pcb_design(
    footprints: tuple[Footprint, ...] = (),
    w: float = 80.0,
    h: float = 40.0,
    nets: tuple[NetEntry, ...] | None = None,
) -> PCBDesign:
    """Minimal PCBDesign with configurable outline and footprints."""
    if nets is None:
        nets = (NetEntry(number=0, name=""),)
    return PCBDesign(
        outline=make_board_outline(w, h),
        design_rules=DesignRules(),
        nets=nets,
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------


def make_component(
    ref: str,
    value: str = "10k",
    footprint: str = "R_0805",
    lcsc: str | None = None,
    pins: tuple[Pin, ...] = (),
    description: str | None = None,
) -> Component:
    """Minimal Component for requirements."""
    return Component(
        ref=ref,
        value=value,
        footprint=footprint,
        lcsc=lcsc,
        pins=pins,
        description=description,
    )


def make_pin(
    number: str,
    name: str = "",
    pin_type: PinType = PinType.PASSIVE,
    net: str | None = None,
) -> Pin:
    """Minimal Pin."""
    return Pin(number=number, name=name, pin_type=pin_type, function=None, net=net)


# ---------------------------------------------------------------------------
# ProjectRequirements
# ---------------------------------------------------------------------------


def make_requirements(
    components: tuple[Component, ...] | None = None,
    nets: tuple[Net, ...] = (),
    features: tuple[FeatureBlock, ...] | None = None,
    mechanical: MechanicalConstraints | None = None,
    w: float = 80.0,
    h: float = 40.0,
) -> ProjectRequirements:
    """Minimal ProjectRequirements with sane defaults.

    If *components* is ``None``, creates three default passives (R1, R2, C1).
    If *features* is ``None``, creates a single FeatureBlock containing all
    component refs.
    """
    if components is None:
        components = (
            make_component("R1", "10k", "R_0805"),
            make_component("R2", "4.7k", "R_0805"),
            make_component("C1", "100nF", "C_0805"),
        )
    if features is None:
        features = (
            FeatureBlock(
                name="Main",
                description="Test block",
                components=tuple(c.ref for c in components),
                nets=(),
                subcircuits=(),
            ),
        )
    if mechanical is None:
        mechanical = MechanicalConstraints(
            board_width_mm=w, board_height_mm=h,
        )
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=features,
        components=components,
        nets=nets,
        mechanical=mechanical,
    )

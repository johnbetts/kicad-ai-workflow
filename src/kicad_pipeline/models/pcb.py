"""Data models for KiCad PCB design representation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class TrackType(Enum):
    """Type of copper track segment."""

    TRACK = "track"
    ARC = "arc"


class ZoneFill(Enum):
    """Fill style for copper zones."""

    SOLID = "solid"
    HATCHED = "hatched"


class PlacementConstraintType(Enum):
    """Type of placement constraint for PCB component layout."""

    FIXED = "fixed"
    EDGE = "edge"
    NEAR = "near"
    GROUP = "group"
    AWAY_FROM = "away_from"


class BoardEdge(Enum):
    """Edge of the PCB board for edge-based placement."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class Point:
    """2D coordinate in mm."""

    x: float
    y: float


@dataclass(frozen=True)
class PlacementConstraint:
    """A placement constraint for a PCB component.

    Attributes:
        ref: Component reference designator.
        constraint_type: Type of placement constraint.
        target_ref: Reference of the target component (for NEAR/AWAY_FROM).
        edge: Board edge (for EDGE constraints).
        x: Fixed X position (for FIXED constraints).
        y: Fixed Y position (for FIXED constraints).
        rotation: Fixed rotation (for FIXED/EDGE constraints).
        max_distance_mm: Maximum distance from target (for NEAR).
        min_distance_mm: Minimum distance from target (for AWAY_FROM).
        group_name: Group identifier (for GROUP constraints).
        priority: Higher priority constraints are resolved first.
    """

    ref: str
    constraint_type: PlacementConstraintType
    target_ref: str | None = None
    target_pin: str | None = None
    edge: BoardEdge | None = None
    x: float | None = None
    y: float | None = None
    rotation: float | None = None
    max_distance_mm: float | None = None
    min_distance_mm: float | None = None
    group_name: str | None = None
    priority: int = 0
    layer: str | None = None


@dataclass(frozen=True)
class PlacementResult:
    """Result of constraint-based placement solving.

    Attributes:
        positions: Mapping from ref to placed position.
        rotations: Mapping from ref to placement rotation.
        violations: Descriptions of constraints that could not be satisfied.
    """

    positions: dict[str, Point]
    rotations: dict[str, float]
    violations: tuple[str, ...]


@dataclass(frozen=True)
class NetEntry:
    """A net number + name entry."""

    number: int
    name: str


@dataclass(frozen=True)
class Pad:
    """A footprint pad."""

    number: str  # "1", "A1", ""
    pad_type: str  # "smd", "thru_hole", "np_thru_hole"
    shape: str  # "rect", "roundrect", "circle", "oval"
    position: Point  # relative to footprint origin
    size_x: float
    size_y: float
    layers: tuple[str, ...]
    net_number: int | None = None
    net_name: str | None = None
    drill_diameter: float | None = None  # for thru_hole
    roundrect_ratio: float | None = None
    uuid: str = ""


@dataclass(frozen=True)
class FootprintText:
    """Text label on a footprint (ref, value, user)."""

    text_type: str  # "reference", "value", "user"
    text: str
    position: Point  # relative to footprint origin
    layer: str
    rotation: float = 0.0
    effects_size: float = 1.0
    hidden: bool = False
    uuid: str = ""


@dataclass(frozen=True)
class FootprintLine:
    """A line on a footprint layer (courtyard, fab, silkscreen)."""

    start: Point
    end: Point
    layer: str
    width: float = 0.12
    uuid: str = ""


@dataclass(frozen=True)
class FootprintArc:
    """An arc on a footprint layer."""

    start: Point
    mid: Point
    end: Point
    layer: str
    width: float = 0.12
    uuid: str = ""


@dataclass(frozen=True)
class FootprintCircle:
    """A circle on a footprint layer."""

    center: Point
    end: Point  # KiCad stores a point on the circumference
    layer: str
    width: float = 0.12
    uuid: str = ""


FootprintGraphic: TypeAlias = FootprintLine | FootprintArc | FootprintCircle


@dataclass(frozen=True)
class Footprint3DModel:
    """Reference to a 3D model file for a footprint.

    Attributes:
        path: Model file path, typically using ``${KICAD9_3DMODEL_DIR}/...``.
        offset: Translation offset ``(x, y, z)`` in mm.
        scale: Scale factors ``(x, y, z)``.
        rotate: Rotation angles ``(x, y, z)`` in degrees.
    """

    path: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Footprint:
    """A placed footprint on the PCB."""

    lib_id: str  # "R_0805:R_0805_2012Metric"
    ref: str  # "R1"
    value: str  # "10k"
    position: Point  # board coordinates
    rotation: float = 0.0
    layer: str = "F.Cu"
    pads: tuple[Pad, ...] = ()
    graphics: tuple[FootprintLine | FootprintArc | FootprintCircle, ...] = ()
    texts: tuple[FootprintText, ...] = ()
    lcsc: str | None = None
    uuid: str = ""
    attr: str = "smd"  # "smd", "through_hole"
    models: tuple[Footprint3DModel, ...] = ()
    datasheet: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class Track:
    """A copper track segment."""

    start: Point
    end: Point
    width: float
    layer: str
    net_number: int
    uuid: str = ""


@dataclass(frozen=True)
class Via:
    """A via connecting layers."""

    position: Point
    drill: float
    size: float
    layers: tuple[str, ...]  # ("F.Cu", "B.Cu") or ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
    net_number: int
    uuid: str = ""


@dataclass(frozen=True)
class ZonePolygon:
    """A copper zone (pour)."""

    net_number: int
    net_name: str
    layer: str
    name: str
    polygon: tuple[Point, ...]  # outline points
    min_thickness: float = 0.25
    fill: ZoneFill = ZoneFill.SOLID
    thermal_relief_gap: float = 0.3
    thermal_relief_bridge: float = 0.5
    clearance_mm: float = 0.3
    filled_polygons: tuple[tuple[Point, ...], ...] = ()
    uuid: str = ""


@dataclass(frozen=True)
class Keepout:
    """A keepout zone on one or more layers."""

    polygon: tuple[Point, ...]
    layers: tuple[str, ...]
    no_copper: bool = True
    no_vias: bool = False
    no_tracks: bool = False
    uuid: str = ""
    tag: str = ""
    """Optional tag to classify keepout type (e.g. ``'mounting_hole'``, ``'antenna'``)."""


@dataclass(frozen=True)
class BoardOutline:
    """PCB board outline (Edge.Cuts)."""

    polygon: tuple[Point, ...]  # ordered points forming closed polygon
    width: float = 0.05  # Edge.Cuts line width


@dataclass(frozen=True)
class NetClass:
    """A net classification with electrical rules.

    Groups nets by function (power, analog, digital) and assigns
    appropriate trace widths, clearances, and via sizes.
    """

    name: str
    clearance_mm: float = 0.2
    trace_width_mm: float = 0.25
    via_diameter_mm: float = 0.8
    via_drill_mm: float = 0.508
    diff_pair_width_mm: float = 0.2
    diff_pair_gap_mm: float = 0.25
    guard_traces: bool = False
    nets: tuple[str, ...] = ()


@dataclass(frozen=True)
class DesignRules:
    """PCB design rules / net class configuration."""

    default_trace_width_mm: float = 0.25
    power_trace_width_mm: float = 0.5
    usb_trace_width_mm: float = 0.3
    analog_trace_width_mm: float = 0.2
    default_clearance_mm: float = 0.2
    min_via_drill_mm: float = 0.508
    min_via_diameter_mm: float = 0.9
    copper_pour_clearance_mm: float = 0.2
    layer_count: int = 2


@dataclass(frozen=True)
class PCBDesign:
    """Complete KiCad PCB design."""

    outline: BoardOutline
    design_rules: DesignRules
    nets: tuple[NetEntry, ...]
    footprints: tuple[Footprint, ...]
    tracks: tuple[Track, ...]
    vias: tuple[Via, ...]
    zones: tuple[ZonePolygon, ...]
    keepouts: tuple[Keepout, ...]
    version: int = 20241229
    generator: str = "kicad-ai-pipeline"
    generator_version: str = "9.0"
    netclasses: tuple[NetClass, ...] = ()
    drc_exclusions: tuple[str, ...] = ()
    title: str = ""
    date: str = ""
    revision: str = ""
    company: str = ""

    def get_footprint(self, ref: str) -> Footprint | None:
        """Return footprint by ref, or None."""
        for fp in self.footprints:
            if fp.ref == ref:
                return fp
        return None

    def get_net_number(self, name: str) -> int | None:
        """Return net number by name, or None."""
        for n in self.nets:
            if n.name == name:
                return n.number
        return None

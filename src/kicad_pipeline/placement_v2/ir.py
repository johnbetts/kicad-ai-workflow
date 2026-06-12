"""Constraint IR — the single source of truth for placement intent.

Every placement rule is a frozen dataclass here. The same constraint
objects are consumed by the solvers (cell generators, floorplanner)
and by the sign-off verifier, so enforcement and checking can never
drift apart. Constraints are compiled from three sources, in priority
order: netlist topology, per-part-class rules, and persisted human
feedback locks (see ``compile.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kicad_pipeline.models.pcb import Point

Polygon = tuple[Point, ...]
"""Ordered polygon vertices in mm. Frame depends on context (cell-local
for :class:`CellKeepout`, board frame after placement)."""


class Edge(Enum):
    """A board edge in the PCB coordinate convention (Y grows down)."""

    NORTH = "north"  # y = y_min
    SOUTH = "south"  # y = y_max
    EAST = "east"  # x = x_max
    WEST = "west"  # x = x_min


class Axis(Enum):
    """Layout axis for sequence constraints."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class KeepoutKind(Enum):
    """Why a keepout region exists; drives what may violate it."""

    RF_ANTENNA = "rf_antenna"  # no copper, no components
    ISOLATION = "isolation"  # creepage slot / clearance region
    THERMAL = "thermal"  # heat-sensitive exclusion


class Severity(Enum):
    """Violation severity. CRITICAL and MAJOR block sign-off."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class ConstraintSource(Enum):
    """Where a constraint came from (priority: feedback > part > netlist)."""

    NETLIST = "netlist"
    PART_RULE = "part_rule"
    HUMAN_FEEDBACK = "human_feedback"


@dataclass(frozen=True)
class PadRef:
    """A specific pad: component ref + pin number."""

    ref: str  # "U1"
    pin: str  # "3", "A1"

    def __str__(self) -> str:
        return f"{self.ref}.{self.pin}"


@dataclass(frozen=True)
class PinAttach:
    """*src* pad must sit within *max_mm* of *dst* pad (edge-to-edge intent).

    The canonical decoupling/ESD/snubber constraint: the cap's pad is
    attached to the IC pad it serves, identified by the shared net.
    """

    src: PadRef  # the supporting component's pad (e.g. C3.1)
    dst: PadRef  # the served pad (e.g. U1.VDD pad)
    net: str  # shared net that justifies the attachment
    max_mm: float  # hard limit, verified at sign-off
    ideal_mm: float  # solver target (<= max_mm)
    source: ConstraintSource = ConstraintSource.NETLIST


@dataclass(frozen=True)
class SequenceAlong:
    """*refs* must appear in order along *axis* (signal-flow chains, arrays).

    Verified by checking the relevant coordinate of each ref is strictly
    monotonic in sequence order. ``pitch_mm`` of ``None`` means spacing
    is free; a value pins members to a fixed pitch (relay/ADC arrays).
    """

    axis: Axis
    refs: tuple[str, ...]
    pitch_mm: float | None = None
    max_span_mm: float | None = None  # whole chain must fit in this span
    source: ConstraintSource = ConstraintSource.NETLIST


@dataclass(frozen=True)
class EdgePin:
    """*ref* must sit on a board edge with its opening facing outward.

    ``opening`` is the wire/jack entry direction in the FOOTPRINT frame
    at rotation 0 (from part rules) — needed when the courtyard is
    symmetric and the opening cannot be derived from geometry (Phoenix
    terminal blocks).
    """

    ref: str
    edge: Edge | None = None  # None = solver picks nearest edge
    face_out: bool = True
    max_edge_distance_mm: float = 5.0
    opening: tuple[float, float] | None = None
    source: ConstraintSource = ConstraintSource.NETLIST


@dataclass(frozen=True)
class AttachBundle:
    """Straight pad-to-pad connections between two parts must not cross.

    Compiled from the netlist: when two components are joined by two or
    more two-pin nets (relay contacts -> screw terminal, ADC channel ->
    its connector), the pin ORDER on one side must mirror the other's
    physical pad order or the traces cross. Gate C feedback 2026-06-11
    item 2 (relay NO/NC) converted to this countable rule: segment
    intersections between the attach lines must be zero.
    """

    ref_a: str
    ref_b: str
    pad_pairs: tuple[tuple[PadRef, PadRef], ...]  # (a-side pad, b-side pad) per net
    nets: tuple[str, ...]  # net names, parallel to pad_pairs
    source: ConstraintSource = ConstraintSource.NETLIST


@dataclass(frozen=True)
class FanoutLine:
    """One ratsnest line leaving a connector pad.

    ``candidates`` lists every other pad on the net; the verifier draws
    the line to the NEAREST candidate in board space — the deterministic
    proxy for the rendered ratsnest MST edge (global nets like GND have
    many pads, but the visible line goes to the closest one).
    """

    net: str
    src: PadRef  # the connector's pad
    candidates: tuple[PadRef, ...]  # every other pad on the net


@dataclass(frozen=True)
class ConnectorFanout:
    """All ratsnest lines leaving one connector must be crossing-free.

    Human finding 2026-06-11 (analog board): each terminal's AIN line
    went up-right while its GND line went up-left — an X on all four
    channels that AttachBundle could not see because GND is a global
    net. Verified by pairwise segment intersection over the nearest-
    candidate lines.
    """

    ref: str  # the free-pin-order connector
    lines: tuple[FanoutLine, ...]
    source: ConstraintSource = ConstraintSource.NETLIST


@dataclass(frozen=True)
class GroupAssoc:
    """An edge connector functionally owned by one FeatureBlock.

    Compiled when the majority of the connector's SIGNAL-net partners
    live in a single FeatureBlock (J14's nine SPI/TFT nets all end at
    U3 -> the MCU block). The connector is that group's EDGE SUBGROUP:
    it must claim the edge SEGMENT adjacent to the parent group's
    placement — group first, connector second (board owner directive,
    2026-06-11 evening addendum). Verified at Gate A by projecting the
    parent hull onto the connector's edge axis: the connector centroid
    must fall inside the projection interval ± ``tolerance_mm``.
    """

    ref: str  # the connector ("J14")
    group: str  # parent FeatureBlock name ("MCU")
    partner_refs: tuple[str, ...]  # majority signal-net partners ("U3", ...)
    tolerance_mm: float = 15.0
    source: ConstraintSource = ConstraintSource.NETLIST


@dataclass(frozen=True)
class CellKeepout:
    """A keepout region owned by a component, in CELL-LOCAL frame.

    The polygon transforms with the owning cell (rotation included), so
    "the keepout is under the antenna" is true by construction. The
    verifier re-derives the board-frame polygon from the owner's final
    position and rotation and asserts the board file matches.
    """

    owner: str  # ref of the owning component ("U1")
    polygon: Polygon  # local frame, relative to owner origin
    kind: KeepoutKind
    source: ConstraintSource = ConstraintSource.PART_RULE


@dataclass(frozen=True)
class IsolationGap:
    """Minimum edge-to-edge gap between two voltage domains' cells."""

    domain_a: str  # e.g. "MAINS"
    domain_b: str  # e.g. "LOGIC"
    min_mm: float
    source: ConstraintSource = ConstraintSource.PART_RULE


@dataclass(frozen=True)
class BoardContain:
    """Every pad, courtyard, and 3D body must sit inside the outline."""

    margin_mm: float = 0.5
    source: ConstraintSource = ConstraintSource.NETLIST


@dataclass(frozen=True)
class Violation:
    """A failed constraint check, produced by generators and the verifier.

    ``measured`` vs ``limit`` makes every violation quantitative — there
    is no "looks wrong" verdict at this layer.
    """

    constraint: str  # repr of the violated constraint
    refs: tuple[str, ...]
    severity: Severity
    measured: float
    limit: float
    message: str


@dataclass(frozen=True)
class ConstraintSet:
    """The complete compiled placement intent for one board."""

    pin_attach: tuple[PinAttach, ...] = ()
    sequences: tuple[SequenceAlong, ...] = ()
    edge_pins: tuple[EdgePin, ...] = ()
    keepouts: tuple[CellKeepout, ...] = ()
    isolation: tuple[IsolationGap, ...] = ()
    bundles: tuple[AttachBundle, ...] = ()
    fanouts: tuple[ConnectorFanout, ...] = ()
    group_assocs: tuple[GroupAssoc, ...] = ()
    contain: BoardContain = field(default_factory=BoardContain)

    def for_refs(self, refs: frozenset[str]) -> ConstraintSet:
        """Subset of constraints whose participants all lie within *refs*.

        Used by cell generators to extract the constraints they must
        prove internally; board-level constraints (edge pins, isolation,
        containment) are kept only when they name a member ref.
        """
        return ConstraintSet(
            pin_attach=tuple(
                c for c in self.pin_attach
                if c.src.ref in refs and c.dst.ref in refs
            ),
            sequences=tuple(
                c for c in self.sequences if all(r in refs for r in c.refs)
            ),
            edge_pins=tuple(c for c in self.edge_pins if c.ref in refs),
            keepouts=tuple(c for c in self.keepouts if c.owner in refs),
            isolation=(),  # domain-level, never intra-cell
            bundles=tuple(
                b for b in self.bundles if b.ref_a in refs and b.ref_b in refs
            ),
            fanouts=(),  # board-level: candidates span the whole netlist
            group_assocs=(),  # board-level: parent hull spans other cells
            contain=self.contain,
        )

    def merged_with(self, other: ConstraintSet) -> ConstraintSet:
        """Union of two constraint sets (containment: tighter margin wins)."""
        contain = (
            self.contain
            if self.contain.margin_mm >= other.contain.margin_mm
            else other.contain
        )
        return ConstraintSet(
            pin_attach=self.pin_attach + other.pin_attach,
            sequences=self.sequences + other.sequences,
            edge_pins=self.edge_pins + other.edge_pins,
            keepouts=self.keepouts + other.keepouts,
            isolation=self.isolation + other.isolation,
            bundles=self.bundles + other.bundles,
            fanouts=self.fanouts + other.fanouts,
            group_assocs=self.group_assocs + other.group_assocs,
            contain=contain,
        )

    def count(self) -> int:
        """Total number of constraints (containment counts as one)."""
        return (
            len(self.pin_attach)
            + len(self.sequences)
            + len(self.edge_pins)
            + len(self.keepouts)
            + len(self.isolation)
            + len(self.bundles)
            + len(self.fanouts)
            + len(self.group_assocs)
            + 1
        )

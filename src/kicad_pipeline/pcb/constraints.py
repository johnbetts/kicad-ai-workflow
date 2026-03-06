"""Constraint-based PCB placement solver.

Replaces the keyword-based zone grid-fill with a constraint-driven approach
that understands fixed positions, edge connectors, decoupling proximity,
signal paths, and board templates.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import (
    BoardEdge,
    PlacementConstraint,
    PlacementConstraintType,
    PlacementResult,
    Point,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import BoardOutline, Keepout
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.pcb.board_templates import BoardTemplate

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------

_DEFAULT_GRID_MM: float = 0.5

PLACEMENT_GAP_MM: float = 0.5
"""Minimum gap between footprint courtyards to prevent solder mask bridging."""

_THT_GAP_MM: float = 1.0
"""Larger gap for through-hole components (bigger pads, solder ring)."""

_THT_SIZE_THRESHOLD_MM: float = 5.0
"""Footprint dimension above which the THT gap is used."""


def _placement_gap(w: float, h: float) -> float:
    """Return placement gap based on footprint dimensions."""
    return _THT_GAP_MM if max(w, h) > _THT_SIZE_THRESHOLD_MM else PLACEMENT_GAP_MM


class _OccupancyGrid:
    """2D boolean grid tracking placed courtyards and keepouts.

    All operations are in board coordinates (mm). Internally, coordinates
    are mapped to integer grid cells at *grid_mm* resolution.
    """

    def __init__(
        self, width_mm: float, height_mm: float, grid_mm: float = _DEFAULT_GRID_MM,
    ) -> None:
        self._grid = grid_mm
        self._cols = max(1, math.ceil(width_mm / grid_mm))
        self._rows = max(1, math.ceil(height_mm / grid_mm))
        self._cells: list[list[bool]] = [
            [False] * self._cols for _ in range(self._rows)
        ]

    def _to_grid(self, mm: float) -> int:
        return round(mm / self._grid)

    def mark_rect(self, x: float, y: float, w: float, h: float) -> None:
        """Mark a rectangular area as occupied."""
        c0 = max(0, self._to_grid(x))
        r0 = max(0, self._to_grid(y))
        c1 = min(self._cols, self._to_grid(x + w))
        r1 = min(self._rows, self._to_grid(y + h))
        for r in range(r0, r1):
            for c in range(c0, c1):
                self._cells[r][c] = True

    def is_rect_free(self, x: float, y: float, w: float, h: float) -> bool:
        """Check whether a rectangular area is entirely unoccupied.

        Returns ``False`` if the rectangle extends beyond the grid boundary
        (i.e. off the board), ensuring no component is placed out of bounds.
        """
        c0 = self._to_grid(x)
        r0 = self._to_grid(y)
        c1 = self._to_grid(x + w)
        r1 = self._to_grid(y + h)
        # Reject rectangles that extend beyond the board
        if c0 < 0 or r0 < 0 or c1 > self._cols or r1 > self._rows:
            return False
        for r in range(r0, r1):
            for c in range(c0, c1):
                if self._cells[r][c]:
                    return False
        return True

    def find_nearest_free(
        self, target_x: float, target_y: float, w: float, h: float,
    ) -> tuple[float, float] | None:
        """Find the nearest free rectangle to ``(target_x, target_y)``.

        Searches in expanding rings around the target position.

        Returns:
            ``(x, y)`` of the nearest free position, or ``None`` if the grid
            is fully occupied.
        """
        best: tuple[float, float] | None = None
        best_dist = float("inf")

        # Search in a spiral from target
        max_r = max(self._rows, self._cols)
        tx = self._to_grid(target_x)
        ty = self._to_grid(target_y)

        for radius in range(max_r):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue  # Only check the ring perimeter
                    r = ty + dr
                    c = tx + dc
                    if r < 0 or c < 0 or r >= self._rows or c >= self._cols:
                        continue
                    x_mm = c * self._grid
                    y_mm = r * self._grid
                    if self.is_rect_free(x_mm, y_mm, w, h):
                        dist = math.hypot(x_mm - target_x, y_mm - target_y)
                        if dist < best_dist:
                            best_dist = dist
                            best = (x_mm, y_mm)
            if best is not None:
                return best  # Found on this ring, no closer exists
        return best


# ---------------------------------------------------------------------------
# Edge position calculator
# ---------------------------------------------------------------------------


def _edge_position(
    edge: BoardEdge,
    board_w: float,
    board_h: float,
    comp_w: float,
    comp_h: float,
    offset: float,
    margin: float = 3.0,
) -> tuple[float, float, float]:
    """Calculate ``(x, y, rotation)`` for placing a component along a board edge.

    The component's centre is placed at the board edge with *margin* mm
    inward, and offset along the edge by *offset* mm from the edge start.

    Args:
        edge: Which board edge to place on.
        board_w: Board width in mm.
        board_h: Board height in mm.
        comp_w: Component width in mm.
        comp_h: Component height in mm.
        offset: Position along the edge in mm from the edge start.
        margin: Inset from the board edge in mm.

    Returns:
        ``(x, y, rotation)`` tuple.
    """
    if edge == BoardEdge.TOP:
        return (offset, margin + comp_h / 2.0, 0.0)
    elif edge == BoardEdge.BOTTOM:
        return (offset, board_h - margin - comp_h / 2.0, 180.0)
    elif edge == BoardEdge.LEFT:
        return (margin + comp_h / 2.0, offset, 90.0)
    else:  # RIGHT
        return (board_w - margin - comp_h / 2.0, offset, 270.0)


# ---------------------------------------------------------------------------
# Constraint generation from requirements
# ---------------------------------------------------------------------------


def _is_connector(ref: str, footprint_id: str) -> bool:
    """Return True if component looks like a connector."""
    prefix = "".join(ch for ch in ref if ch.isalpha()).upper()
    if prefix == "J":
        return True
    fp_upper = footprint_id.upper()
    return any(
        kw in fp_upper
        for kw in ("TERMINAL", "HEADER", "PINSOCKET", "PINHEADER", "CONNECTOR")
    )


def _is_screw_terminal(footprint_id: str) -> bool:
    """Return True if footprint looks like a screw terminal or terminal block."""
    fp_upper = footprint_id.upper()
    return any(
        kw in fp_upper
        for kw in ("TERMINAL", "SCREW", "TERMINALBLOCK", "PHOENIX")
    )


def _connector_edge(
    ref: str,
    footprint_id: str,
    board_template_name: str | None = None,
) -> BoardEdge:
    """Choose the best board edge for a connector based on its type.

    Screw terminals and terminal blocks go on LEFT or BOTTOM edges.
    Pin headers default to TOP.  GPIO headers on RPi HATs stay FIXED
    (handled separately).

    Args:
        ref: Component reference designator.
        footprint_id: Footprint identifier string.
        board_template_name: Name of the active board template, if any.

    Returns:
        The recommended :class:`BoardEdge` for the connector.
    """
    if _is_screw_terminal(footprint_id):
        # RPi HATs: screw terminals on BOTTOM (opposite GPIO header at top)
        if board_template_name == "RPI_HAT":
            return BoardEdge.BOTTOM
        # Other boards: alternate between LEFT and BOTTOM
        num = int("".join(ch for ch in ref if ch.isdigit()) or "0")
        return BoardEdge.LEFT if num % 2 == 1 else BoardEdge.BOTTOM
    fp_upper = footprint_id.upper()
    if "PINSOCKET" in fp_upper or "PINHEADER" in fp_upper:
        return BoardEdge.TOP
    return BoardEdge.LEFT


def _is_decoupling_cap(ref: str, value: str) -> bool:
    """Return True if component looks like a decoupling capacitor."""
    if not ref.startswith("C"):
        return False
    val_lower = value.lower()
    return any(
        kw in val_lower
        for kw in ("nf", "pf", "100n", "10n", "1u", "10u", "0.1u", "4.7u")
    )


def _is_power_net(net_name: str) -> bool:
    """Return True if *net_name* is a power or ground net."""
    name = net_name.upper().strip()
    if name in ("GND", "AGND", "DGND", "PGND", "VGND", "VSS", "VEE"):
        return True
    return name.startswith("+") or name.startswith("V") or name.startswith("-")


def build_signal_adjacency(
    requirements: ProjectRequirements,
) -> dict[str, set[str]]:
    """Build an adjacency graph of components connected by signal nets.

    Power/ground nets (GND, +3V3, VCC, etc.) are excluded since they
    connect most components and would defeat the purpose of grouping.

    Args:
        requirements: Project requirements with components and nets.

    Returns:
        Adjacency dict mapping each ref to the set of refs it is
        connected to via signal nets.
    """
    adj: dict[str, set[str]] = {}
    for comp in requirements.components:
        adj[comp.ref] = set()

    for net in requirements.nets:
        if _is_power_net(net.name):
            continue
        refs_in_net = [c.ref for c in net.connections]
        for i, r1 in enumerate(refs_in_net):
            for r2 in refs_in_net[i + 1:]:
                adj.setdefault(r1, set()).add(r2)
                adj.setdefault(r2, set()).add(r1)

    return adj


def trace_linear_chains(
    adj: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    """Trace linear chains in the signal adjacency graph.

    A linear chain is a sequence of components where interior nodes have
    exactly degree 2 in the adjacency graph (i.e. they connect to exactly
    two signal neighbours). Chains start and end at components with
    degree != 2 (endpoints, tees, or isolated).

    Args:
        adj: Adjacency dict from :func:`build_signal_adjacency`.

    Returns:
        List of chains, each a tuple of ref strings in order.
    """
    visited: set[tuple[str, str]] = set()
    chains: list[tuple[str, ...]] = []

    # Find chain start points: nodes with degree != 2
    # These are endpoints (degree 1) or branch points (degree 3+)
    start_nodes = [ref for ref, neighbours in adj.items() if len(neighbours) != 2]
    # Also consider isolated nodes (degree 0), but they aren't chains
    start_nodes = [ref for ref in start_nodes if len(adj.get(ref, set())) > 0]

    for start in start_nodes:
        for neighbour in adj.get(start, set()):
            if (start, neighbour) in visited or (neighbour, start) in visited:
                continue
            # Walk the chain from start through neighbour
            chain = [start]
            prev = start
            current = neighbour
            while True:
                chain.append(current)
                visited.add((prev, current))
                visited.add((current, prev))
                neighbours = adj.get(current, set()) - {prev}
                if len(adj.get(current, set())) != 2:
                    break  # End of chain (branch point or endpoint)
                if not neighbours:
                    break
                nxt = next(iter(neighbours))
                prev = current
                current = nxt
            if len(chain) >= 2:
                chains.append(tuple(chain))

    # Handle pure cycles (all nodes have degree 2) — pick any unvisited
    for ref, neighbours in adj.items():
        if len(neighbours) == 2 and not any(
            (ref, n) in visited for n in neighbours
        ):
            chain = [ref]
            prev = ref
            current = next(iter(neighbours))
            while current != ref:
                chain.append(current)
                visited.add((prev, current))
                visited.add((current, prev))
                neighbours_set = adj.get(current, set()) - {prev}
                if not neighbours_set:
                    break
                prev = current
                current = next(iter(neighbours_set))
            if len(chain) >= 2:
                chains.append(tuple(chain))

    return chains


def constraints_from_requirements(
    requirements: ProjectRequirements,
    board_template: BoardTemplate | None,
    footprint_sizes: dict[str, tuple[float, float]],
) -> tuple[PlacementConstraint, ...]:
    """Auto-generate placement constraints from project requirements.

    Generates constraints based on:

    - Template fixed components -> FIXED
    - Connectors (J*, terminal/header) -> EDGE
    - Decoupling caps sharing power nets with ICs -> NEAR(IC, 3mm)
    - Same FeatureBlock -> GROUP
    - All components -> AWAY_FROM mounting holes

    Args:
        requirements: Project requirements document.
        board_template: Optional board template for fixed positions.
        footprint_sizes: Mapping from ref to ``(width, height)`` in mm.

    Returns:
        Tuple of :class:`PlacementConstraint` objects.
    """
    constraints: list[PlacementConstraint] = []

    # 1. Template fixed components
    if board_template is not None:
        for fc in board_template.fixed_components:
            for comp in requirements.components:
                if comp.ref == fc.ref_pattern:
                    constraints.append(PlacementConstraint(
                        ref=comp.ref,
                        constraint_type=PlacementConstraintType.FIXED,
                        x=fc.x_mm,
                        y=fc.y_mm,
                        rotation=fc.rotation,
                        priority=100,
                    ))
                    break

    # 2. Build net-to-refs map for proximity analysis
    net_refs: dict[str, set[str]] = {}
    for net in requirements.nets:
        refs_in_net: set[str] = set()
        for conn in net.connections:
            refs_in_net.add(conn.ref)
        net_refs[net.name] = refs_in_net

    # Identify ICs (U* refs)
    ic_refs: set[str] = set()
    for comp in requirements.components:
        if comp.ref.startswith("U"):
            ic_refs.add(comp.ref)

    # 3. Connectors -> EDGE (with type-aware edge selection)
    tmpl_name = board_template.name if board_template is not None else None
    for comp in requirements.components:
        if _is_connector(comp.ref, comp.footprint):
            # Skip if already fixed by template
            if any(
                c.ref == comp.ref and c.constraint_type == PlacementConstraintType.FIXED
                for c in constraints
            ):
                continue
            edge = _connector_edge(comp.ref, comp.footprint, tmpl_name)
            constraints.append(PlacementConstraint(
                ref=comp.ref,
                constraint_type=PlacementConstraintType.EDGE,
                edge=edge,
                priority=50,
            ))

    # 4. Decoupling caps -> NEAR their associated IC (pin-level targeting)
    _gnd_names = {"GND", "AGND", "DGND", "VSS", "GNDA", "GNDD"}
    for comp in requirements.components:
        if _is_decoupling_cap(comp.ref, comp.value):
            # Find IC sharing a POWER net (skip GND — target the supply pin)
            for net in requirements.nets:
                if net.name.upper() in _gnd_names:
                    continue  # skip ground nets
                cap_in_net = any(c.ref == comp.ref for c in net.connections)
                if not cap_in_net:
                    continue
                for conn in net.connections:
                    if conn.ref in ic_refs and conn.ref != comp.ref:
                        constraints.append(PlacementConstraint(
                            ref=comp.ref,
                            constraint_type=PlacementConstraintType.NEAR,
                            target_ref=conn.ref,
                            target_pin=conn.pin,
                            max_distance_mm=5.0,
                            min_distance_mm=3.0,
                            priority=30,
                        ))
                        break
                else:
                    continue
                break

    # 4b. Passives sharing signal nets with ICs/switches/connectors -> NEAR
    passive_prefixes = ("R", "C", "L", "D")
    dominant_prefixes = ("U", "J", "P", "Q", "SW")
    already_near = {c.ref for c in constraints
                    if c.constraint_type == PlacementConstraintType.NEAR}

    for comp in requirements.components:
        ref_alpha = "".join(ch for ch in comp.ref if ch.isalpha())
        if ref_alpha not in passive_prefixes:
            continue
        if comp.ref in already_near:
            continue  # already has NEAR from decoupling cap logic (section 4)

        # Find dominant component sharing a signal net
        best_target: tuple[str, str] | None = None
        for net in requirements.nets:
            if _is_power_net(net.name):
                continue
            comp_in_net = any(c.ref == comp.ref for c in net.connections)
            if not comp_in_net:
                continue
            for conn in net.connections:
                conn_alpha = "".join(ch for ch in conn.ref if ch.isalpha())
                if conn_alpha in dominant_prefixes and conn.ref != comp.ref:
                    best_target = (conn.ref, conn.pin)
                    break
            if best_target:
                break

        if best_target:
            constraints.append(PlacementConstraint(
                ref=comp.ref,
                constraint_type=PlacementConstraintType.NEAR,
                target_ref=best_target[0],
                target_pin=best_target[1],
                max_distance_mm=5.0,
                priority=25,
            ))

    # 5. FeatureBlock -> GROUP
    for fb in requirements.features:
        if len(fb.components) > 1:
            for ref in fb.components:
                # Don't override higher-priority constraints
                if any(c.ref == ref and c.priority >= 20 for c in constraints):
                    continue
                constraints.append(PlacementConstraint(
                    ref=ref,
                    constraint_type=PlacementConstraintType.GROUP,
                    group_name=fb.name,
                    priority=10,
                ))

    # 6. Signal-path chains -> GROUP (higher priority than FeatureBlock)
    adj = build_signal_adjacency(requirements)
    chains = trace_linear_chains(adj)
    for chain_idx, chain in enumerate(chains):
        if len(chain) < 2:
            continue
        chain_group = f"_signal_chain_{chain_idx}"
        for ref in chain:
            # Only upgrade if current constraint is lower priority
            if any(c.ref == ref and c.priority >= 15 for c in constraints):
                continue
            # Remove any existing GROUP at priority 10
            constraints = [
                c for c in constraints
                if not (c.ref == ref and c.constraint_type == PlacementConstraintType.GROUP
                        and c.priority < 15)
            ]
            constraints.append(PlacementConstraint(
                ref=ref,
                constraint_type=PlacementConstraintType.GROUP,
                group_name=chain_group,
                priority=15,
            ))

    # 7. Net-based grouping: components sharing the same signal net
    # This captures star topologies (voltage dividers, filter networks)
    # that trace_linear_chains cannot detect.  Priority 16 overrides
    # signal chain groups (15) but not edge/near/fixed constraints (>=20).
    for net in requirements.nets:
        if _is_power_net(net.name):
            continue
        net_group_refs = [c.ref for c in net.connections]
        if len(net_group_refs) < 3:
            continue
        group_name = f"_net_group_{net.name}"
        for ref in net_group_refs:
            # Don't override higher-priority constraints (NEAR, EDGE, FIXED)
            if any(c.ref == ref and c.priority > 16 for c in constraints):
                continue
            # Remove existing lower-priority GROUP constraints for this ref
            constraints = [
                c for c in constraints
                if not (c.ref == ref
                        and c.constraint_type == PlacementConstraintType.GROUP
                        and c.priority <= 16)
            ]
            constraints.append(PlacementConstraint(
                ref=ref,
                constraint_type=PlacementConstraintType.GROUP,
                group_name=group_name,
                priority=16,
            ))

    return tuple(constraints)


# ---------------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------------


def solve_placement(
    constraints: tuple[PlacementConstraint, ...],
    board_outline: BoardOutline,
    footprint_sizes: dict[str, tuple[float, float]],
    keepouts: tuple[Keepout, ...] = (),
    grid_mm: float = _DEFAULT_GRID_MM,
    requirements: ProjectRequirements | None = None,
) -> PlacementResult:
    """Solve component placement using a greedy constraint-based algorithm.

    Placement order:

    1. FIXED — placed at exact positions (highest priority).
    2. EDGE — placed along board edges with spacing and rotation.
    3. NEAR — placed adjacent to already-placed target components.
    4. GROUP — placed as clusters in available board space.
    5. Remaining — placed in leftover space.
    6. Rotation optimization (when *requirements* is provided).

    Args:
        constraints: Placement constraints for components.
        board_outline: Board outline polygon for bounds.
        footprint_sizes: Mapping from ref to ``(width, height)`` in mm.
        keepouts: Keepout zones to avoid.
        grid_mm: Placement grid pitch in mm.
        requirements: Optional project requirements for rotation optimization.

    Returns:
        :class:`PlacementResult` with positions, rotations, and violations.
    """
    # Compute board dimensions
    xs = [p.x for p in board_outline.polygon]
    ys = [p.y for p in board_outline.polygon]
    board_w = max(xs) - min(xs)
    board_h = max(ys) - min(ys)
    origin_x = min(xs)
    origin_y = min(ys)

    grid = _OccupancyGrid(board_w, board_h, grid_mm)
    positions: dict[str, Point] = {}
    rotations: dict[str, float] = {}
    violations: list[str] = []

    # Mark board-edge margin as occupied (prevent copper_edge_clearance DRC)
    margin_cells = max(1, int(1.0 / grid_mm))
    for mc in range(grid._cols):
        for mr in range(margin_cells):
            grid._cells[mr][mc] = True
            grid._cells[grid._rows - 1 - mr][mc] = True
    for mr2 in range(grid._rows):
        for mc2 in range(margin_cells):
            grid._cells[mr2][mc2] = True
            grid._cells[mr2][grid._cols - 1 - mc2] = True

    # Mark keepouts on grid
    for ko in keepouts:
        ko_xs = [p.x - origin_x for p in ko.polygon]
        ko_ys = [p.y - origin_y for p in ko.polygon]
        if ko_xs and ko_ys:
            grid.mark_rect(
                min(ko_xs), min(ko_ys),
                max(ko_xs) - min(ko_xs), max(ko_ys) - min(ko_ys),
            )

    # Sort constraints by priority (highest first), then by type order
    type_order = {
        PlacementConstraintType.FIXED: 0,
        PlacementConstraintType.EDGE: 1,
        PlacementConstraintType.NEAR: 2,
        PlacementConstraintType.GROUP: 3,
        PlacementConstraintType.AWAY_FROM: 4,
    }
    sorted_constraints: list[PlacementConstraint] = sorted(
        constraints,
        key=lambda c: (-c.priority, type_order.get(c.constraint_type, 5)),
    )

    # Collect all refs from constraints
    all_refs: set[str] = {c.ref for c in constraints}

    # Group constraints by ref (pick highest priority per ref)
    ref_constraint: dict[str, PlacementConstraint] = {}
    for pc in sorted_constraints:
        if pc.ref not in ref_constraint:
            ref_constraint[pc.ref] = pc

    # 1. Place FIXED
    for ref, c in ref_constraint.items():
        if c.constraint_type != PlacementConstraintType.FIXED:
            continue
        if c.x is not None and c.y is not None:
            x_rel = c.x - origin_x
            y_rel = c.y - origin_y
            w, h = footprint_sizes.get(ref, (3.0, 3.0))
            positions[ref] = Point(x=c.x, y=c.y)
            rotations[ref] = c.rotation if c.rotation is not None else 0.0
            gap = _placement_gap(w, h)
            grid.mark_rect(
                x_rel - w / 2 - gap, y_rel - h / 2 - gap,
                w + 2 * gap, h + 2 * gap,
            )
            log.debug("FIXED: %s at (%.1f, %.1f)", ref, c.x, c.y)

    # 2. Place EDGE
    edge_groups: dict[BoardEdge, list[str]] = {}
    for ref, c in ref_constraint.items():
        if c.constraint_type != PlacementConstraintType.EDGE and ref not in positions:
            continue
        if c.constraint_type == PlacementConstraintType.EDGE and ref not in positions:
            edge = c.edge or BoardEdge.LEFT
            edge_groups.setdefault(edge, []).append(ref)

    for edge, refs in edge_groups.items():
        is_horizontal = edge in (BoardEdge.TOP, BoardEdge.BOTTOM)
        edge_len = board_w if is_horizontal else board_h
        # Sort by footprint size (largest first) for better distribution
        refs_sorted = sorted(
            refs,
            key=lambda r: footprint_sizes.get(r, (3.0, 3.0))[0 if is_horizontal else 1],
            reverse=True,
        )
        spacing = edge_len / (len(refs_sorted) + 1)
        for i, ref in enumerate(refs_sorted):
            w, h = footprint_sizes.get(ref, (3.0, 3.0))
            base = origin_x if is_horizontal else origin_y
            # Component half-width along the edge axis
            half_extent = w / 2.0 if is_horizontal else h / 2.0
            offset = spacing * (i + 1) + base
            # Clamp offset so entire component body stays within the board
            offset = max(offset, half_extent + 1.0 + base)
            offset = min(offset, edge_len + base - half_extent - 1.0)
            x, y, rot = _edge_position(edge, board_w, board_h, w, h, offset)
            x += origin_x
            y += origin_y
            positions[ref] = Point(x=x, y=y)
            rotations[ref] = rot
            gap = _placement_gap(w, h)
            grid.mark_rect(
                x - origin_x - w / 2 - gap, y - origin_y - h / 2 - gap,
                w + 2 * gap, h + 2 * gap,
            )
            log.debug("EDGE(%s): %s at (%.1f, %.1f) rot=%.0f", edge.value, ref, x, y, rot)

    # 3. Place NEAR (with retry for deferred placements whose targets
    # appear later in the priority order)
    near_refs = [
        (ref, c) for ref, c in ref_constraint.items()
        if c.constraint_type == PlacementConstraintType.NEAR and ref not in positions
    ]

    # Build pin position lookup for pin-level NEAR targeting
    # Compute offsets for any component that is a NEAR target (ICs, switches, etc.)
    near_targets = {c.target_ref for _, c in near_refs if c.target_ref is not None}
    pin_offsets: dict[str, dict[str, tuple[float, float]]] = {}
    for comp in (requirements.components if requirements is not None else []):
        if comp.ref.startswith("U") or comp.ref in near_targets:
            # Estimate pin offsets from footprint dimensions
            w, h = footprint_sizes.get(comp.ref, (3.0, 3.0))
            comp_pins: dict[str, tuple[float, float]] = {}
            # Simple heuristic: distribute pins around perimeter
            pin_list = [c.pin for net in (requirements.nets if requirements is not None else [])
                        for c in net.connections if c.ref == comp.ref]
            for pidx, pin in enumerate(sorted(set(pin_list))):
                n_pins = max(1, len(set(pin_list)))
                # Single pin → center
                if n_pins == 1:
                    comp_pins[pin] = (0.0, 0.0)
                    continue
                # Left-right distribution (DIP convention)
                if pidx < n_pins // 2:
                    denom_l = max(1, n_pins // 2 - 1)
                    y_off = -h / 2.0 + h * pidx / denom_l if n_pins > 2 else 0.0
                    comp_pins[pin] = (-w / 2.0, y_off)
                else:
                    ridx = pidx - n_pins // 2
                    denom = max(1, n_pins - n_pins // 2 - 1)
                    comp_pins[pin] = (w / 2.0, h / 2.0 - h * ridx / denom if denom > 0 else 0.0)
            pin_offsets[comp.ref] = comp_pins

    def _place_near(ref: str, c: PlacementConstraint) -> bool:
        target = c.target_ref
        if target is None or target not in positions:
            return False
        target_center = positions[target]
        pin_pos = target_center
        preferred_angle: float | None = None

        # Pin-level targeting: compute pin position AND outward direction
        if c.target_pin is not None and target in pin_offsets:
            pin_off = pin_offsets[target].get(c.target_pin)
            if pin_off is not None:
                rot_rad = math.radians(rotations.get(target, 0.0))
                cos_r = math.cos(rot_rad)
                sin_r = math.sin(rot_rad)
                rpx = pin_off[0] * cos_r - pin_off[1] * sin_r
                rpy = pin_off[0] * sin_r + pin_off[1] * cos_r
                pin_pos = Point(
                    x=target_center.x + rpx,
                    y=target_center.y + rpy,
                )
                # Outward direction: from component center through pin
                if abs(rpx) > 0.01 or abs(rpy) > 0.01:
                    preferred_angle = math.atan2(rpy, rpx)

        # For edge-mounted targets (connectors), flip inward —
        # passives should be on the board-interior side
        target_constraint = ref_constraint.get(target)
        if (target_constraint is not None
                and target_constraint.constraint_type
                == PlacementConstraintType.EDGE):
            if preferred_angle is not None:
                preferred_angle += math.pi  # flip outward → inward
            else:
                # No pin info — point toward board center
                board_cx = origin_x + board_w / 2.0
                board_cy = origin_y + board_h / 2.0
                preferred_angle = math.atan2(
                    board_cy - target_center.y,
                    board_cx - target_center.x,
                )

        max_dist = c.max_distance_mm or 5.0
        enforce_min = c.min_distance_mm or 0.0
        w, h = footprint_sizes.get(ref, (3.0, 3.0))

        # Build angle search order: prefer outward from pin, then nearby
        if preferred_angle is not None:
            angles = [preferred_angle + math.radians(d)
                      for d in (0, 45, -45, 90, -90, 135, -135, 180)]
        else:
            angles = [math.radians(a) for a in range(0, 360, 45)]

        # Search from close to far, starting at the pin position
        min_dist = max(max(w, h) * 0.75, enforce_min)
        distances = [min_dist + (max_dist - min_dist) * i / 3 for i in range(4)]
        distances.extend([max_dist * 1.5, max_dist * 2.0])
        for dist in distances:
            for angle in angles:
                trial_x = pin_pos.x + dist * math.cos(angle)
                trial_y = pin_pos.y + dist * math.sin(angle)
                rx = trial_x - origin_x - w / 2
                ry = trial_y - origin_y - h / 2
                if rx >= 0 and ry >= 0 and grid.is_rect_free(rx, ry, w, h):
                    positions[ref] = Point(x=trial_x, y=trial_y)
                    # Set initial rotation: align passive toward target pin
                    if preferred_angle is not None and _is_two_pin_passive(ref):
                        angle_deg = math.degrees(preferred_angle) % 360
                        # Snap to nearest 90° — pad axis should point toward pin
                        rotations[ref] = round(angle_deg / 90.0) * 90.0 % 360.0
                    else:
                        rotations[ref] = 0.0
                    gap = _placement_gap(w, h)
                    grid.mark_rect(rx - gap, ry - gap, w + 2 * gap, h + 2 * gap)
                    log.debug("NEAR(%s.%s): %s at (%.1f, %.1f) angle=%.0f°",
                              target, c.target_pin or "?", ref, trial_x, trial_y,
                              math.degrees(angle))
                    return True
        # Fallback: nearest free spot to pin position (respecting min_distance)
        free = grid.find_nearest_free(
            pin_pos.x - origin_x, pin_pos.y - origin_y, w, h,
        )
        if free is not None:
            fx = free[0] + origin_x
            fy = free[1] + origin_y
            # Enforce min_distance from target center (avoid courtyard overlap)
            if enforce_min > 0:
                dist_to_target = math.hypot(
                    fx - target_center.x, fy - target_center.y,
                )
                if dist_to_target < enforce_min:
                    # Push outward from target center
                    angle = math.atan2(
                        fy - target_center.y, fx - target_center.x,
                    )
                    fx = target_center.x + enforce_min * math.cos(angle)
                    fy = target_center.y + enforce_min * math.sin(angle)
            positions[ref] = Point(x=fx, y=fy)
            rotations[ref] = 0.0
            gap = _placement_gap(w, h)
            gx = fx - origin_x - w / 2
            gy = fy - origin_y - h / 2
            grid.mark_rect(gx - gap, gy - gap, w + 2 * gap, h + 2 * gap)
            return True
        violations.append(f"Could not place {ref} near {target}")
        return False

    # Multiple passes to resolve dependency chains (A -> B -> C)
    for _pass in range(3):
        deferred = []
        for ref, c in near_refs:
            if ref in positions:
                continue
            if not _place_near(ref, c):
                deferred.append((ref, c))
        near_refs = deferred
        if not deferred:
            break

    # 4. Place GROUP (sorted by connectivity degree — highest first for
    # central positions and shorter average routes)
    def _connectivity_degree(ref: str) -> int:
        """Count non-power signal nets this component participates in."""
        if requirements is None:
            return 0
        count = 0
        for net in requirements.nets:
            if _is_power_net(net.name):
                continue
            if any(c.ref == ref for c in net.connections):
                count += 1
        return count

    group_members: dict[str, list[str]] = {}
    for ref, c in ref_constraint.items():
        if c.constraint_type == PlacementConstraintType.GROUP and ref not in positions:
            gname = c.group_name or "default"
            group_members.setdefault(gname, []).append(ref)

    for gname, refs in group_members.items():
        # Sort by connectivity (highest first) for better central placement
        refs.sort(key=_connectivity_degree, reverse=True)
        # Compute uniform cell size for the group
        item_w = max(footprint_sizes.get(r, (3.0, 3.0))[0] + 2.0 for r in refs)
        max_h = max(footprint_sizes.get(r, (3.0, 3.0))[1] for r in refs) + 2.0

        # Wrap to multiple rows if group exceeds board width
        max_row_w = board_w - 10.0  # 5mm margin each side
        cols_per_row = max(1, int(max_row_w / item_w))
        num_rows = math.ceil(len(refs) / cols_per_row)

        block_w = min(len(refs), cols_per_row) * item_w
        block_h = num_rows * max_h

        # Start from board centre
        centre_x = board_w / 2.0
        centre_y = board_h / 2.0
        start = grid.find_nearest_free(centre_x, centre_y, block_w, block_h)
        if start is None:
            start = (5.0, 5.0)

        base_x = start[0] + origin_x
        base_y = start[1] + origin_y
        for idx, ref in enumerate(refs):
            col = idx % cols_per_row
            row = idx // cols_per_row
            w, h = footprint_sizes.get(ref, (3.0, 3.0))
            x_pos = base_x + col * item_w + w / 2
            y_pos = base_y + row * max_h + max_h / 2
            positions[ref] = Point(x=x_pos, y=y_pos)
            rotations[ref] = 0.0
            gap = _placement_gap(w, h)
            grid.mark_rect(
                x_pos - origin_x - w / 2 - gap,
                y_pos - origin_y - max_h / 2 - gap,
                w + 2.0 + 2 * gap, max_h + 2 * gap,
            )
            log.debug(
                "GROUP(%s): %s at (%.1f, %.1f) [row=%d col=%d]",
                gname, ref, positions[ref].x, positions[ref].y, row, col,
            )

    # 5. Place any remaining unplaced refs (highest connectivity first)
    unplaced = sorted(
        [ref for ref in all_refs if ref not in positions],
        key=_connectivity_degree, reverse=True,
    )
    for ref in unplaced:
        w, h = footprint_sizes.get(ref, (3.0, 3.0))
        free = grid.find_nearest_free(board_w / 2.0, board_h / 2.0, w, h)
        if free is not None:
            positions[ref] = Point(x=free[0] + origin_x, y=free[1] + origin_y)
            rotations[ref] = 0.0
            gap = _placement_gap(w, h)
            grid.mark_rect(free[0] - gap, free[1] - gap, w + 2 * gap, h + 2 * gap)
        else:
            violations.append(f"No space for {ref} on the board")
            positions[ref] = Point(x=board_w / 2.0 + origin_x, y=board_h / 2.0 + origin_y)
            rotations[ref] = 0.0

    # 6. Optimize rotations when requirements are available
    # Two iterations: neighbours' rotations affect each other, so a second
    # pass picks up improvements missed when neighbours hadn't settled yet.
    if requirements is not None:
        for _ in range(2):
            rotations = optimize_rotations(
                positions, rotations, requirements,
                footprint_sizes=footprint_sizes,
            )

    return PlacementResult(
        positions=positions,
        rotations=rotations,
        violations=tuple(violations),
    )


# ---------------------------------------------------------------------------
# RPi HAT placement strategy
# ---------------------------------------------------------------------------


def rpi_hat_constraints(
    requirements: ProjectRequirements,
    board_template: BoardTemplate,
    footprint_sizes: dict[str, tuple[float, float]],
) -> tuple[PlacementConstraint, ...]:
    """Generate placement constraints specific to an RPi HAT design.

    Strategy:

    - GPIO header J1: FIXED at template position
    - Screw terminals J2-J5: EDGE(LEFT) or EDGE(BOTTOM)
    - ADC U1: GROUP("ADC_CORE") centred in available area
    - DIP switch SW1: NEAR(U1)
    - Bypass caps C1/C2: NEAR(U1, 3mm)
    - Voltage dividers per channel: GROUP with linear layout

    Falls back to :func:`constraints_from_requirements` for components
    not explicitly handled.

    Args:
        requirements: Project requirements document.
        board_template: The RPi HAT board template.
        footprint_sizes: Mapping from ref to ``(width, height)`` in mm.

    Returns:
        Tuple of :class:`PlacementConstraint` objects.
    """
    base = constraints_from_requirements(requirements, board_template, footprint_sizes)
    extra: list[PlacementConstraint] = []

    # Place primary IC at board centre (between J1 at top and J2-J5 at bottom)
    ic_ref: str | None = None
    for comp in requirements.components:
        if comp.ref.startswith("U"):
            ic_ref = comp.ref
            break
    if ic_ref is not None:
        cx = board_template.board_width_mm / 2.0
        cy = board_template.board_height_mm * 0.55  # slightly below centre
        extra.append(PlacementConstraint(
            ref=ic_ref,
            constraint_type=PlacementConstraintType.FIXED,
            x=cx,
            y=cy,
            rotation=0.0,
            priority=60,
        ))

    # DIP switches -> NEAR(IC) — priority must be higher than its dependents
    sw_ref: str | None = None
    for comp in requirements.components:
        if comp.ref.startswith("SW") and ic_ref is not None:
            sw_ref = comp.ref
            extra.append(PlacementConstraint(
                ref=comp.ref,
                constraint_type=PlacementConstraintType.NEAR,
                target_ref=ic_ref,
                max_distance_mm=10.0,
                priority=32,
            ))

    # Pull-up/address resistors sharing nets with SW -> NEAR(SW, pin)
    if sw_ref is not None:
        # Map net_name -> SW pin number for pin-level targeting
        sw_net_pin: dict[str, str] = {}
        for net in requirements.nets:
            for c in net.connections:
                if c.ref == sw_ref:
                    sw_net_pin[net.name] = c.pin
        for comp in requirements.components:
            if not comp.ref.startswith("R"):
                continue
            for net in requirements.nets:
                if net.name not in sw_net_pin:
                    continue
                if any(c.ref == comp.ref for c in net.connections):
                    extra.append(PlacementConstraint(
                        ref=comp.ref,
                        constraint_type=PlacementConstraintType.NEAR,
                        target_ref=sw_ref,
                        target_pin=sw_net_pin[net.name],
                        max_distance_mm=5.0,
                        priority=28,
                    ))
                    break

    # Screw terminals -> EDGE(BOTTOM) for RPi HATs (opposite GPIO header)
    for comp in requirements.components:
        if not _is_connector(comp.ref, comp.footprint):
            continue
        if _is_screw_terminal(comp.footprint):
            # Skip if already FIXED by template
            if any(
                c.ref == comp.ref and c.constraint_type == PlacementConstraintType.FIXED
                for c in base
            ):
                continue
            extra.append(PlacementConstraint(
                ref=comp.ref,
                constraint_type=PlacementConstraintType.EDGE,
                edge=BoardEdge.BOTTOM,
                priority=55,
            ))

    # Channel grouping: trace signal nets from screw terminals through
    # voltage dividers to the ADC, placing each channel's passives near
    # their input connector.
    #
    # Topology per channel:
    #   Jx --[SENSx]--> Rx_top --[AINx]--> Rx_bot, Cx_filter, U1
    #
    # We place Rx_top, Rx_bot, Cx_filter all NEAR their Jx connector.
    channel_assigned: set[str] = set()
    for net in requirements.nets:
        if _is_power_net(net.name):
            continue
        # Find SENS-type nets: connect a screw terminal (Jx) to a resistor
        conn_refs = [c.ref for c in net.connections]
        j_refs = [r for r in conn_refs if r.startswith("J") and r != "J1"]
        r_refs = [r for r in conn_refs if r.startswith("R")]
        if len(j_refs) != 1 or len(r_refs) != 1:
            continue
        anchor_j = j_refs[0]
        top_r = r_refs[0]
        # Capture the J terminal pin from this SENS net
        j_pin: str | None = None
        for conn in net.connections:
            if conn.ref == anchor_j:
                j_pin = conn.pin
                break
        # Find the ADC-side net that top_r also connects to
        channel_parts = [top_r]
        for other_net in requirements.nets:
            if other_net.name == net.name or _is_power_net(other_net.name):
                continue
            other_refs = [c.ref for c in other_net.connections]
            if top_r not in other_refs:
                continue
            # This is the AINx net — collect only passives (R, C, L)
            for r in other_refs:
                prefix = "".join(ch for ch in r if ch.isalpha()).upper()
                if prefix not in ("R", "C", "L"):
                    continue
                if r != top_r and r not in channel_parts:
                    channel_parts.append(r)
        # Place all channel parts NEAR the screw terminal
        for ref in channel_parts:
            if ref in channel_assigned:
                continue
            extra.append(PlacementConstraint(
                ref=ref,
                constraint_type=PlacementConstraintType.NEAR,
                target_ref=anchor_j,
                target_pin=j_pin,
                max_distance_mm=8.0,
                priority=35,
            ))
            channel_assigned.add(ref)

    # Merge: higher-priority extras override base
    merged: dict[str, PlacementConstraint] = {}
    for bc in base:
        if bc.ref not in merged or bc.priority > merged[bc.ref].priority:
            merged[bc.ref] = bc
    for ec in extra:
        if ec.ref not in merged or ec.priority > merged[ec.ref].priority:
            merged[ec.ref] = ec

    return tuple(merged.values())


# ---------------------------------------------------------------------------
# Component rotation optimization (FEAT-7)
# ---------------------------------------------------------------------------


def _is_two_pin_passive(ref: str) -> bool:
    """Return True if *ref* is a 2-pin passive (R, C, or L)."""
    prefix = "".join(ch for ch in ref if ch.isalpha()).upper()
    return prefix in ("R", "C", "L")


def _rotated_pad_offset(
    px: float, py: float, rotation_deg: float,
) -> tuple[float, float]:
    """Rotate a pad's relative ``(px, py)`` offset by *rotation_deg* degrees.

    Uses the standard 2-D rotation matrix.

    Args:
        px: Pad X offset from component centre.
        py: Pad Y offset from component centre.
        rotation_deg: Rotation angle in degrees (counter-clockwise).

    Returns:
        Rotated ``(x, y)`` offset.
    """
    rad = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    return (px * cos_r - py * sin_r, px * sin_r + py * cos_r)


def _build_pad_connectivity(
    requirements: ProjectRequirements,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Map ``(ref, pin_number)`` to connected ``[(neighbour_ref, pin), ...]``.

    Only signal nets are considered (power/GND excluded), matching the
    filtering in :func:`build_signal_adjacency`.

    Args:
        requirements: Project requirements with net connections.

    Returns:
        Mapping from ``(ref, pin)`` to list of connected ``(ref, pin)`` pairs.
    """
    result: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for net in requirements.nets:
        if _is_power_net(net.name):
            continue
        conns = [(c.ref, c.pin) for c in net.connections]
        for i, (r1, p1) in enumerate(conns):
            for r2, p2 in conns[i + 1:]:
                result.setdefault((r1, p1), []).append((r2, p2))
                result.setdefault((r2, p2), []).append((r1, p1))
    return result


def optimize_rotations(
    positions: dict[str, Point],
    rotations: dict[str, float],
    requirements: ProjectRequirements,
    footprint_sizes: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Optimize component rotations for shorter routing.

    When *footprint_sizes* is provided, uses pad-aware cost: for 2-pin
    passives, pad-1 is at ``(-w/2, 0)`` and pad-2 at ``(+w/2, 0)`` at 0
    degrees.  Each candidate rotation {0, 90, 180, 270} is scored by summing
    Manhattan distance from each connected pad to the neighbour's connected
    pad (or centre when pad layout is unknown).

    Without *footprint_sizes*, falls back to center-based direction snapping
    for backward compatibility.

    Args:
        positions: Placed positions for all components.
        rotations: Current rotations for all components.
        requirements: Project requirements with net connections.
        footprint_sizes: Optional mapping from ref to ``(width, height)`` mm.

    Returns:
        Updated rotation dict (original is not modified).
    """
    result = dict(rotations)

    # Build ref -> set of connected refs (via signal nets)
    adj = build_signal_adjacency(requirements)

    # --- Pad-aware path ---
    if footprint_sizes is not None:
        pad_conn = _build_pad_connectivity(requirements)

        # Build per-ref pin→pad-offset map for 2-pin passives
        # Convention: pad-1 at (-w/2, 0), pad-2 at (+w/2, 0) at rotation 0
        def _passive_pad_offsets(
            ref: str,
        ) -> dict[str, tuple[float, float]] | None:
            if not _is_two_pin_passive(ref):
                return None
            size = footprint_sizes.get(ref)
            if size is None:
                return None
            w = size[0]
            return {"1": (-w / 2.0, 0.0), "2": (w / 2.0, 0.0)}

        def _pad_world_pos(
            ref: str,
            pin: str,
            rot: float,
            pad_offsets: dict[str, tuple[float, float]] | None,
        ) -> tuple[float, float]:
            """Return world position of a pad given component rotation."""
            pos = positions[ref]
            if pad_offsets is not None and pin in pad_offsets:
                px, py = pad_offsets[pin]
                ox, oy = _rotated_pad_offset(px, py, rot)
                return (pos.x + ox, pos.y + oy)
            return (pos.x, pos.y)

        # Two-pass: passives first, then ICs
        # Pass 1: 2-pin passives
        for ref in positions:
            if not _is_two_pin_passive(ref):
                continue
            offsets = _passive_pad_offsets(ref)
            if offsets is None:
                continue
            # Collect this component's pin-level connections
            pin_conns: list[tuple[str, str, str]] = []  # (my_pin, nb_ref, nb_pin)
            for pin in ("1", "2"):
                for nb_ref, nb_pin in pad_conn.get((ref, pin), []):
                    if nb_ref in positions:
                        pin_conns.append((pin, nb_ref, nb_pin))
            if not pin_conns:
                continue

            best_rot = result.get(ref, 0.0)
            best_cost = float("inf")
            for trial_rot in (0.0, 90.0, 180.0, 270.0):
                cost = 0.0
                for my_pin, nb_ref, nb_pin in pin_conns:
                    mx, my = _pad_world_pos(ref, my_pin, trial_rot, offsets)
                    nb_offsets = _passive_pad_offsets(nb_ref)
                    nb_rot = result.get(nb_ref, 0.0)
                    nx, ny = _pad_world_pos(nb_ref, nb_pin, nb_rot, nb_offsets)
                    cost += abs(mx - nx) + abs(my - ny)
                if cost < best_cost:
                    best_cost = cost
                    best_rot = trial_rot
            result[ref] = best_rot

        # Pass 2: ICs (U*)
        for ref in positions:
            if not ref.startswith("U"):
                continue
            size = footprint_sizes.get(ref)
            if size is None:
                continue
            # Collect IC pin connections
            ic_pin_conns: list[tuple[str, str, str]] = []
            for comp in requirements.components:
                if comp.ref != ref:
                    continue
                for comp_pin in comp.pins:
                    for nb_ref, nb_pin in pad_conn.get((ref, comp_pin.number), []):
                        if nb_ref in positions:
                            ic_pin_conns.append((comp_pin.number, nb_ref, nb_pin))
                break
            if not ic_pin_conns:
                continue

            # For ICs we don't have detailed pad layout — use centre-based
            # with pad offsets on the *neighbour* side for differentiation
            pos = positions[ref]
            best_rot = result.get(ref, 0.0)
            best_cost = float("inf")
            for trial_rot in (0.0, 90.0, 180.0, 270.0):
                cost = 0.0
                for _my_pin, nb_ref, nb_pin in ic_pin_conns:
                    nb_offsets = _passive_pad_offsets(nb_ref)
                    nb_rot = result.get(nb_ref, 0.0)
                    nx, ny = _pad_world_pos(nb_ref, nb_pin, nb_rot, nb_offsets)
                    cost += abs(pos.x - nx) + abs(pos.y - ny)
                if cost < best_cost:
                    best_cost = cost
                    best_rot = trial_rot
            result[ref] = best_rot

        return result

    # --- Fallback: center-based (no footprint_sizes) ---

    # 2-pin passives: align toward nearest connected neighbour
    for ref in positions:
        if not _is_two_pin_passive(ref):
            continue
        neighbours = adj.get(ref, set())
        if not neighbours:
            continue
        pos = positions[ref]
        best_neighbour: str | None = None
        best_dist = float("inf")
        for nb in neighbours:
            if nb not in positions:
                continue
            nb_pos = positions[nb]
            d = math.hypot(nb_pos.x - pos.x, nb_pos.y - pos.y)
            if d < best_dist:
                best_dist = d
                best_neighbour = nb

        if best_neighbour is not None:
            nb_pos = positions[best_neighbour]
            dx = nb_pos.x - pos.x
            dy = nb_pos.y - pos.y
            angle = math.degrees(math.atan2(dy, dx))
            snapped = round(angle / 90.0) * 90.0
            result[ref] = snapped % 360.0

    # ICs: center-based is rotation-invariant, keep current rotation
    return result


# ---------------------------------------------------------------------------
# Courtyard collision checking (FEAT-9)
# ---------------------------------------------------------------------------


def check_courtyard_collisions(
    positions: dict[str, Point],
    footprint_sizes: dict[str, tuple[float, float]],
    keepouts: tuple[Keepout, ...] = (),
) -> tuple[str, ...]:
    """Check for courtyard overlaps between placed components.

    Uses axis-aligned bounding boxes centered on each component position.
    Also checks for overlaps with keepout zones.

    Args:
        positions: Placed positions for all components.
        footprint_sizes: Mapping from ref to ``(width, height)`` in mm.
        keepouts: Optional keepout zones to check against.

    Returns:
        Tuple of violation description strings.
    """
    violations: list[str] = []
    refs = list(positions.keys())

    # Component vs component collision
    for i, ref_a in enumerate(refs):
        pos_a = positions[ref_a]
        w_a, h_a = footprint_sizes.get(ref_a, (3.0, 3.0))
        ax0 = pos_a.x - w_a / 2.0
        ay0 = pos_a.y - h_a / 2.0
        ax1 = pos_a.x + w_a / 2.0
        ay1 = pos_a.y + h_a / 2.0

        for ref_b in refs[i + 1:]:
            pos_b = positions[ref_b]
            w_b, h_b = footprint_sizes.get(ref_b, (3.0, 3.0))
            bx0 = pos_b.x - w_b / 2.0
            by0 = pos_b.y - h_b / 2.0
            bx1 = pos_b.x + w_b / 2.0
            by1 = pos_b.y + h_b / 2.0

            if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
                violations.append(
                    f"Courtyard collision: {ref_a} and {ref_b}"
                )

    # Component vs keepout collision
    for ref in refs:
        pos = positions[ref]
        w, h = footprint_sizes.get(ref, (3.0, 3.0))
        cx0 = pos.x - w / 2.0
        cy0 = pos.y - h / 2.0
        cx1 = pos.x + w / 2.0
        cy1 = pos.y + h / 2.0

        for ko_idx, ko in enumerate(keepouts):
            ko_xs = [p.x for p in ko.polygon]
            ko_ys = [p.y for p in ko.polygon]
            if not ko_xs:
                continue
            kx0 = min(ko_xs)
            ky0 = min(ko_ys)
            kx1 = max(ko_xs)
            ky1 = max(ko_ys)

            if cx0 < kx1 and cx1 > kx0 and cy0 < ky1 and cy1 > ky0:
                violations.append(
                    f"Courtyard collision: {ref} overlaps keepout zone {ko_idx}"
                )

    return tuple(violations)

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
        """Check whether a rectangular area is entirely unoccupied."""
        c0 = max(0, self._to_grid(x))
        r0 = max(0, self._to_grid(y))
        c1 = min(self._cols, self._to_grid(x + w))
        r1 = min(self._rows, self._to_grid(y + h))
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
        # Alternate screw terminals between LEFT and BOTTOM
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

    # 4. Decoupling caps -> NEAR their associated IC
    for comp in requirements.components:
        if _is_decoupling_cap(comp.ref, comp.value):
            # Find IC sharing a power net
            for net in requirements.nets:
                cap_in_net = any(c.ref == comp.ref for c in net.connections)
                if not cap_in_net:
                    continue
                for conn in net.connections:
                    if conn.ref in ic_refs and conn.ref != comp.ref:
                        constraints.append(PlacementConstraint(
                            ref=comp.ref,
                            constraint_type=PlacementConstraintType.NEAR,
                            target_ref=conn.ref,
                            max_distance_mm=3.0,
                            priority=30,
                        ))
                        break
                else:
                    continue
                break

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
) -> PlacementResult:
    """Solve component placement using a greedy constraint-based algorithm.

    Placement order:

    1. FIXED — placed at exact positions (highest priority).
    2. EDGE — placed along board edges with spacing and rotation.
    3. NEAR — placed adjacent to already-placed target components.
    4. GROUP — placed as clusters in available board space.
    5. Remaining — placed in leftover space.

    Args:
        constraints: Placement constraints for components.
        board_outline: Board outline polygon for bounds.
        footprint_sizes: Mapping from ref to ``(width, height)`` in mm.
        keepouts: Keepout zones to avoid.
        grid_mm: Placement grid pitch in mm.

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
    sorted_constraints = sorted(
        constraints,
        key=lambda c: (-c.priority, type_order.get(c.constraint_type, 5)),
    )

    # Collect all refs from constraints
    all_refs = {c.ref for c in constraints}

    # Group constraints by ref (pick highest priority per ref)
    ref_constraint: dict[str, PlacementConstraint] = {}
    for c in sorted_constraints:
        if c.ref not in ref_constraint:
            ref_constraint[c.ref] = c

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
            gap = PLACEMENT_GAP_MM
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
        edge_len = board_w if edge in (BoardEdge.TOP, BoardEdge.BOTTOM) else board_h
        spacing = edge_len / (len(refs) + 1)
        for i, ref in enumerate(refs):
            w, h = footprint_sizes.get(ref, (3.0, 3.0))
            base = origin_x if edge in (BoardEdge.TOP, BoardEdge.BOTTOM) else origin_y
            offset = spacing * (i + 1) + base
            x, y, rot = _edge_position(edge, board_w, board_h, w, h, offset)
            x += origin_x
            y += origin_y
            positions[ref] = Point(x=x, y=y)
            rotations[ref] = rot
            gap = PLACEMENT_GAP_MM
            grid.mark_rect(
                x - origin_x - w / 2 - gap, y - origin_y - h / 2 - gap,
                w + 2 * gap, h + 2 * gap,
            )
            log.debug("EDGE(%s): %s at (%.1f, %.1f) rot=%.0f", edge.value, ref, x, y, rot)

    # 3. Place NEAR
    for ref, c in ref_constraint.items():
        if c.constraint_type != PlacementConstraintType.NEAR or ref in positions:
            continue
        target = c.target_ref
        if target is None or target not in positions:
            # Target not placed yet — defer
            continue
        target_pos = positions[target]
        max_dist = c.max_distance_mm or 5.0
        w, h = footprint_sizes.get(ref, (3.0, 3.0))
        # Try positions around the target
        placed = False
        for angle_deg in range(0, 360, 45):
            angle = math.radians(angle_deg)
            trial_x = target_pos.x + max_dist * math.cos(angle)
            trial_y = target_pos.y + max_dist * math.sin(angle)
            rx = trial_x - origin_x - w / 2
            ry = trial_y - origin_y - h / 2
            if rx >= 0 and ry >= 0 and grid.is_rect_free(rx, ry, w, h):
                positions[ref] = Point(x=trial_x, y=trial_y)
                rotations[ref] = 0.0
                gap = PLACEMENT_GAP_MM
                grid.mark_rect(rx - gap, ry - gap, w + 2 * gap, h + 2 * gap)
                log.debug("NEAR(%s): %s at (%.1f, %.1f)", target, ref, trial_x, trial_y)
                placed = True
                break
        if not placed:
            # Find nearest free spot to target
            free = grid.find_nearest_free(
                target_pos.x - origin_x, target_pos.y - origin_y, w, h,
            )
            if free is not None:
                positions[ref] = Point(x=free[0] + origin_x, y=free[1] + origin_y)
                rotations[ref] = 0.0
                gap = PLACEMENT_GAP_MM
                grid.mark_rect(free[0] - gap, free[1] - gap, w + 2 * gap, h + 2 * gap)
            else:
                violations.append(f"Could not place {ref} near {target}")

    # 4. Place GROUP
    group_members: dict[str, list[str]] = {}
    for ref, c in ref_constraint.items():
        if c.constraint_type == PlacementConstraintType.GROUP and ref not in positions:
            gname = c.group_name or "default"
            group_members.setdefault(gname, []).append(ref)

    for gname, refs in group_members.items():
        # Find a free area large enough for the group
        total_w = sum(footprint_sizes.get(r, (3.0, 3.0))[0] + 2.0 for r in refs)
        max_h = max(footprint_sizes.get(r, (3.0, 3.0))[1] for r in refs) + 2.0

        # Start from board centre
        centre_x = board_w / 2.0
        centre_y = board_h / 2.0
        start = grid.find_nearest_free(centre_x, centre_y, total_w, max_h)
        if start is None:
            start = (5.0, 5.0)

        x_cursor = start[0] + origin_x
        y_cursor = start[1] + origin_y
        for ref in refs:
            w, h = footprint_sizes.get(ref, (3.0, 3.0))
            positions[ref] = Point(x=x_cursor + w / 2, y=y_cursor + max_h / 2)
            rotations[ref] = 0.0
            gap = PLACEMENT_GAP_MM
            grid.mark_rect(
                x_cursor - origin_x - gap, y_cursor - origin_y - gap,
                w + 2.0 + 2 * gap, max_h + 2 * gap,
            )
            x_cursor += w + 2.0
            log.debug(
                "GROUP(%s): %s at (%.1f, %.1f)",
                gname, ref, positions[ref].x, positions[ref].y,
            )

    # 5. Place any remaining unplaced refs
    unplaced = [ref for ref in all_refs if ref not in positions]
    for ref in unplaced:
        w, h = footprint_sizes.get(ref, (3.0, 3.0))
        free = grid.find_nearest_free(board_w / 2.0, board_h / 2.0, w, h)
        if free is not None:
            positions[ref] = Point(x=free[0] + origin_x, y=free[1] + origin_y)
            rotations[ref] = 0.0
            gap = PLACEMENT_GAP_MM
            grid.mark_rect(free[0] - gap, free[1] - gap, w + 2 * gap, h + 2 * gap)
        else:
            violations.append(f"No space for {ref} on the board")
            positions[ref] = Point(x=board_w / 2.0 + origin_x, y=board_h / 2.0 + origin_y)
            rotations[ref] = 0.0

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

    # Find the primary IC (first U* ref)
    ic_ref: str | None = None
    for comp in requirements.components:
        if comp.ref.startswith("U"):
            ic_ref = comp.ref
            break

    # DIP switches -> NEAR(IC)
    for comp in requirements.components:
        if comp.ref.startswith("SW") and ic_ref is not None:
            extra.append(PlacementConstraint(
                ref=comp.ref,
                constraint_type=PlacementConstraintType.NEAR,
                target_ref=ic_ref,
                max_distance_mm=10.0,
                priority=25,
            ))

    # Screw terminals -> EDGE(LEFT or BOTTOM), alternating
    screw_idx = 0
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
            edge = BoardEdge.LEFT if screw_idx % 2 == 0 else BoardEdge.BOTTOM
            extra.append(PlacementConstraint(
                ref=comp.ref,
                constraint_type=PlacementConstraintType.EDGE,
                edge=edge,
                priority=55,
            ))
            screw_idx += 1

    # Voltage divider chains: pairs of R* sharing a signal net -> GROUP
    adj = build_signal_adjacency(requirements)
    divider_idx = 0
    visited_divider: set[str] = set()
    for ref, neighbours in adj.items():
        if not ref.startswith("R") or ref in visited_divider:
            continue
        # Check for R-R pairs on same signal net
        r_neighbours = [n for n in neighbours if n.startswith("R")]
        if r_neighbours:
            chain = [ref, *r_neighbours]
            group_name = f"_divider_{divider_idx}"
            for r in chain:
                if r not in visited_divider:
                    extra.append(PlacementConstraint(
                        ref=r,
                        constraint_type=PlacementConstraintType.GROUP,
                        group_name=group_name,
                        priority=20,
                    ))
                    visited_divider.add(r)
            divider_idx += 1

    # Merge: higher-priority extras override base
    merged: dict[str, PlacementConstraint] = {}
    for c in base:
        if c.ref not in merged or c.priority > merged[c.ref].priority:
            merged[c.ref] = c
    for c in extra:
        if c.ref not in merged or c.priority > merged[c.ref].priority:
            merged[c.ref] = c

    return tuple(merged.values())


# ---------------------------------------------------------------------------
# Component rotation optimization (FEAT-7)
# ---------------------------------------------------------------------------


def _is_two_pin_passive(ref: str) -> bool:
    """Return True if *ref* is a 2-pin passive (R, C, or L)."""
    prefix = "".join(ch for ch in ref if ch.isalpha()).upper()
    return prefix in ("R", "C", "L")


def optimize_rotations(
    positions: dict[str, Point],
    rotations: dict[str, float],
    requirements: ProjectRequirements,
) -> dict[str, float]:
    """Optimize component rotations for shorter routing.

    For 2-pin passives (R, C, L), aligns the pad axis with the direction
    toward the nearest connected component.  For ICs (U*), tries 0/90/180/270
    and picks the rotation minimizing total Manhattan distance to connected pads.

    Args:
        positions: Placed positions for all components.
        rotations: Current rotations for all components.
        requirements: Project requirements with net connections.

    Returns:
        Updated rotation dict (original is not modified).
    """
    result = dict(rotations)

    # Build ref -> set of connected refs (via signal nets)
    adj = build_signal_adjacency(requirements)

    # 2-pin passives: align toward nearest connected neighbour
    for ref in positions:
        if not _is_two_pin_passive(ref):
            continue
        neighbours = adj.get(ref, set())
        if not neighbours:
            continue
        # Find nearest neighbour by Euclidean distance
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
            # Snap to nearest 90 degrees
            snapped = round(angle / 90.0) * 90.0
            result[ref] = snapped % 360.0

    # ICs: try 0/90/180/270 and pick minimum total Manhattan distance
    for ref in positions:
        if not ref.startswith("U"):
            continue
        neighbours = adj.get(ref, set())
        placed_neighbours = [nb for nb in neighbours if nb in positions]
        if not placed_neighbours:
            continue
        pos = positions[ref]
        best_rot = result.get(ref, 0.0)
        best_cost = float("inf")
        for trial_rot in (0.0, 90.0, 180.0, 270.0):
            cost = 0.0
            for nb in placed_neighbours:
                nb_pos = positions[nb]
                cost += abs(nb_pos.x - pos.x) + abs(nb_pos.y - pos.y)
            if cost < best_cost:
                best_cost = cost
                best_rot = trial_rot
        result[ref] = best_rot

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

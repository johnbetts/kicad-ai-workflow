"""Shared ratsnest utilities for placement visualization.

Provides net-to-pad mapping, minimum spanning tree computation, and
power-net filtering used by both matplotlib-based rendering
(``placement_render.py``) and kicad-cli SVG overlay (``kicad_export.py``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign

# Power net names excluded from ratsnest visualization
POWER_NETS: frozenset[str] = frozenset({
    "GND", "VIN", "VCC", "+3V3", "+5V", "+12V", "+24V",
    "+3.3V", "+1.8V", "VBUS", "VBAT",
})


def rotate_point(
    px: float, py: float, angle_deg: float,
) -> tuple[float, float]:
    """Rotate a point around the origin by *angle_deg* degrees (KiCad CW convention)."""
    # KiCad stores rotation as CW in screen view (Y-down).
    # Negate angle for the standard CCW rotation matrix.
    rad = math.radians(-angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return px * cos_a - py * sin_a, px * sin_a + py * cos_a


def build_net_pad_map(
    pcb: PCBDesign,
) -> dict[str, list[tuple[float, float]]]:
    """Build a mapping of net_name to list of absolute pad (x, y) positions.

    Only includes signal nets (skips power/ground nets and unnamed nets).

    Args:
        pcb: The PCB design to extract pad positions from.

    Returns:
        Dict mapping net name to list of absolute (x, y) tuples.
    """
    net_pads: dict[str, list[tuple[float, float]]] = {}
    for fp in pcb.footprints:
        fx, fy, frot = fp.position.x, fp.position.y, fp.rotation
        for pad in fp.pads:
            net = pad.net_name
            if not net or net.upper() in POWER_NETS:
                continue
            rx, ry = rotate_point(pad.position.x, pad.position.y, frot)
            abs_x, abs_y = fx + rx, fy + ry
            net_pads.setdefault(net, []).append((abs_x, abs_y))
    return net_pads


def minimum_spanning_tree(
    points: list[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Compute MST edges for a set of 2-D points (Prim's algorithm).

    Returns list of (index_a, index_b) pairs forming the tree.

    Args:
        points: List of (x, y) coordinates.

    Returns:
        List of edge pairs as index tuples.
    """
    n = len(points)
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]

    in_tree = [False] * n
    min_cost = [float("inf")] * n
    min_edge: list[int] = [-1] * n
    edges: list[tuple[int, int]] = []

    in_tree[0] = True
    for j in range(1, n):
        dx = points[0][0] - points[j][0]
        dy = points[0][1] - points[j][1]
        min_cost[j] = dx * dx + dy * dy
        min_edge[j] = 0

    for _ in range(n - 1):
        best = -1
        best_cost = float("inf")
        for j in range(n):
            if not in_tree[j] and min_cost[j] < best_cost:
                best_cost = min_cost[j]
                best = j
        if best < 0:
            break
        in_tree[best] = True
        edges.append((min_edge[best], best))
        for j in range(n):
            if not in_tree[j]:
                dx = points[best][0] - points[j][0]
                dy = points[best][1] - points[j][1]
                dist2 = dx * dx + dy * dy
                if dist2 < min_cost[j]:
                    min_cost[j] = dist2
                    min_edge[j] = best

    return edges

"""Render PCB component placement to PNG for visual review.

Produces color-coded placement diagrams showing:
- Board outline with component bounding boxes
- FeatureBlock group coloring (or voltage domain fallback)
- Dotted bounding box boundaries around groups
- Pad-to-pad ratsnest (thin lines between connected pads on signal nets)
- Quality score overlay

Requires matplotlib (optional dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.visualization.ratsnest import (
    build_net_pad_map as _build_net_pad_map_shared,
)
from kicad_pipeline.visualization.ratsnest import (
    minimum_spanning_tree as _minimum_spanning_tree_shared,
)
from kicad_pipeline.visualization.ratsnest import (
    rotate_point as _rotate_point_shared,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint, PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.functional_grouper import VoltageDomain
    from kicad_pipeline.optimization.scoring import QualityScore

logger = logging.getLogger(__name__)

# Voltage domain -> display color (keys match VoltageDomain.value)
_DOMAIN_COLORS: dict[str, str] = {
    "24v": "#ff4444",      # VIN_24V - red
    "5v": "#ff8800",       # POWER_5V - orange
    "3v3": "#44aaff",      # DIGITAL_3V3 - blue
    "analog": "#44ff44",   # ANALOG - green
    "mixed": "#cccccc",    # MIXED - gray
}

# Display labels for legend
_DOMAIN_LABELS: dict[str, str] = {
    "24v": "24V Power",
    "5v": "5V Power",
    "3v3": "3.3V Digital",
    "analog": "Analog",
    "mixed": "Mixed/Unclassified",
}

# FeatureBlock group color map — uses keyword matching, not exact names
_GROUP_COLOR_MAP: dict[str, str] = {
    "power": "#ff8800",
    "relay": "#ff4444",
    "mcu": "#44cc44",
    "analog": "#ffcc00",
    "ethernet": "#4488ff",
    "display": "#cc44ff",
}

# Backward-compatible alias
GROUP_COLORS = _GROUP_COLOR_MAP

# Fallback palette for groups not in the named map
_FALLBACK_GROUP_PALETTE: tuple[str, ...] = (
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
)

# Power net names — canonical source is ratsnest.py (imported at top of file)


def _fp_size(fp: Footprint) -> tuple[float, float]:
    """Footprint size for rendering — matches optimizer collision detection.

    Delegates to :func:`~kicad_pipeline.pcb.footprints.estimate_courtyard_mm`
    so renderer bounding boxes agree with the optimizer's collision model.
    """
    from kicad_pipeline.pcb.footprints import estimate_courtyard_mm

    return estimate_courtyard_mm(fp)


def _get_group_color(group_name: str, idx: int) -> str:
    """Get color for a group name using keyword matching, falling back to palette."""
    lower = group_name.lower()
    for key, color in _GROUP_COLOR_MAP.items():
        if key in lower:
            return color
    return _FALLBACK_GROUP_PALETTE[idx % len(_FALLBACK_GROUP_PALETTE)]


def _rotate_point(
    px: float, py: float, angle_deg: float,
) -> tuple[float, float]:
    """Rotate a point around the origin by *angle_deg* degrees."""
    return _rotate_point_shared(px, py, angle_deg)


def _build_net_pad_map(
    pcb: PCBDesign,
) -> dict[str, list[tuple[float, float]]]:
    """Build a mapping of net_name to list of absolute pad (x, y) positions."""
    return _build_net_pad_map_shared(pcb)


def _minimum_spanning_tree(
    points: list[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Compute MST edges for a set of 2-D points (Prim's algorithm)."""
    return _minimum_spanning_tree_shared(points)


def _draw_pad_ratsnest(
    ax: object,
    pcb: PCBDesign,
) -> None:
    """Draw pad-to-pad ratsnest lines for signal nets.

    Uses absolute pad positions (rotation-aware) and MST to avoid
    long crossing lines.  Gives visual clues about which pads need
    to connect, making rotation/orientation issues immediately visible.
    """
    net_pads = _build_net_pad_map(pcb)

    for _net_name, pads in net_pads.items():
        if len(pads) < 2:
            continue
        edges = _minimum_spanning_tree(pads)
        for i, j in edges:
            ax.plot(  # type: ignore[attr-defined]
                [pads[i][0], pads[j][0]],
                [pads[i][1], pads[j][1]],
                "-",
                color="#4466aa",
                alpha=0.35,
                linewidth=0.5,
            )


def render_placement(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    output_path: str | Path,
    *,
    title: str | None = None,
    score: QualityScore | None = None,
    domain_map: dict[str, VoltageDomain] | None = None,
    group_map: dict[str, str] | None = None,
    show_ratsnest: bool = True,
    figsize: tuple[float, float] = (18, 12),
    dpi: int = 200,
) -> Path:
    """Render PCB placement to a PNG file.

    Args:
        pcb: The PCBDesign to render.
        requirements: Project requirements (for net/component info).
        output_path: Where to save the PNG.
        title: Plot title (auto-generated if None).
        score: Optional QualityScore to overlay.
        domain_map: Optional ref->VoltageDomain map for coloring.
            Used when group_map is None.
        group_map: Optional ref->group_name map for group-based coloring.
            When provided, components are colored by FeatureBlock group
            instead of voltage domain, and group boundaries are drawn.
        show_ratsnest: Draw signal net connections.
        figsize: Figure size in inches.
        dpi: Output resolution.

    Returns:
        Path to the saved image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Rectangle
    except ImportError as exc:
        msg = "matplotlib is required for placement rendering: pip install matplotlib"
        raise ImportError(msg) from exc

    # Determine coloring mode
    use_groups = group_map is not None and len(group_map) > 0

    # Build color mapping
    if use_groups:
        assert group_map is not None  # for type narrowing
        # Assign color per group
        unique_groups = sorted(set(group_map.values()))
        group_color_map: dict[str, str] = {
            name: _get_group_color(name, i)
            for i, name in enumerate(unique_groups)
        }
        ref_color: dict[str, str] = {
            ref: group_color_map.get(gname, "#cccccc")
            for ref, gname in group_map.items()
        }
    else:
        # Fallback to voltage domain coloring
        if domain_map is None:
            from kicad_pipeline.optimization.functional_grouper import (
                classify_voltage_domains,
            )
            domain_map = classify_voltage_domains(requirements)
        ref_domain: dict[str, str] = {}
        for comp in requirements.components:
            d = domain_map.get(comp.ref)
            ref_domain[comp.ref] = d.value if d else "MIXED"
        ref_color = {
            ref: _DOMAIN_COLORS.get(dval, "#cccccc")
            for ref, dval in ref_domain.items()
        }

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Board outline
    outline = pcb.outline
    if outline and outline.polygon:
        xs = [p.x for p in outline.polygon]
        ys = [p.y for p in outline.polygon]
        ax.plot(xs, ys, "k-", linewidth=2)
        ax.fill(xs, ys, alpha=0.05, color="green")

    # Draw footprints — center boxes on pad centroid, not KiCad origin
    from kicad_pipeline.pcb.pin_map import origin_to_centroid

    for fp in pcb.footprints:
        w, h = _fp_size(fp)
        color = ref_color.get(fp.ref, "#cccccc")
        x, y = origin_to_centroid(fp, fp.position.x, fp.position.y, fp.rotation)

        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", alpha=0.6, linewidth=0.8,
        )
        ax.add_patch(rect)
        fontsize = 5 if len(fp.ref) <= 3 else 4
        ax.text(x, y, fp.ref, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="black")

    # Draw group bounding boxes with dotted lines
    if use_groups:
        assert group_map is not None
        ref_pos = {
            fp.ref: origin_to_centroid(
                fp, fp.position.x, fp.position.y, fp.rotation,
            )
            for fp in pcb.footprints
        }
        unique_groups = sorted(set(group_map.values()))
        group_color_map = {
            name: _get_group_color(name, i)
            for i, name in enumerate(unique_groups)
        }

        for gname in unique_groups:
            grefs = [r for r, g in group_map.items() if g == gname and r in ref_pos]
            if len(grefs) < 2:
                continue

            gxs = [ref_pos[r][0] for r in grefs]
            gys = [ref_pos[r][1] for r in grefs]
            margin = 2.0
            gx_min = min(gxs) - margin
            gy_min = min(gys) - margin
            gx_max = max(gxs) + margin
            gy_max = max(gys) + margin

            color = group_color_map.get(gname, "#888888")
            boundary = Rectangle(
                (gx_min, gy_min),
                gx_max - gx_min,
                gy_max - gy_min,
                fill=False,
                edgecolor=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.add_patch(boundary)

            # Group name label above boundary
            ax.text(
                (gx_min + gx_max) / 2, gy_min - 0.5,
                gname, ha="center", va="bottom",
                fontsize=6, fontweight="bold", color=color, alpha=0.8,
            )

    # Pad-to-pad ratsnest — draw lines between pads that share a signal net
    if show_ratsnest:
        _draw_pad_ratsnest(ax, pcb)

    # Legend
    if use_groups:
        assert group_map is not None
        unique_groups = sorted(set(group_map.values()))
        group_color_map = {
            name: _get_group_color(name, i)
            for i, name in enumerate(unique_groups)
        }
        legend_patches = [
            mpatches.Patch(color=group_color_map[g], alpha=0.6, label=g)
            for g in unique_groups
        ]
    else:
        used_domains = set(ref_color.values())
        legend_patches = [
            mpatches.Patch(color=c, alpha=0.6, label=_DOMAIN_LABELS.get(n, n))
            for n, c in _DOMAIN_COLORS.items()
            if c in used_domains or n in {
                ref_domain.get(comp.ref, "MIXED")
                for comp in requirements.components
            }
        ]
        # Re-build properly using domain labels
        if domain_map is not None:
            used_domain_vals = {
                domain_map.get(comp.ref)
                for comp in requirements.components
                if domain_map.get(comp.ref) is not None
            }
            used_names = {d.value for d in used_domain_vals if d is not None}
            legend_patches = [
                mpatches.Patch(
                    color=_DOMAIN_COLORS.get(n, "#cccccc"),
                    alpha=0.6,
                    label=_DOMAIN_LABELS.get(n, n),
                )
                for n in _DOMAIN_COLORS
                if n in used_names
            ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper left", fontsize=7)

    # Score overlay
    if score is not None:
        ax.text(
            0.99, 0.01,
            f"Score: {score.overall_score:.3f} ({score.grade})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()  # KiCad Y-down
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    if title is None:
        name = requirements.project.name or "PCB"
        title = f"{name} -- Placement ({len(pcb.footprints)} components)"
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    out = Path(output_path)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Placement render saved: %s", out)
    return out


def render_zones(
    zones: list[object],
    board_bounds: tuple[float, float, float, float],
    output_path: str | Path,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 9),
    dpi: int = 150,
) -> Path:
    """Render board zone partitioning to PNG.

    Shows colored rectangular zones with labels. Used for Level 1
    visual verification of the zone partitioner.

    Args:
        zones: List of BoardZone instances.
        board_bounds: (min_x, min_y, max_x, max_y).
        output_path: Where to save the PNG.
        title: Optional plot title.
        figsize: Figure size in inches.
        dpi: Output resolution.

    Returns:
        Path to the saved image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        msg = "matplotlib is required for placement rendering: pip install matplotlib"
        raise ImportError(msg) from exc

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    bx1, by1, bx2, by2 = board_bounds

    # Board outline
    board_rect = Rectangle(
        (bx1, by1), bx2 - bx1, by2 - by1,
        fill=True, facecolor="#f0f0f0", edgecolor="black", linewidth=2,
    )
    ax.add_patch(board_rect)

    # Draw zones
    zone_colors = list(_GROUP_COLOR_MAP.values()) + list(_FALLBACK_GROUP_PALETTE)
    for i, zone in enumerate(zones):
        zx1, zy1, zx2, zy2 = zone.rect  # type: ignore[attr-defined]
        color = zone_colors[i % len(zone_colors)]

        # Match zone name to color
        for key, c in _GROUP_COLOR_MAP.items():
            if key in zone.name.lower():  # type: ignore[attr-defined]
                color = c
                break

        rect = Rectangle(
            (zx1, zy1), zx2 - zx1, zy2 - zy1,
            fill=True, facecolor=color, edgecolor="black",
            alpha=0.3, linewidth=1.5,
        )
        ax.add_patch(rect)

        # Zone label
        ax.text(
            (zx1 + zx2) / 2, (zy1 + zy2) / 2,
            f"{zone.name}\n({', '.join(zone.groups)})",  # type: ignore[attr-defined]
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color="black",
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title or "Board Zone Partitioning (Level 1)")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    out = Path(output_path)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Zone render saved: %s", out)
    return out

"""Render PCB component placement to PNG for visual review.

Produces color-coded placement diagrams showing:
- Board outline with component bounding boxes
- Voltage domain coloring (VIN_24V=red, BUCK_12V=orange, etc.)
- Signal net ratsnest (thin lines between connected components)
- Subcircuit grouping annotations
- Quality score overlay

Requires matplotlib (optional dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint, PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.functional_grouper import VoltageDomain
    from kicad_pipeline.optimization.scoring import QualityScore

logger = logging.getLogger(__name__)

# Voltage domain → display color
_DOMAIN_COLORS: dict[str, str] = {
    "VIN_24V": "#ff4444",
    "BUCK_12V": "#ff8800",
    "LDO_5V": "#ffcc00",
    "DIGITAL_3V3": "#44aaff",
    "ANALOG": "#44ff44",
    "MIXED": "#cccccc",
}

# Power net names to exclude from ratsnest
_POWER_NETS = frozenset({
    "GND", "VIN", "VCC", "+3V3", "+5V", "+12V", "+24V",
    "+3.3V", "+1.8V", "VBUS", "VBAT",
})


def _fp_size(fp: Footprint) -> tuple[float, float]:
    """Estimate footprint width/height from pad span."""
    if not fp.pads:
        return 2.0, 2.0
    xs = [p.position.x for p in fp.pads]
    ys = [p.position.y for p in fp.pads]
    w = max(xs) - min(xs) + 1.5
    h = max(ys) - min(ys) + 1.5
    return max(w, 1.5), max(h, 1.5)


def render_placement(
    pcb: PCBDesign,
    requirements: ProjectRequirements,
    output_path: str | Path,
    *,
    title: str | None = None,
    score: QualityScore | None = None,
    domain_map: dict[str, VoltageDomain] | None = None,
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
        domain_map: Optional ref→VoltageDomain map for coloring.
            If None, classify_voltage_domains is called.
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
        from matplotlib.patches import FancyBboxPatch
    except ImportError as exc:
        msg = "matplotlib is required for placement rendering: pip install matplotlib"
        raise ImportError(msg) from exc

    # Build domain map if not provided
    if domain_map is None:
        from kicad_pipeline.optimization.functional_grouper import classify_voltage_domains
        domain_map = classify_voltage_domains(requirements)

    # Map ref → domain name string
    ref_domain: dict[str, str] = {}
    for comp in requirements.components:
        d = domain_map.get(comp.ref)
        ref_domain[comp.ref] = d.value if d else "MIXED"

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Board outline
    outline = pcb.outline
    if outline and outline.polygon:
        xs = [p.x for p in outline.polygon]
        ys = [p.y for p in outline.polygon]
        ax.plot(xs, ys, "k-", linewidth=2)
        ax.fill(xs, ys, alpha=0.05, color="green")

    # Draw footprints
    for fp in pcb.footprints:
        x, y = fp.position.x, fp.position.y
        w, h = _fp_size(fp)
        domain = ref_domain.get(fp.ref, "MIXED")
        color = _DOMAIN_COLORS.get(domain, "#cccccc")

        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", alpha=0.6, linewidth=0.8,
        )
        ax.add_patch(rect)
        fontsize = 5 if len(fp.ref) <= 3 else 4
        ax.text(x, y, fp.ref, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="black")

    # Signal ratsnest
    if show_ratsnest:
        ref_pos = {fp.ref: (fp.position.x, fp.position.y) for fp in pcb.footprints}
        for net in requirements.nets:
            if net.name.upper() in _POWER_NETS:
                continue
            conns = [c.ref for c in net.connections if c.ref in ref_pos]
            if len(conns) >= 2:
                for i in range(len(conns) - 1):
                    p1, p2 = ref_pos[conns[i]], ref_pos[conns[i + 1]]
                    ax.plot(
                        [p1[0], p2[0]], [p1[1], p2[1]],
                        "-", color="#888888", alpha=0.2, linewidth=0.3,
                    )

    # Legend
    used_domains = set(ref_domain.values())
    legend_patches = [
        mpatches.Patch(color=c, alpha=0.6, label=n)
        for n, c in _DOMAIN_COLORS.items()
        if n in used_domains
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
        title = f"{name} — Placement ({len(pcb.footprints)} components)"
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()

    out = Path(output_path)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Placement render saved: %s", out)
    return out

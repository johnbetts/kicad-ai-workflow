"""Excellon drill file generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign


def _mm_to_drill(mm: float) -> str:
    """Format mm value with 4 decimal places for Excellon."""
    return f"{mm:.4f}"


def generate_drill_file(pcb: PCBDesign, pth_only: bool = True) -> str:
    """Generate Excellon 2 drill file for PCB.

    Args:
        pcb: The PCB design to generate drill files for.
        pth_only: If True, generate PTH file; if False, generate NPTH file.
    """
    lines: list[str] = []
    lines.append("M48")
    lines.append("METRIC,TZ")
    if pth_only:
        lines.append("; Plated through holes")
    else:
        lines.append("; Non-plated through holes")

    pad_type_filter = "thru_hole" if pth_only else "np_thru_hole"

    # Collect drill hits: (drill_diameter, x, y)
    hits: list[tuple[float, float, float]] = []
    for fp in pcb.footprints:
        for pad in fp.pads:
            if pad.pad_type == pad_type_filter and pad.drill_diameter is not None:
                cx = fp.position.x + pad.position.x
                cy = fp.position.y + pad.position.y
                hits.append((pad.drill_diameter, cx, cy))

    # Vias are plated through holes
    if pth_only:
        for via in pcb.vias:
            hits.append((via.drill, via.position.x, via.position.y))

    # Build tool table: collect unique drill sizes
    sizes_seen: list[float] = []
    for drill, _x, _y in hits:
        if drill not in sizes_seen:
            sizes_seen.append(drill)
    sizes_seen.sort()

    tool_map: dict[float, int] = {}
    for i, size in enumerate(sizes_seen, start=1):
        tool_num = i
        tool_map[size] = tool_num
        lines.append(f"T{tool_num}C{size:.4f}")

    lines.append("%")
    lines.append("G05")

    # Group hits by tool
    tool_hits: dict[int, list[tuple[float, float]]] = {
        t: [] for t in tool_map.values()
    }
    for drill, x, y in hits:
        tool_hits[tool_map[drill]].append((x, y))

    for tool_num in sorted(tool_hits):
        if tool_hits[tool_num]:
            lines.append(f"T{tool_num}")
            for x, y in tool_hits[tool_num]:
                lines.append(f"X{x * 10000:.0f}Y{y * 10000:.0f}")

    lines.append("M30")
    return "\n".join(lines) + "\n"


def generate_drill_files(pcb: PCBDesign) -> dict[str, str]:
    """Generate PTH and NPTH drill files.

    Returns:
        Dict with keys "project-PTH.drl" and "project-NPTH.drl".
    """
    return {
        "project-PTH.drl": generate_drill_file(pcb, pth_only=True),
        "project-NPTH.drl": generate_drill_file(pcb, pth_only=False),
    }

"""RS-274X Gerber file generator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Pad, PCBDesign, Track

GERBER_VERSION = "RS-274X"
COORD_FORMAT = (4, 6)  # 4 integer digits, 6 decimal digits
COORD_SCALE = 1_000_000  # 1mm = 1,000,000 units

_LAYER_FILE_FUNCTIONS: dict[str, str] = {
    "F.Cu": "Copper,L1,Top",
    "In1.Cu": "Copper,L2,Inr",
    "In2.Cu": "Copper,L3,Inr",
    "B.Cu": "Copper,L2,Bot",
    "F.Silkscreen": "Legend,Top",
    "B.Silkscreen": "Legend,Bot",
    "F.Mask": "SolderMask,Top",
    "B.Mask": "SolderMask,Bot",
    "F.Paste": "SolderPaste,Top",
    "B.Paste": "SolderPaste,Bot",
    "Edge.Cuts": "Profile,NP",
}


def _mm_to_gerber(mm: float) -> str:
    """Convert mm to integer Gerber coordinate string."""
    return str(round(mm * COORD_SCALE))


def _make_gerber_header(layer_name: str, project_name: str = "project") -> list[str]:
    """Return list of Gerber header lines for the given layer."""
    file_function = _LAYER_FILE_FUNCTIONS.get(layer_name, "Other,User")
    return [
        "%FSLAX46Y46*%",
        "%MOMM*%",
        f"%LN{layer_name}*%",
        "%TF.GenerationSoftware,kicad-ai-pipeline,1.0*%",
        f"%TF.FileFunction,{file_function}*%",
        "%TF.SameCoordinates,Original*%",
        "%ADD10C,0.100000*%",
    ]


def _make_gerber_footer() -> list[str]:
    """Return Gerber footer lines."""
    return ["M02*"]


def _render_pad_flash(pad: Pad, fp_x: float, fp_y: float) -> list[str]:
    """Render a pad flash command for an SMD pad."""
    cx = fp_x + pad.position.x
    cy = fp_y + pad.position.y
    return [
        f"%ADD11R,{pad.size_x:.6f}X{pad.size_y:.6f}*%",
        "D11*",
        f"X{_mm_to_gerber(cx)}Y{_mm_to_gerber(cy)}D03*",
    ]


def _render_track(track: Track) -> list[str]:
    """Render a track segment as a Gerber line."""
    return [
        "G01*",
        f"X{_mm_to_gerber(track.start.x)}Y{_mm_to_gerber(track.start.y)}D02*",
        f"X{_mm_to_gerber(track.end.x)}Y{_mm_to_gerber(track.end.y)}D01*",
    ]


def generate_copper_layer(pcb: PCBDesign, layer: str) -> str:
    """Generate a complete RS-274X Gerber string for one copper layer."""
    lines: list[str] = []
    lines.extend(_make_gerber_header(layer))
    lines.append("G01*")

    for fp in pcb.footprints:
        for pad in fp.pads:
            if pad.pad_type == "smd" and layer in pad.layers:
                lines.extend(_render_pad_flash(pad, fp.position.x, fp.position.y))

    for track in pcb.tracks:
        if track.layer == layer:
            lines.extend(_render_track(track))

    lines.extend(_make_gerber_footer())
    return "\n".join(lines) + "\n"


def generate_edge_cuts(pcb: PCBDesign) -> str:
    """Generate Edge.Cuts Gerber from board outline polygon."""
    lines: list[str] = []
    lines.extend(_make_gerber_header("Edge.Cuts"))
    lines.append("G01*")

    pts = pcb.outline.polygon
    if pts:
        lines.append(
            f"X{_mm_to_gerber(pts[0].x)}Y{_mm_to_gerber(pts[0].y)}D02*"
        )
        for pt in pts[1:]:
            lines.append(
                f"X{_mm_to_gerber(pt.x)}Y{_mm_to_gerber(pt.y)}D01*"
            )
        # Close the polygon
        lines.append(
            f"X{_mm_to_gerber(pts[0].x)}Y{_mm_to_gerber(pts[0].y)}D01*"
        )

    lines.extend(_make_gerber_footer())
    return "\n".join(lines) + "\n"


def generate_all_gerbers(
    pcb: PCBDesign, project_name: str = "project"
) -> dict[str, str]:
    """Generate Gerbers for all standard layers. Returns dict: {filename: content}."""
    layers: list[tuple[str, str]] = [
        (f"{project_name}-F_Cu.gbr", "F.Cu"),
        (f"{project_name}-B_Cu.gbr", "B.Cu"),
        (f"{project_name}-F_SilkS.gbr", "F.Silkscreen"),
        (f"{project_name}-B_SilkS.gbr", "B.Silkscreen"),
        (f"{project_name}-F_Mask.gbr", "F.Mask"),
        (f"{project_name}-B_Mask.gbr", "B.Mask"),
    ]
    result: dict[str, str] = {}
    for filename, layer in layers:
        result[filename] = generate_copper_layer(pcb, layer)
    result[f"{project_name}-Edge_Cuts.gbr"] = generate_edge_cuts(pcb)
    return result


def write_gerbers(gerbers: dict[str, str], output_dir: str | Path) -> None:
    """Write all Gerber strings to files in output_dir. Create dir if needed."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, content in gerbers.items():
        (out / filename).write_text(content, encoding="utf-8")

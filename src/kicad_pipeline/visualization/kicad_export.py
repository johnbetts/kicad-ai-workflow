"""High-fidelity PCB image export via ``kicad-cli``.

Generates publication-quality PNG images of PCB layouts by using KiCad's
own renderer — showing actual pad geometry, silkscreen, courtyards, and
board outline exactly as they appear in the PCB editor.

Two export strategies are attempted in order:

1. **SVG export** (``kicad-cli pcb export svg``) → convert to PNG
   - Shows pads, silkscreen text, courtyards, copper — the "editor view"
   - Requires an SVG→PNG converter (``cairosvg``, ``rsvg-convert``, or
     ImageMagick ``convert``)

2. **3-D render** (``kicad-cli pcb render``) — fallback
   - Produces a photorealistic top-down PNG directly
   - Less useful for placement review (no net labels, less contrast)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.cli.kicad_cli import find_kicad_cli

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign

logger = logging.getLogger(__name__)

_DEFAULT_LAYERS = "F.Cu,F.SilkS,F.CrtYd,Edge.Cuts"
_DEFAULT_WIDTH = 2400


def export_pcb_image(
    pcb_path: Path,
    output_path: Path,
    layers: str = _DEFAULT_LAYERS,
    width: int = _DEFAULT_WIDTH,
    pcb: PCBDesign | None = None,
) -> Path:
    """Export a high-fidelity PNG image of a KiCad PCB file.

    Attempts SVG export first (best for placement review), then falls
    back to a 3-D render if no SVG→PNG converter is available.

    When *pcb* is provided, ratsnest lines are injected into the SVG
    before conversion to PNG, showing signal-net connectivity.

    Args:
        pcb_path: Path to the ``.kicad_pcb`` file.
        output_path: Desired output PNG path.
        layers: Comma-separated layer list for SVG export.
        width: Image width in pixels (used by both strategies).
        pcb: Optional PCBDesign for ratsnest overlay on SVG export.

    Returns:
        The resolved *output_path* on success.

    Raises:
        KiCadPipelineError: If ``kicad-cli`` is not found.
        RuntimeError: If both export strategies fail.
    """
    pcb_path = Path(pcb_path)
    output_path = Path(output_path)
    if not pcb_path.is_file():
        msg = f"PCB file not found: {pcb_path}"
        raise FileNotFoundError(msg)

    kicad_cli = find_kicad_cli()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Strategy 1: SVG export → PNG conversion --------------------------
    svg_path = _export_svg(kicad_cli, pcb_path, layers)
    if svg_path is not None:
        if pcb is not None:
            _inject_ratsnest_into_svg(svg_path, pcb)
        png_path = _convert_svg_to_png(svg_path, output_path, width)
        if png_path is not None:
            logger.info("Hi-fi export (SVG→PNG) written to %s", png_path)
            return png_path
        logger.warning("SVG exported but no PNG converter available, trying 3-D render")

    # --- Strategy 2: 3-D render (direct PNG) ------------------------------
    rendered = _render_3d(kicad_cli, pcb_path, output_path, width)
    if rendered is not None:
        logger.info("Hi-fi export (3-D render) written to %s", rendered)
        return rendered

    msg = (
        "Both SVG export and 3-D render failed. Ensure kicad-cli supports "
        "'pcb export svg' or 'pcb render', and that an SVG converter "
        "(cairosvg, rsvg-convert, or ImageMagick) is installed."
    )
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def _export_svg(
    kicad_cli: str,
    pcb_path: Path,
    layers: str,
) -> Path | None:
    """Run ``kicad-cli pcb export svg`` and return the SVG path, or *None*."""
    with tempfile.TemporaryDirectory(prefix="kicad_svg_") as tmpdir:
        svg_out = Path(tmpdir) / f"{pcb_path.stem}.svg"
        cmd = [
            kicad_cli,
            "pcb",
            "export",
            "svg",
            "--mode-single",
            "-l",
            layers,
            "--exclude-drawing-sheet",
            "--fit-page-to-board",
            "-o",
            str(svg_out),
            str(pcb_path),
        ]
        logger.debug("SVG export command: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("SVG export failed: %s", exc)
            return None

        if result.returncode != 0:
            logger.warning(
                "SVG export exited %d: %s", result.returncode, result.stderr[:300],
            )
            return None

        if not svg_out.is_file():
            logger.warning("SVG export did not produce output file")
            return None

        # Copy to a persistent location (tmpdir will be cleaned up).
        fd, tmp_path = tempfile.mkstemp(suffix=".svg", prefix="kicad_hifi_")
        os.close(fd)
        persistent = Path(tmp_path)
        shutil.copy2(svg_out, persistent)
        return persistent


# ---------------------------------------------------------------------------
# SVG → PNG conversion (try multiple backends)
# ---------------------------------------------------------------------------

def _convert_svg_to_png(
    svg_path: Path,
    output_path: Path,
    width: int,
) -> Path | None:
    """Convert an SVG file to PNG using the first available backend.

    Backends tried in order:
    1. ``cairosvg`` Python library
    2. ``rsvg-convert`` CLI
    3. ImageMagick ``convert`` CLI
    """
    png = _convert_cairosvg(svg_path, output_path, width)
    if png is not None:
        _cleanup_temp_svg(svg_path)
        return png

    png = _convert_rsvg(svg_path, output_path, width)
    if png is not None:
        _cleanup_temp_svg(svg_path)
        return png

    png = _convert_imagemagick(svg_path, output_path, width)
    if png is not None:
        _cleanup_temp_svg(svg_path)
        return png

    _cleanup_temp_svg(svg_path)
    return None


def _convert_cairosvg(
    svg_path: Path, output_path: Path, width: int,
) -> Path | None:
    """Try converting with the ``cairosvg`` Python library."""
    try:
        import cairosvg  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("cairosvg not installed")
        return None

    try:
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(output_path),
            output_width=width,
        )
        if output_path.is_file() and output_path.stat().st_size > 0:
            return output_path
    except Exception as exc:
        logger.warning("cairosvg conversion failed: %s", exc)
    return None


def _convert_rsvg(
    svg_path: Path, output_path: Path, width: int,
) -> Path | None:
    """Try converting with ``rsvg-convert`` (librsvg)."""
    rsvg = shutil.which("rsvg-convert")
    if rsvg is None:
        logger.debug("rsvg-convert not found on PATH")
        return None

    cmd = [rsvg, "-w", str(width), "-o", str(output_path), str(svg_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and output_path.is_file():
            return output_path
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("rsvg-convert failed: %s", exc)
    return None


def _convert_imagemagick(
    svg_path: Path, output_path: Path, width: int,
) -> Path | None:
    """Try converting with ImageMagick ``convert`` (or ``magick``)."""
    for exe in ("magick", "convert"):
        binary = shutil.which(exe)
        if binary is None:
            continue
        # Build command — magick uses sub-command syntax on v7+.
        if exe == "magick":
            cmd = [binary, "convert", "-resize", f"{width}x", str(svg_path), str(output_path)]
        else:
            cmd = [binary, "-resize", f"{width}x", str(svg_path), str(output_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and output_path.is_file():
                return output_path
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("%s conversion failed: %s", exe, exc)

    logger.debug("ImageMagick not found on PATH")
    return None


# ---------------------------------------------------------------------------
# 3-D render fallback
# ---------------------------------------------------------------------------

def _render_3d(
    kicad_cli: str,
    pcb_path: Path,
    output_path: Path,
    width: int,
) -> Path | None:
    """Fallback: use ``kicad-cli pcb render`` for a 3-D top-down PNG."""
    cmd = [
        kicad_cli,
        "pcb",
        "render",
        "--side",
        "top",
        "-w",
        str(width),
        "-o",
        str(output_path),
        str(pcb_path),
    ]
    logger.debug("3-D render command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("3-D render failed: %s", exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "3-D render exited %d: %s", result.returncode, result.stderr[:300],
        )
        return None

    if output_path.is_file() and output_path.stat().st_size > 0:
        return output_path
    return None


# ---------------------------------------------------------------------------
# Ratsnest SVG injection
# ---------------------------------------------------------------------------

_RATSNEST_COLOR = "#4466aa"
_RATSNEST_OPACITY = "0.45"
_RATSNEST_STROKE_WIDTH = "0.3"

# KiCad SVG namespace
_SVG_NS = "http://www.w3.org/2000/svg"


def _inject_ratsnest_into_svg(svg_path: Path, pcb: PCBDesign) -> None:
    """Inject ratsnest lines into a KiCad-exported SVG.

    Parses the SVG, extracts the viewBox for coordinate mapping, adds
    ``<line>`` elements for each MST edge from the net-pad map, and
    writes the modified SVG back.

    Args:
        svg_path: Path to the SVG file to modify in place.
        pcb: The PCBDesign with footprint/pad data for ratsnest.
    """
    from kicad_pipeline.visualization.ratsnest import (
        build_net_pad_map,
        minimum_spanning_tree,
    )

    net_pads = build_net_pad_map(pcb)
    if not net_pads:
        return

    # Parse SVG
    ET.register_namespace("", _SVG_NS)
    try:
        tree = ET.parse(str(svg_path))
    except ET.ParseError:
        logger.warning("Failed to parse SVG for ratsnest injection")
        return

    root = tree.getroot()

    # Extract viewBox for coordinate mapping
    viewbox = root.get("viewBox")
    if not viewbox:
        logger.debug("SVG has no viewBox — skipping ratsnest injection")
        return

    parts = viewbox.split()
    if len(parts) != 4:
        logger.debug("Unexpected viewBox format: %s", viewbox)
        return

    # KiCad SVG viewBox is in mm, matching PCB coordinates.
    # Pad positions are in board mm — they map directly to SVG viewBox coords.

    # Build ratsnest group
    ratsnest_group = ET.SubElement(root, f"{{{_SVG_NS}}}g")
    ratsnest_group.set("id", "ratsnest")
    ratsnest_group.set("opacity", _RATSNEST_OPACITY)

    line_count = 0
    for _net_name, pads in net_pads.items():
        if len(pads) < 2:
            continue
        edges = minimum_spanning_tree(pads)
        for i, j in edges:
            line = ET.SubElement(ratsnest_group, f"{{{_SVG_NS}}}line")
            line.set("x1", f"{pads[i][0]:.4f}")
            line.set("y1", f"{pads[i][1]:.4f}")
            line.set("x2", f"{pads[j][0]:.4f}")
            line.set("y2", f"{pads[j][1]:.4f}")
            line.set("stroke", _RATSNEST_COLOR)
            line.set("stroke-width", _RATSNEST_STROKE_WIDTH)
            line_count += 1

    if line_count > 0:
        tree.write(str(svg_path), xml_declaration=True, encoding="unicode")
        logger.info("Injected %d ratsnest lines into SVG", line_count)


def _cleanup_temp_svg(svg_path: Path) -> None:
    """Remove a temporary SVG file, ignoring errors."""
    import contextlib

    with contextlib.suppress(OSError):
        svg_path.unlink(missing_ok=True)

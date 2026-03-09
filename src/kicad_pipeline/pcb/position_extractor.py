"""Extract footprint positions from existing KiCad PCB files or IPC connections.

Enables layout preservation during regeneration: footprint positions from an
existing board are extracted and injected as fixed positions into the placement
solver, so manual placement/routing work is not lost.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.sexp.parser import parse_file

if TYPE_CHECKING:
    from kicad_pipeline.sexp.writer import SExpNode

log = logging.getLogger(__name__)


def _extract_ref(fp_node: SExpNode) -> str | None:
    """Extract reference designator from a footprint node.

    Handles both KiCad 9 ``(property "Reference" "R1" ...)`` and legacy
    ``(fp_text reference "R1" ...)`` formats.
    """
    if not isinstance(fp_node, list):
        return None
    for child in fp_node:
        if not isinstance(child, list) or len(child) < 3:
            continue
        tag = child[0]
        if tag == "property" and child[1] == "Reference":
            return str(child[2])
        if tag == "fp_text" and child[1] == "reference":
            return str(child[2])
    return None


def _extract_position(fp_node: SExpNode) -> tuple[float, float, float] | None:
    """Extract (x, y, rotation) from a footprint's ``(at ...)`` node."""
    if not isinstance(fp_node, list):
        return None
    for child in fp_node:
        if not isinstance(child, list) or len(child) < 3:
            continue
        if child[0] == "at":
            x = float(str(child[1]))
            y = float(str(child[2]))
            rotation = float(str(child[3])) if len(child) > 3 else 0.0
            return (x, y, rotation)
    return None


def _extract_ref_text_position(
    fp_node: SExpNode,
) -> tuple[float, float, float] | None:
    """Extract reference text (x, y, rotation) from a footprint node.

    Searches for ``(property "Reference" ... (at x y angle))`` or legacy
    ``(fp_text reference ... (at x y angle))``.

    Returns:
        ``(x, y, rotation)`` relative to the footprint origin, or ``None``.
    """
    if not isinstance(fp_node, list):
        return None
    for child in fp_node:
        if not isinstance(child, list) or len(child) < 3:
            continue
        tag = child[0]
        is_ref_property = tag == "property" and child[1] == "Reference"
        is_ref_fp_text = tag == "fp_text" and child[1] == "reference"
        if is_ref_property or is_ref_fp_text:
            # Find the (at x y angle) within this property/fp_text node
            for sub in child:
                if isinstance(sub, list) and len(sub) >= 3 and sub[0] == "at":
                    rx = float(str(sub[1]))
                    ry = float(str(sub[2]))
                    rrot = float(str(sub[3])) if len(sub) > 3 else 0.0
                    return (rx, ry, rrot)
    return None


def positions_from_pcb_file(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """Parse a ``.kicad_pcb`` file and extract ref -> (x_mm, y_mm, rotation_deg).

    Args:
        path: Path to the ``.kicad_pcb`` file.

    Returns:
        Mapping of reference designator to position tuple.
    """
    tree = parse_file(path)
    result: dict[str, tuple[float, float, float]] = {}
    if not isinstance(tree, list):
        return result
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "footprint":
            continue
        ref = _extract_ref(node)
        pos = _extract_position(node)
        if ref is not None and pos is not None:
            result[ref] = pos
    return result


def ref_text_positions_from_pcb_file(
    path: str | Path,
) -> dict[str, tuple[float, float, float]]:
    """Parse a ``.kicad_pcb`` file and extract ref text positions.

    Args:
        path: Path to the ``.kicad_pcb`` file.

    Returns:
        Mapping of reference designator to text ``(x, y, rotation)``
        relative to the footprint origin.
    """
    tree = parse_file(path)
    result: dict[str, tuple[float, float, float]] = {}
    if not isinstance(tree, list):
        return result
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "footprint":
            continue
        ref = _extract_ref(node)
        text_pos = _extract_ref_text_position(node)
        if ref is not None and text_pos is not None:
            result[ref] = text_pos
    return result


def positions_from_source(
    source: str | Path | object,
) -> dict[str, tuple[float, float, float]]:
    """Unified position extraction from file path or IPC connection.

    Args:
        source: Either a file path (str/Path) to a ``.kicad_pcb`` file, or a
            :class:`~kicad_pipeline.ipc.connection.KiCadConnection` instance.

    Returns:
        Mapping of reference designator to ``(x_mm, y_mm, rotation_deg)``.
    """
    if isinstance(source, str | Path):
        path = Path(source)
        if path.is_file():
            log.info("Extracting positions from PCB file: %s", path)
            return positions_from_pcb_file(path)
        log.warning("PCB file not found, no positions to preserve: %s", path)
        return {}

    # Assume IPC connection object
    from kicad_pipeline.ipc.board_ops import pull_footprint_positions

    log.info("Pulling positions from live KiCad IPC connection")
    return pull_footprint_positions(source)  # type: ignore[arg-type]


def ref_text_positions_from_source(
    source: str | Path | object,
) -> dict[str, tuple[float, float, float]]:
    """Unified ref text position extraction from file or IPC.

    Args:
        source: File path to ``.kicad_pcb`` or IPC connection.

    Returns:
        Mapping of ref to ``(text_x, text_y, text_rotation)`` relative
        to footprint origin.
    """
    if isinstance(source, str | Path):
        path = Path(source)
        if path.is_file():
            return ref_text_positions_from_pcb_file(path)
        return {}

    # IPC path — extract from live board
    from kicad_pipeline.ipc.board_ops import pull_ref_text_positions

    return pull_ref_text_positions(source)  # type: ignore[arg-type]

"""Type mapping between kicad-python board objects and pipeline models.

All coordinate conversion passes through ``kipy.units`` helpers
(mm floats <-> nanometre ints used by KiCad's IPC layer).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import IPCSyncError, IPCUnavailableError

if TYPE_CHECKING:

    from kicad_pipeline.ipc.connection import KiCadConnection

log = logging.getLogger(__name__)


def _require_kipy() -> None:
    from kicad_pipeline.ipc import _HAS_KIPY

    if not _HAS_KIPY:
        raise IPCUnavailableError("kicad-python is not installed")


def footprint_positions_from_board(
    conn: KiCadConnection,
) -> dict[str, tuple[float, float, float]]:
    """Extract footprint positions from the live KiCad board.

    Returns:
        Mapping of reference designator to ``(x_mm, y_mm, rotation_deg)``.

    Raises:
        IPCSyncError: If reading the board fails.
    """
    _require_kipy()

    try:
        board = conn.client.get_board()
        positions: dict[str, tuple[float, float, float]] = {}
        for fp in board.get_footprints():
            ref = fp.reference.value
            pos = fp.position
            x_mm = _nm_to_mm(pos.x)
            y_mm = _nm_to_mm(pos.y)
            rot = pos.rotation.value if pos.HasField("rotation") else 0.0
            positions[ref] = (x_mm, y_mm, rot)
        return positions
    except Exception as exc:
        raise IPCSyncError(f"Failed to read footprint positions: {exc}") from exc


def tracks_from_board(
    conn: KiCadConnection,
) -> tuple[tuple[float, float, float, float, float, str, int], ...]:
    """Extract tracks from the live board.

    Returns:
        Tuple of ``(start_x, start_y, end_x, end_y, width, layer, net)`` in mm.
    """
    _require_kipy()

    try:
        board = conn.client.get_board()
        result: list[tuple[float, float, float, float, float, str, int]] = []
        for trk in board.get_tracks():
            result.append((
                _nm_to_mm(trk.start.x),
                _nm_to_mm(trk.start.y),
                _nm_to_mm(trk.end.x),
                _nm_to_mm(trk.end.y),
                _nm_to_mm(trk.width),
                trk.layer.name if hasattr(trk.layer, "name") else str(trk.layer),
                trk.net,
            ))
        return tuple(result)
    except Exception as exc:
        raise IPCSyncError(f"Failed to read tracks: {exc}") from exc


def nets_from_board(
    conn: KiCadConnection,
) -> tuple[tuple[int, str], ...]:
    """Extract net list from the live board.

    Returns:
        Tuple of ``(net_number, net_name)``.
    """
    _require_kipy()

    try:
        board = conn.client.get_board()
        result: list[tuple[int, str]] = []
        for net in board.get_nets():
            result.append((net.number, net.name))
        return tuple(result)
    except Exception as exc:
        raise IPCSyncError(f"Failed to read nets: {exc}") from exc


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def _nm_to_mm(nm: int) -> float:
    """Convert nanometres to millimetres."""
    return nm / 1_000_000.0


def _mm_to_nm(mm: float) -> int:
    """Convert millimetres to nanometres."""
    return int(mm * 1_000_000)

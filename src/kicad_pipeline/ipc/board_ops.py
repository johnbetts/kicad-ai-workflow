"""High-level board operations via KiCad IPC.

These functions compose :mod:`connection` and :mod:`converter` into
user-facing actions: push a PCB file, refill zones, and pull state back.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import IPCSyncError, IPCUnavailableError
from kicad_pipeline.models.pcb import Footprint, PCBDesign, Point

if TYPE_CHECKING:
    from kicad_pipeline.ipc.connection import KiCadConnection

log = logging.getLogger(__name__)


def _require_kipy() -> None:
    from kicad_pipeline.ipc import _HAS_KIPY

    if not _HAS_KIPY:
        raise IPCUnavailableError("kicad-python is not installed")


def push_pcb_to_kicad(pcb_path: str | Path, conn: KiCadConnection) -> None:
    """Signal KiCad to reload the PCB file from disk.

    Uses ``board.revert()`` so KiCad re-reads the file without requiring
    the user to close/reopen.

    Args:
        pcb_path: Path to the ``.kicad_pcb`` file on disk.
        conn: Active IPC connection.

    Raises:
        IPCSyncError: If the revert command fails.
    """
    _require_kipy()

    path = Path(pcb_path).resolve()
    log.info("Pushing PCB to KiCad: %s", path)

    try:
        board = conn.client.get_board()
        board.revert()
        log.info("KiCad board reverted (reloaded from disk)")
    except Exception as exc:
        raise IPCSyncError(f"Failed to push PCB to KiCad: {exc}") from exc


def refill_zones(conn: KiCadConnection) -> None:
    """Fill all copper zones on the active board via IPC.

    This replaces the ``pcbnew`` subprocess approach, using the live
    IPC connection to trigger zone fill in the running KiCad instance.

    Args:
        conn: Active IPC connection.

    Raises:
        IPCSyncError: If the zone fill command fails.
    """
    _require_kipy()

    log.info("Refilling zones via IPC...")
    try:
        board = conn.client.get_board()
        board.refill_zones()
        log.info("Zone fill complete via IPC")
    except Exception as exc:
        raise IPCSyncError(f"Failed to refill zones via IPC: {exc}") from exc


def pull_footprint_positions(
    conn: KiCadConnection,
) -> dict[str, tuple[float, float, float]]:
    """Read footprint positions from the live KiCad board.

    Returns:
        Mapping of reference designator to ``(x_mm, y_mm, rotation_deg)``.

    Raises:
        IPCSyncError: If reading fails.
    """
    from kicad_pipeline.ipc.converter import footprint_positions_from_board

    return footprint_positions_from_board(conn)


def pull_board_snapshot(
    conn: KiCadConnection,
    design: PCBDesign,
) -> PCBDesign:
    """Read the live board state and merge updated positions into *design*.

    This enables the "user routes in KiCad, pipeline reads back" workflow.
    Currently merges footprint positions; future phases will also pull
    tracks and vias.

    Args:
        conn: Active IPC connection.
        design: The pipeline's current PCB design to update.

    Returns:
        A new :class:`PCBDesign` with updated footprint positions.

    Raises:
        IPCSyncError: If reading the board state fails.
    """
    from kicad_pipeline.ipc.converter import footprint_positions_from_board

    positions = footprint_positions_from_board(conn)

    updated_fps: list[Footprint] = []
    for fp in design.footprints:
        if fp.ref in positions:
            x, y, rot = positions[fp.ref]
            updated_fps.append(
                replace(fp, position=Point(x=x, y=y), rotation=rot)
            )
        else:
            updated_fps.append(fp)

    return replace(design, footprints=tuple(updated_fps))

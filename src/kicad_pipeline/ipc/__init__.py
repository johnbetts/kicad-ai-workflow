"""KiCad 9 IPC API integration.

Provides real-time communication with a running KiCad 9 instance via the
``kicad-python`` client library.  All public functions degrade gracefully
when the optional dependency is missing — callers get
:class:`~kicad_pipeline.exceptions.IPCUnavailableError` rather than
an ``ImportError``.

The ``_HAS_KIPY`` sentinel is ``True`` only when ``kipy`` can be imported.
"""

from __future__ import annotations

try:
    import kipy as _kipy  # noqa: F401

    _HAS_KIPY: bool = True
except ImportError:  # pragma: no cover — optional dep
    _HAS_KIPY = False

from kicad_pipeline.ipc.board_ops import (
    pull_board_snapshot,
    pull_footprint_positions,
    push_pcb_to_kicad,
    refill_zones,
)
from kicad_pipeline.ipc.connection import (
    IPCConnectionInfo,
    KiCadConnection,
    connect,
    is_available,
)

__all__ = [
    "_HAS_KIPY",
    "IPCConnectionInfo",
    "KiCadConnection",
    "connect",
    "is_available",
    "pull_board_snapshot",
    "pull_footprint_positions",
    "push_pcb_to_kicad",
    "refill_zones",
]

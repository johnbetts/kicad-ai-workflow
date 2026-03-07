"""KiCad IPC connection lifecycle.

Manages connecting to / disconnecting from a running KiCad 9 instance
over its Unix-socket IPC channel (protobuf transport).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import IPCConnectionError, IPCUnavailableError

if TYPE_CHECKING:
    from types import TracebackType

    from kipy import KiCad

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS: int = 2000


def _require_kipy() -> None:
    """Raise :class:`IPCUnavailableError` when ``kipy`` is not installed."""
    from kicad_pipeline.ipc import _HAS_KIPY

    if not _HAS_KIPY:
        raise IPCUnavailableError(
            "kicad-python is not installed.  Install it with: "
            "pip install 'kicad-ai-pipeline[ipc]'"
        )


@dataclass(frozen=True)
class IPCConnectionInfo:
    """Snapshot of IPC connection metadata.

    Attributes:
        socket_path: Unix socket path used for the connection.
        kicad_version: KiCad version string reported by the remote.
        connected: Whether the connection is alive.
    """

    socket_path: str
    kicad_version: str
    connected: bool


class KiCadConnection:
    """Context-managed wrapper around a ``kipy.KiCad`` client.

    Usage::

        with connect() as conn:
            refill_zones(conn)
    """

    def __init__(self, client: KiCad, socket_path: str) -> None:
        self._client = client
        self._socket_path = socket_path
        self._connected = True

    # -- public properties ---------------------------------------------------

    @property
    def client(self) -> KiCad:
        """Return the underlying ``kipy.KiCad`` client."""
        return self._client

    @property
    def info(self) -> IPCConnectionInfo:
        """Return connection metadata."""
        version = ""
        with contextlib.suppress(Exception):
            version = self._client.get_version()
        return IPCConnectionInfo(
            socket_path=self._socket_path,
            kicad_version=version,
            connected=self._connected,
        )

    # -- context-manager protocol --------------------------------------------

    def __enter__(self) -> KiCadConnection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying IPC connection."""
        if self._connected:
            try:
                self._client.close()
            except Exception:
                log.debug("Ignoring error during KiCad IPC close", exc_info=True)
            finally:
                self._connected = False


def connect(
    socket_path: str | None = None,
    *,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> KiCadConnection:
    """Open an IPC connection to a running KiCad 9 instance.

    Args:
        socket_path: Unix socket path.  ``None`` uses the default KiCad
            socket (auto-discovered by ``kipy``).
        timeout_ms: Connection timeout in milliseconds.

    Returns:
        A :class:`KiCadConnection` that can be used as a context manager.

    Raises:
        IPCUnavailableError: ``kicad-python`` is not installed.
        IPCConnectionError: The socket connection failed (KiCad not running,
            wrong path, timeout, etc.).
    """
    _require_kipy()

    import kipy

    try:
        if socket_path is not None:
            client = kipy.KiCad(address=socket_path, timeout=timeout_ms)
        else:
            client = kipy.KiCad(timeout=timeout_ms)

        # Ping to verify the connection is alive.
        client.get_version()
    except Exception as exc:
        raise IPCConnectionError(
            f"Failed to connect to KiCad IPC"
            f"{f' at {socket_path}' if socket_path else ''}: {exc}"
        ) from exc

    resolved = socket_path or "<default>"
    log.info("Connected to KiCad IPC (%s)", resolved)
    return KiCadConnection(client, resolved)


def is_available() -> bool:
    """Check whether KiCad IPC is reachable *right now*.

    Returns ``True`` only when ``kipy`` is installed **and** a running KiCad
    instance responds to a version ping.  Never raises.
    """
    try:
        with connect() as conn:
            _ = conn.info
        return True
    except Exception:
        return False

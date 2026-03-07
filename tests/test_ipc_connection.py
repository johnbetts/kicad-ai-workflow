"""Tests for kicad_pipeline.ipc.connection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.exceptions import IPCConnectionError, IPCUnavailableError
from kicad_pipeline.ipc.connection import (
    IPCConnectionInfo,
    KiCadConnection,
    _require_kipy,
    connect,
    is_available,
)

# ---------------------------------------------------------------------------
# _require_kipy
# ---------------------------------------------------------------------------


class TestRequireKipy:
    """Tests for the _require_kipy guard."""

    def test_raises_when_kipy_missing(self) -> None:
        with patch("kicad_pipeline.ipc._HAS_KIPY", False), \
             pytest.raises(IPCUnavailableError, match="not installed"):
            _require_kipy()

    def test_passes_when_kipy_present(self) -> None:
        with patch("kicad_pipeline.ipc._HAS_KIPY", True):
            _require_kipy()  # should not raise


# ---------------------------------------------------------------------------
# IPCConnectionInfo
# ---------------------------------------------------------------------------


class TestIPCConnectionInfo:
    """Tests for the frozen dataclass."""

    def test_frozen(self) -> None:
        info = IPCConnectionInfo(
            socket_path="/tmp/kicad.sock", kicad_version="9.0.1", connected=True,
        )
        assert info.socket_path == "/tmp/kicad.sock"
        assert info.kicad_version == "9.0.1"
        assert info.connected is True
        with pytest.raises(AttributeError):
            info.connected = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# KiCadConnection
# ---------------------------------------------------------------------------


class TestKiCadConnection:
    """Tests for the KiCadConnection wrapper."""

    def _make_conn(self) -> KiCadConnection:
        client = MagicMock()
        client.get_version.return_value = "9.0.1"
        return KiCadConnection(client, "/tmp/kicad.sock")

    def test_context_manager(self) -> None:
        conn = self._make_conn()
        with conn:
            assert conn._connected is True
        assert conn._connected is False
        conn.client.close.assert_called_once()

    def test_info_returns_dataclass(self) -> None:
        conn = self._make_conn()
        info = conn.info
        assert isinstance(info, IPCConnectionInfo)
        assert info.kicad_version == "9.0.1"
        assert info.connected is True

    def test_info_handles_version_failure(self) -> None:
        conn = self._make_conn()
        conn.client.get_version.side_effect = RuntimeError("oops")
        info = conn.info
        assert info.kicad_version == ""
        assert info.connected is True

    def test_close_idempotent(self) -> None:
        conn = self._make_conn()
        conn.close()
        conn.close()  # second close should not raise
        assert conn._connected is False

    def test_close_swallows_errors(self) -> None:
        conn = self._make_conn()
        conn.client.close.side_effect = RuntimeError("socket gone")
        conn.close()  # should not raise
        assert conn._connected is False


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


class TestConnect:
    """Tests for the connect() function."""

    def test_raises_when_kipy_missing(self) -> None:
        with patch("kicad_pipeline.ipc._HAS_KIPY", False), \
             pytest.raises(IPCUnavailableError):
            connect()

    def test_connect_default(self) -> None:
        mock_kipy = MagicMock()
        mock_client = MagicMock()
        mock_client.get_version.return_value = "9.0.2"
        mock_kipy.KiCad.return_value = mock_client

        with patch("kicad_pipeline.ipc._HAS_KIPY", True), \
             patch.dict(sys.modules, {"kipy": mock_kipy}):
            conn = connect()
            assert isinstance(conn, KiCadConnection)
            assert conn.info.kicad_version == "9.0.2"
            mock_kipy.KiCad.assert_called_once_with(timeout=2000)

    def test_connect_custom_socket(self) -> None:
        mock_kipy = MagicMock()
        mock_client = MagicMock()
        mock_client.get_version.return_value = "9.0.2"
        mock_kipy.KiCad.return_value = mock_client

        with patch("kicad_pipeline.ipc._HAS_KIPY", True), \
             patch.dict(sys.modules, {"kipy": mock_kipy}):
            conn = connect(socket_path="/tmp/custom.sock", timeout_ms=5000)
            mock_kipy.KiCad.assert_called_once_with(
                address="/tmp/custom.sock", timeout=5000,
            )
            conn.close()

    def test_connect_failure(self) -> None:
        mock_kipy = MagicMock()
        mock_kipy.KiCad.side_effect = ConnectionRefusedError("no KiCad")

        with patch("kicad_pipeline.ipc._HAS_KIPY", True), \
             patch.dict(sys.modules, {"kipy": mock_kipy}), \
             pytest.raises(IPCConnectionError, match="Failed to connect"):
            connect()

    def test_connect_version_ping_failure(self) -> None:
        mock_kipy = MagicMock()
        mock_client = MagicMock()
        mock_client.get_version.side_effect = TimeoutError("ping timeout")
        mock_kipy.KiCad.return_value = mock_client

        with patch("kicad_pipeline.ipc._HAS_KIPY", True), \
             patch.dict(sys.modules, {"kipy": mock_kipy}), \
             pytest.raises(IPCConnectionError, match="Failed to connect"):
            connect()


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------


class TestIsAvailable:
    """Tests for the is_available() convenience check."""

    @patch("kicad_pipeline.ipc.connection.connect")
    def test_true_when_connected(self, mock_connect: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)
        assert is_available() is True

    @patch("kicad_pipeline.ipc.connection.connect")
    def test_false_when_unavailable(self, mock_connect: MagicMock) -> None:
        mock_connect.side_effect = IPCUnavailableError("no kipy")
        assert is_available() is False

    @patch("kicad_pipeline.ipc.connection.connect")
    def test_false_when_connection_fails(self, mock_connect: MagicMock) -> None:
        mock_connect.side_effect = IPCConnectionError("refused")
        assert is_available() is False

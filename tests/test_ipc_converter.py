"""Tests for kicad_pipeline.ipc.converter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.exceptions import IPCSyncError, IPCUnavailableError
from kicad_pipeline.ipc.converter import (
    _mm_to_nm,
    _nm_to_mm,
    footprint_positions_from_board,
    nets_from_board,
    tracks_from_board,
)

# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestUnitConversion:
    """Tests for nm <-> mm helpers."""

    def test_nm_to_mm(self) -> None:
        assert _nm_to_mm(1_000_000) == 1.0
        assert _nm_to_mm(0) == 0.0
        assert _nm_to_mm(500_000) == pytest.approx(0.5)

    def test_mm_to_nm(self) -> None:
        assert _mm_to_nm(1.0) == 1_000_000
        assert _mm_to_nm(0.0) == 0
        assert _mm_to_nm(0.25) == 250_000

    def test_roundtrip(self) -> None:
        for val in (0.0, 0.127, 1.27, 10.0, 50.8):
            assert _nm_to_mm(_mm_to_nm(val)) == pytest.approx(val, abs=1e-6)


# ---------------------------------------------------------------------------
# footprint_positions_from_board
# ---------------------------------------------------------------------------


class TestFootprintPositions:
    """Tests for reading footprint positions from a mocked board."""

    def _make_conn(self) -> MagicMock:
        conn = MagicMock()
        return conn

    def _make_footprint(
        self, ref: str, x_nm: int, y_nm: int, rot: float = 0.0,
    ) -> MagicMock:
        fp = MagicMock()
        fp.reference.value = ref
        fp.position.x = x_nm
        fp.position.y = y_nm
        fp.position.HasField.return_value = True
        fp.position.rotation.value = rot
        return fp

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_extracts_positions(self) -> None:
        conn = self._make_conn()
        board = MagicMock()
        conn.client.get_board.return_value = board

        board.get_footprints.return_value = [
            self._make_footprint("R1", 10_000_000, 20_000_000, 90.0),
            self._make_footprint("C1", 30_000_000, 40_000_000, 0.0),
        ]

        result = footprint_positions_from_board(conn)
        assert result == {
            "R1": (10.0, 20.0, 90.0),
            "C1": (30.0, 40.0, 0.0),
        }

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_empty_board(self) -> None:
        conn = self._make_conn()
        board = MagicMock()
        conn.client.get_board.return_value = board
        board.get_footprints.return_value = []

        result = footprint_positions_from_board(conn)
        assert result == {}

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_raises_on_failure(self) -> None:
        conn = self._make_conn()
        conn.client.get_board.side_effect = RuntimeError("board gone")

        with pytest.raises(IPCSyncError, match="Failed to read footprint"):
            footprint_positions_from_board(conn)

    @patch("kicad_pipeline.ipc._HAS_KIPY", False)
    def test_raises_when_kipy_missing(self) -> None:
        conn = self._make_conn()
        with pytest.raises(IPCUnavailableError):
            footprint_positions_from_board(conn)


# ---------------------------------------------------------------------------
# tracks_from_board
# ---------------------------------------------------------------------------


class TestTracksFromBoard:
    """Tests for reading tracks."""

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_extracts_tracks(self) -> None:
        conn = MagicMock()
        board = MagicMock()
        conn.client.get_board.return_value = board

        trk = MagicMock()
        trk.start.x = 1_000_000
        trk.start.y = 2_000_000
        trk.end.x = 3_000_000
        trk.end.y = 4_000_000
        trk.width = 250_000
        trk.layer.name = "F.Cu"
        trk.net = 5
        board.get_tracks.return_value = [trk]

        result = tracks_from_board(conn)
        assert len(result) == 1
        assert result[0] == (1.0, 2.0, 3.0, 4.0, 0.25, "F.Cu", 5)

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_raises_on_failure(self) -> None:
        conn = MagicMock()
        conn.client.get_board.side_effect = RuntimeError("fail")
        with pytest.raises(IPCSyncError, match="Failed to read tracks"):
            tracks_from_board(conn)


# ---------------------------------------------------------------------------
# nets_from_board
# ---------------------------------------------------------------------------


class TestNetsFromBoard:
    """Tests for reading net list."""

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_extracts_nets(self) -> None:
        conn = MagicMock()
        board = MagicMock()
        conn.client.get_board.return_value = board

        net1 = MagicMock()
        net1.number = 0
        net1.name = ""
        net2 = MagicMock()
        net2.number = 1
        net2.name = "GND"
        board.get_nets.return_value = [net1, net2]

        result = nets_from_board(conn)
        assert result == ((0, ""), (1, "GND"))

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_raises_on_failure(self) -> None:
        conn = MagicMock()
        conn.client.get_board.side_effect = RuntimeError("fail")
        with pytest.raises(IPCSyncError, match="Failed to read nets"):
            nets_from_board(conn)

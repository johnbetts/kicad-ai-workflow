"""Tests for kicad_pipeline.ipc.board_ops."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.exceptions import IPCSyncError, IPCUnavailableError
from kicad_pipeline.ipc.board_ops import (
    pull_board_snapshot,
    pull_footprint_positions,
    push_pcb_to_kicad,
    refill_zones,
)
from kicad_pipeline.ipc.connection import KiCadConnection
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    PCBDesign,
    Point,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn() -> KiCadConnection:
    """Build a KiCadConnection with a mocked client."""
    client = MagicMock()
    client.get_version.return_value = "9.0.1"
    return KiCadConnection(client, "/tmp/test.sock")


def _make_design(footprints: tuple[Footprint, ...] = ()) -> PCBDesign:
    """Build a minimal PCBDesign for testing."""
    return PCBDesign(
        outline=BoardOutline(polygon=(Point(0, 0), Point(50, 0), Point(50, 50), Point(0, 50))),
        design_rules=DesignRules(),
        nets=(NetEntry(0, ""), NetEntry(1, "GND")),
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


# ---------------------------------------------------------------------------
# push_pcb_to_kicad
# ---------------------------------------------------------------------------


class TestPushPcb:
    """Tests for pushing a PCB file to KiCad."""

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_push_calls_revert(self) -> None:
        conn = _make_conn()
        board = MagicMock()
        conn.client.get_board.return_value = board

        push_pcb_to_kicad("/tmp/test.kicad_pcb", conn)

        board.revert.assert_called_once()

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_push_raises_on_failure(self) -> None:
        conn = _make_conn()
        conn.client.get_board.side_effect = RuntimeError("no board open")

        with pytest.raises(IPCSyncError, match="Failed to push"):
            push_pcb_to_kicad("/tmp/test.kicad_pcb", conn)

    @patch("kicad_pipeline.ipc._HAS_KIPY", False)
    def test_push_raises_when_kipy_missing(self) -> None:
        conn = _make_conn()
        with pytest.raises(IPCUnavailableError):
            push_pcb_to_kicad("/tmp/test.kicad_pcb", conn)


# ---------------------------------------------------------------------------
# refill_zones
# ---------------------------------------------------------------------------


class TestRefillZones:
    """Tests for IPC zone fill."""

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_refill_calls_api(self) -> None:
        conn = _make_conn()
        board = MagicMock()
        conn.client.get_board.return_value = board

        refill_zones(conn)

        board.refill_zones.assert_called_once()

    @patch("kicad_pipeline.ipc._HAS_KIPY", True)
    def test_refill_raises_on_failure(self) -> None:
        conn = _make_conn()
        conn.client.get_board.side_effect = RuntimeError("fail")

        with pytest.raises(IPCSyncError, match="Failed to refill"):
            refill_zones(conn)

    @patch("kicad_pipeline.ipc._HAS_KIPY", False)
    def test_refill_raises_when_kipy_missing(self) -> None:
        conn = _make_conn()
        with pytest.raises(IPCUnavailableError):
            refill_zones(conn)


# ---------------------------------------------------------------------------
# pull_footprint_positions
# ---------------------------------------------------------------------------


class TestPullFootprintPositions:
    """Tests for pull_footprint_positions (delegates to converter)."""

    @patch("kicad_pipeline.ipc.converter.footprint_positions_from_board")
    def test_delegates_to_converter(self, mock_conv: MagicMock) -> None:
        mock_conv.return_value = {"R1": (10.0, 20.0, 0.0)}
        conn = _make_conn()

        result = pull_footprint_positions(conn)
        assert result == {"R1": (10.0, 20.0, 0.0)}
        mock_conv.assert_called_once_with(conn)


# ---------------------------------------------------------------------------
# pull_board_snapshot
# ---------------------------------------------------------------------------


class TestPullBoardSnapshot:
    """Tests for merging live board state into a PCBDesign."""

    @patch("kicad_pipeline.ipc.converter.footprint_positions_from_board")
    def test_updates_footprint_positions(self, mock_conv: MagicMock) -> None:
        fp = Footprint(
            lib_id="R_0805:R_0805_2012Metric",
            ref="R1",
            value="10k",
            position=Point(0.0, 0.0),
        )
        design = _make_design(footprints=(fp,))

        mock_conv.return_value = {"R1": (15.0, 25.0, 90.0)}
        conn = _make_conn()

        result = pull_board_snapshot(conn, design)
        updated = result.get_footprint("R1")
        assert updated is not None
        assert updated.position.x == 15.0
        assert updated.position.y == 25.0
        assert updated.rotation == 90.0

    @patch("kicad_pipeline.ipc.converter.footprint_positions_from_board")
    def test_preserves_unknown_refs(self, mock_conv: MagicMock) -> None:
        fp = Footprint(
            lib_id="R_0805:R_0805_2012Metric",
            ref="R1",
            value="10k",
            position=Point(5.0, 5.0),
        )
        design = _make_design(footprints=(fp,))

        mock_conv.return_value = {}  # R1 not found in live board
        conn = _make_conn()

        result = pull_board_snapshot(conn, design)
        assert result.get_footprint("R1") is not None
        assert result.get_footprint("R1").position == Point(5.0, 5.0)  # type: ignore[union-attr]

    @patch("kicad_pipeline.ipc.converter.footprint_positions_from_board")
    def test_raises_on_sync_failure(self, mock_conv: MagicMock) -> None:
        mock_conv.side_effect = IPCSyncError("read failed")
        conn = _make_conn()
        design = _make_design()

        with pytest.raises(IPCSyncError):
            pull_board_snapshot(conn, design)


# ---------------------------------------------------------------------------
# CLI --live flag integration
# ---------------------------------------------------------------------------


class TestCLILiveFlag:
    """Test that --live flag is properly wired."""

    def test_pcb_parser_accepts_live(self) -> None:
        from kicad_pipeline.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["pcb", "-r", "req.json", "-o", "out.kicad_pcb", "--live"])
        assert args.live is True

    def test_pipeline_parser_accepts_live(self) -> None:
        from kicad_pipeline.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["pipeline", "-r", "req.json", "-o", "out/", "--live"])
        assert args.live is True

    def test_pcb_parser_default_no_live(self) -> None:
        from kicad_pipeline.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["pcb", "-r", "req.json", "-o", "out.kicad_pcb"])
        assert args.live is False

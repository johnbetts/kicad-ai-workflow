"""Tests for the dashboard API endpoints using Starlette TestClient."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

nicegui = pytest.importorskip("nicegui")

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Create a minimal output directory with a fake board."""
    board_dir = tmp_path / "test_board"
    board_dir.mkdir()
    # Create a dummy .kicad_pcb file so the board is discovered
    (board_dir / "test_board.kicad_pcb").write_text("(kicad_pcb)")
    return tmp_path


@pytest.fixture()
def app_with_routes(output_dir: Path):
    """Create a Starlette test app with API routes registered."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from kicad_pipeline.dashboard.api import register_api_routes

    test_app = Starlette()
    register_api_routes(test_app, output_dir)
    client = TestClient(test_app)
    return client


class TestPostEvidence:
    """POST /api/evidence creates a record in the ledger."""

    def test_post_evidence(self, app_with_routes, output_dir: Path) -> None:
        client = app_with_routes
        record_data = {
            "kind": "render",
            "stage": "pcb",
            "step": "test_render",
            "board": "test_board",
            "passed": True,
            "summary": "Test render completed",
        }
        response = client.post("/api/evidence", json=record_data)
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "ok"
        assert "id" in body

        # Verify the record was written to the ledger file
        ledger_file = output_dir / "test_board" / ".evidence" / "test_board.jsonl"
        assert ledger_file.exists()
        lines = ledger_file.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["kind"] == "render"
        assert parsed["board"] == "test_board"


class TestGetLedger:
    """GET /api/ledger/{board_name} returns records."""

    def test_get_ledger(self, app_with_routes, output_dir: Path) -> None:
        client = app_with_routes
        # First post a record
        client.post(
            "/api/evidence",
            json={
                "kind": "score",
                "stage": "pcb",
                "step": "scoring",
                "board": "test_board",
                "passed": True,
                "summary": "Score snapshot",
            },
        )
        response = client.get("/api/ledger/test_board")
        assert response.status_code == 200
        body = response.json()
        assert body["board"] == "test_board"
        assert len(body["records"]) == 1
        assert body["records"][0]["kind"] == "score"

    def test_get_ledger_empty(self, app_with_routes) -> None:
        """GET returns empty ledger for unknown board."""
        client = app_with_routes
        response = client.get("/api/ledger/nonexistent_board")
        assert response.status_code == 200
        body = response.json()
        assert body["board"] == "nonexistent_board"
        assert body["records"] == []


class TestApprove:
    """POST /api/approve/{board_name} writes HUMAN_APPROVAL."""

    def test_approve_creates_record(self, app_with_routes, output_dir: Path) -> None:
        client = app_with_routes
        response = client.post(
            "/api/approve/test_board",
            json={"stage": "validation"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "ok"

        # Verify ledger contains approval
        ledger_file = output_dir / "test_board" / ".evidence" / "test_board.jsonl"
        assert ledger_file.exists()
        lines = ledger_file.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["kind"] == "human_approval"
        assert record["stage"] == "validation"
        assert record["passed"] is True


class TestReject:
    """POST /api/reject/{board_name} writes HUMAN_REJECTION."""

    def test_reject_creates_record(self, app_with_routes, output_dir: Path) -> None:
        client = app_with_routes
        response = client.post(
            "/api/reject/test_board",
            json={"stage": "pcb", "feedback": "Components overlap near U1"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "ok"

        # Verify ledger contains rejection with feedback
        ledger_file = output_dir / "test_board" / ".evidence" / "test_board.jsonl"
        assert ledger_file.exists()
        lines = ledger_file.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["kind"] == "human_rejection"
        assert record["stage"] == "pcb"
        assert record["passed"] is False
        assert record["feedback"] == "Components overlap near U1"


class TestListBoards:
    """GET /api/boards lists available boards."""

    def test_list_boards(self, app_with_routes) -> None:
        client = app_with_routes
        response = client.get("/api/boards")
        assert response.status_code == 200
        body = response.json()
        assert "test_board" in body["boards"]

    def test_list_boards_excludes_hidden(self, app_with_routes, output_dir: Path) -> None:
        """Hidden and underscore directories are excluded."""
        client = app_with_routes
        # Create dirs that should be excluded
        (output_dir / ".hidden_board").mkdir()
        (output_dir / "_archive").mkdir()
        response = client.get("/api/boards")
        body = response.json()
        assert ".hidden_board" not in body["boards"]
        assert "_archive" not in body["boards"]


class TestPostLog:
    """POST /api/log pushes to the log buffer."""

    def test_post_log(self, app_with_routes) -> None:
        client = app_with_routes
        response = client.post(
            "/api/log",
            json={"message": "Build started", "level": "info"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "ok"

        # Verify it ended up in the buffer
        from kicad_pipeline.dashboard.app import _log_buffer

        assert any("Build started" in entry for entry in _log_buffer)

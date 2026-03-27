"""Tests for the kanban board data model, persistence, and API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    """Return a temporary project root directory."""
    return tmp_path


@pytest.fixture()
def sample_roadmap(tmp_path: Path) -> Path:
    """Create a sample roadmap.md for import testing."""
    content = """\
# Roadmap

## Bugs (fix these first)
- [ ] **P0** BUG-001: Agents skip verification steps
- [ ] **P1** BUG-002: Agents repeat known mistakes

## Phases

### Phase 1 - Core
- [x] TASK-001: Create evidence models
- [ ] TASK-002: Create evidence ledger

## Wishlist
- [ ] WISH-001: Real-time render preview
- [ ] WISH-002: Annotation overlay on PCB images

## Moonshots
- [ ] MOON-001: Hosted multi-tenant service
- [ ] MOON-002: Natural language to manufactured PCB
"""
    path = tmp_path / "roadmap.md"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Unit tests — data model and persistence
# ---------------------------------------------------------------------------


class TestKanbanCardCreation:
    """KanbanCard defaults and field validation."""

    def test_kanban_card_creation(self) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard

        card = KanbanCard(title="Test card")
        assert card.title == "Test card"
        assert card.card_type == "feature"
        assert card.priority == "P2"
        assert card.level == "framework"
        assert card.status == "backlog"
        assert card.description == ""
        assert card.board_name == ""
        assert card.notes == []
        assert card.id  # auto-generated
        assert card.created  # auto-generated
        assert card.updated  # auto-generated

    def test_kanban_card_custom_fields(self) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard

        card = KanbanCard(
            title="Fix relay placement",
            description="Relays overlap with connectors",
            card_type="bug",
            priority="P0",
            level="board",
            board_name="train_relay",
            status="in_progress",
        )
        assert card.card_type == "bug"
        assert card.priority == "P0"
        assert card.level == "board"
        assert card.board_name == "train_relay"
        assert card.status == "in_progress"


class TestKanbanCardNotes:
    """Notes field on KanbanCard."""

    def test_notes_default_empty(self) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard

        card = KanbanCard(title="No notes")
        assert card.notes == []

    def test_notes_initialized(self) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard

        card = KanbanCard(
            title="With notes",
            notes=["2026-03-27 10:30 -- First note"],
        )
        assert len(card.notes) == 1
        assert "First note" in card.notes[0]

    def test_add_note(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, add_note

        card = add_card(project_root, KanbanCard(title="Noted card"))
        updated = add_note(project_root, card.id, "Fixed placement issue")
        assert len(updated.notes) == 1
        assert "Fixed placement issue" in updated.notes[0]
        # Verify timestamp prefix
        assert updated.notes[0].count("\u2014") >= 1

    def test_add_multiple_notes(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, add_note, load_kanban

        card = add_card(project_root, KanbanCard(title="Multi-note card"))
        add_note(project_root, card.id, "Note one")
        add_note(project_root, card.id, "Note two")
        add_note(project_root, card.id, "Note three")
        reloaded = load_kanban(project_root)
        found = next(c for c in reloaded.cards if c.id == card.id)
        assert len(found.notes) == 3

    def test_add_note_not_found(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import add_note

        with pytest.raises(KeyError, match="not found"):
            add_note(project_root, "nonexistent", "Orphan note")

    def test_notes_roundtrip_persistence(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, add_note, load_kanban

        card = add_card(project_root, KanbanCard(title="Persist test"))
        add_note(project_root, card.id, "Survived reload")
        reloaded = load_kanban(project_root)
        found = next(c for c in reloaded.cards if c.id == card.id)
        assert len(found.notes) == 1
        assert "Survived reload" in found.notes[0]


class TestRelativeTime:
    """_relative_time helper produces human-readable relative timestamps."""

    def test_just_now(self) -> None:
        from kicad_pipeline.dashboard.kanban import _relative_time, _utc_now_iso

        assert _relative_time(_utc_now_iso()) == "just now"

    def test_minutes_ago(self) -> None:
        from datetime import datetime, timedelta, timezone

        from kicad_pipeline.dashboard.kanban import _relative_time

        past = (datetime.now(tz=timezone.utc) - timedelta(minutes=5)).isoformat()
        assert _relative_time(past) == "5m ago"

    def test_hours_ago(self) -> None:
        from datetime import datetime, timedelta, timezone

        from kicad_pipeline.dashboard.kanban import _relative_time

        past = (datetime.now(tz=timezone.utc) - timedelta(hours=3)).isoformat()
        assert _relative_time(past) == "3h ago"

    def test_days_ago(self) -> None:
        from datetime import datetime, timedelta, timezone

        from kicad_pipeline.dashboard.kanban import _relative_time

        past = (datetime.now(tz=timezone.utc) - timedelta(days=7)).isoformat()
        assert _relative_time(past) == "7d ago"

    def test_invalid_timestamp(self) -> None:
        from kicad_pipeline.dashboard.kanban import _relative_time

        assert _relative_time("not-a-date") == "unknown"


class TestLoadEmptyKanban:
    """Loading from missing/empty file returns empty board."""

    def test_load_empty_kanban(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import load_kanban

        board = load_kanban(project_root)
        assert board.cards == []

    def test_load_corrupt_file(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KANBAN_DIR, KANBAN_FILE, load_kanban

        kanban_dir = project_root / KANBAN_DIR
        kanban_dir.mkdir()
        (kanban_dir / KANBAN_FILE).write_text("not json{{{")
        board = load_kanban(project_root)
        assert board.cards == []


class TestAddCard:
    """add_card persists a card to disk."""

    def test_add_card(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, load_kanban

        card = KanbanCard(title="New feature")
        result = add_card(project_root, card)
        assert result.title == "New feature"
        assert result.id == card.id

        # Verify persistence
        reloaded = load_kanban(project_root)
        assert len(reloaded.cards) == 1
        assert reloaded.cards[0].title == "New feature"

    def test_add_multiple_cards(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, load_kanban

        add_card(project_root, KanbanCard(title="Card A"))
        add_card(project_root, KanbanCard(title="Card B"))
        add_card(project_root, KanbanCard(title="Card C"))
        reloaded = load_kanban(project_root)
        assert len(reloaded.cards) == 3


class TestMoveCard:
    """move_card changes the status column."""

    def test_move_card(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, load_kanban, move_card

        card = add_card(project_root, KanbanCard(title="To move"))
        moved = move_card(project_root, card.id, "in_progress")
        assert moved.status == "in_progress"

        reloaded = load_kanban(project_root)
        assert reloaded.cards[0].status == "in_progress"

    def test_move_card_invalid_status(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, move_card

        card = add_card(project_root, KanbanCard(title="Bad move"))
        with pytest.raises(ValueError, match="Invalid status"):
            move_card(project_root, card.id, "invalid_column")

    def test_move_card_not_found(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import move_card

        with pytest.raises(KeyError, match="not found"):
            move_card(project_root, "nonexistent", "done")


class TestUpdateCard:
    """update_card changes arbitrary fields."""

    def test_update_card(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, update_card

        card = add_card(project_root, KanbanCard(title="Original"))
        updated = update_card(
            project_root, card.id, {"title": "Renamed", "priority": "P0"}
        )
        assert updated.title == "Renamed"
        assert updated.priority == "P0"
        assert updated.updated != card.updated  # timestamp refreshed

    def test_update_card_not_found(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import update_card

        with pytest.raises(KeyError, match="not found"):
            update_card(project_root, "nonexistent", {"title": "Nope"})


class TestDeleteCard:
    """delete_card removes a card by ID."""

    def test_delete_card(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, delete_card, load_kanban

        card = add_card(project_root, KanbanCard(title="To delete"))
        delete_card(project_root, card.id)
        reloaded = load_kanban(project_root)
        assert len(reloaded.cards) == 0

    def test_delete_card_not_found(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import delete_card

        with pytest.raises(KeyError, match="not found"):
            delete_card(project_root, "nonexistent")


class TestImportFromRoadmap:
    """import_from_roadmap parses roadmap.md format."""

    def test_import_from_roadmap(self, project_root: Path, sample_roadmap: Path) -> None:
        from kicad_pipeline.dashboard.kanban import import_from_roadmap, load_kanban

        count = import_from_roadmap(project_root, sample_roadmap)
        assert count == 8

        board = load_kanban(project_root)
        assert len(board.cards) == 8

        titles = {c.title for c in board.cards}
        assert "Agents skip verification steps" in titles
        assert "Create evidence models" in titles
        assert "Real-time render preview" in titles
        assert "Hosted multi-tenant service" in titles

        # Check types
        by_title = {c.title: c for c in board.cards}
        assert by_title["Agents skip verification steps"].card_type == "bug"
        assert by_title["Agents skip verification steps"].priority == "P0"
        assert by_title["Agents skip verification steps"].status == "backlog"

        assert by_title["Create evidence models"].card_type == "feature"
        assert by_title["Create evidence models"].status == "done"

        assert by_title["Real-time render preview"].card_type == "wishlist"
        assert by_title["Hosted multi-tenant service"].card_type == "moonshot"

    def test_import_skips_duplicates(self, project_root: Path, sample_roadmap: Path) -> None:
        from kicad_pipeline.dashboard.kanban import import_from_roadmap, load_kanban

        count1 = import_from_roadmap(project_root, sample_roadmap)
        assert count1 == 8
        count2 = import_from_roadmap(project_root, sample_roadmap)
        assert count2 == 0

        board = load_kanban(project_root)
        assert len(board.cards) == 8  # no duplicates

    def test_import_missing_file(self, project_root: Path) -> None:
        from pathlib import Path

        from kicad_pipeline.dashboard.kanban import import_from_roadmap

        count = import_from_roadmap(project_root, Path("/nonexistent/roadmap.md"))
        assert count == 0


class TestFilterByLevel:
    """KanbanBoard.filter_by_level returns matching cards."""

    def test_filter_by_level(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, load_kanban

        add_card(project_root, KanbanCard(title="FW bug", level="framework"))
        add_card(project_root, KanbanCard(title="Deploy config", level="deployment"))
        add_card(project_root, KanbanCard(title="Board fix", level="board", board_name="relay"))
        add_card(project_root, KanbanCard(title="Another FW", level="framework"))

        board = load_kanban(project_root)
        fw = board.filter_by_level("framework")
        assert len(fw) == 2
        deploy = board.filter_by_level("deployment")
        assert len(deploy) == 1
        brd = board.filter_by_level("board")
        assert len(brd) == 1


class TestPrioritySort:
    """KanbanBoard.sorted_by_priority orders P0 first."""

    def test_priority_sort(self, project_root: Path) -> None:
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card, load_kanban

        add_card(project_root, KanbanCard(title="Low", priority="P3"))
        add_card(project_root, KanbanCard(title="Critical", priority="P0"))
        add_card(project_root, KanbanCard(title="Medium", priority="P2"))
        add_card(project_root, KanbanCard(title="High", priority="P1"))

        board = load_kanban(project_root)
        sorted_cards = board.sorted_by_priority(board.cards)
        priorities = [c.priority for c in sorted_cards]
        assert priorities == ["P0", "P1", "P2", "P3"]


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

nicegui = pytest.importorskip("nicegui")


@pytest.fixture()
def kanban_api_client(tmp_path: Path):
    """Create a Starlette test app with kanban API routes.

    Uses tmp_path as project root with an ``output/`` subdirectory so that
    ``_project_root()`` (output_root.parent) resolves back to tmp_path.
    """
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from kicad_pipeline.dashboard.api import register_api_routes

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # Create a board dir so existing routes work
    board_dir = output_dir / "test_board"
    board_dir.mkdir()
    (board_dir / "test_board.kicad_pcb").write_text("(kicad_pcb)")

    test_app = Starlette()
    register_api_routes(test_app, output_dir)
    return TestClient(test_app), tmp_path


class TestKanbanAPI:
    """API endpoints for kanban CRUD."""

    def test_get_empty_kanban(self, kanban_api_client) -> None:  # type: ignore[no-untyped-def]
        client, _ = kanban_api_client
        response = client.get("/api/kanban")
        assert response.status_code == 200
        body = response.json()
        assert body["cards"] == []

    def test_create_card_via_api(self, kanban_api_client) -> None:  # type: ignore[no-untyped-def]
        client, _ = kanban_api_client
        response = client.post(
            "/api/kanban/cards",
            json={"title": "API card", "priority": "P1", "card_type": "bug"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["title"] == "API card"
        assert body["priority"] == "P1"
        assert body["id"]

    def test_update_card_via_api(self, kanban_api_client) -> None:  # type: ignore[no-untyped-def]
        client, _ = kanban_api_client
        create_resp = client.post(
            "/api/kanban/cards", json={"title": "To update"}
        )
        card_id = create_resp.json()["id"]
        response = client.put(
            f"/api/kanban/cards/{card_id}",
            json={"title": "Updated title"},
        )
        assert response.status_code == 200
        assert response.json()["title"] == "Updated title"

    def test_move_card_via_api(self, kanban_api_client) -> None:  # type: ignore[no-untyped-def]
        client, _ = kanban_api_client
        create_resp = client.post(
            "/api/kanban/cards", json={"title": "To move"}
        )
        card_id = create_resp.json()["id"]
        response = client.put(
            f"/api/kanban/cards/{card_id}/move",
            json={"status": "review"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "review"

    def test_delete_card_via_api(self, kanban_api_client) -> None:  # type: ignore[no-untyped-def]
        client, _ = kanban_api_client
        create_resp = client.post(
            "/api/kanban/cards", json={"title": "To delete"}
        )
        card_id = create_resp.json()["id"]
        response = client.delete(f"/api/kanban/cards/{card_id}")
        assert response.status_code == 200

        # Verify it's gone
        get_resp = client.get("/api/kanban")
        assert len(get_resp.json()["cards"]) == 0

    def test_import_roadmap_via_api(self, kanban_api_client, sample_roadmap: Path) -> None:
        client, _ = kanban_api_client
        response = client.post(
            "/api/kanban/import-roadmap",
            json={"roadmap_path": str(sample_roadmap)},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["imported"] == 8

    def test_update_nonexistent_card(self, kanban_api_client) -> None:  # type: ignore[no-untyped-def]
        client, _ = kanban_api_client
        response = client.put(
            "/api/kanban/cards/nonexistent",
            json={"title": "Nope"},
        )
        assert response.status_code == 404

    def test_move_invalid_status(self, kanban_api_client) -> None:  # type: ignore[no-untyped-def]
        client, _ = kanban_api_client
        create_resp = client.post(
            "/api/kanban/cards", json={"title": "Bad move"}
        )
        card_id = create_resp.json()["id"]
        response = client.put(
            f"/api/kanban/cards/{card_id}/move",
            json={"status": "invalid"},
        )
        assert response.status_code == 422

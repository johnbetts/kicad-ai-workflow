"""Kanban board data model and persistence for multi-level project tracking.

Operates at three levels:
- Framework: pipeline codebase bugs (KI-NNN), features, releases
- Deployment: shared config, review policies, component libraries
- Board: per-PCB bugs, placement issues, manufacturing readiness
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

KANBAN_DIR = ".kanban"
KANBAN_FILE = "kanban.json"

VALID_STATUSES = ("backlog", "in_progress", "review", "done")
VALID_LEVELS = ("framework", "deployment", "board")
VALID_TYPES = ("bug", "feature", "release", "wishlist", "moonshot")
VALID_PRIORITIES = ("P0", "P1", "P2", "P3")


def _generate_id() -> str:
    """Generate a short unique identifier."""
    return uuid4().hex[:12]


def _utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


class KanbanCard(BaseModel):
    """A single card on the kanban board."""

    id: str = Field(default_factory=_generate_id)
    title: str
    description: str = ""
    card_type: str = "feature"
    priority: str = "P2"
    level: str = "framework"
    board_name: str = ""
    status: str = "backlog"
    notes: list[str] = Field(default_factory=list)
    created: str = Field(default_factory=_utc_now_iso)
    updated: str = Field(default_factory=_utc_now_iso)


def _relative_time(iso_timestamp: str) -> str:
    """Convert an ISO timestamp to a human-readable relative time string."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        now = datetime.now(tz=timezone.utc)
        # Ensure both are timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = now - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        if days < 30:
            return f"{days}d ago"
        months = days // 30
        return f"{months}mo ago"
    except (ValueError, TypeError):
        return "unknown"


def add_note(project_root: Path, card_id: str, note_text: str) -> KanbanCard:
    """Add a timestamped note to a card. Returns the updated card."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    stamped_note = f"{timestamp} \u2014 {note_text}"
    board = load_kanban(project_root)
    for i, card in enumerate(board.cards):
        if card.id == card_id:
            new_notes = [*card.notes, stamped_note]
            board.cards[i] = card.model_copy(
                update={"notes": new_notes, "updated": _utc_now_iso()}
            )
            save_kanban(project_root, board)
            return board.cards[i]
    msg = f"Card {card_id} not found"
    raise KeyError(msg)


class KanbanBoard(BaseModel):
    """Full kanban board state — persisted as a single JSON file."""

    cards: list[KanbanCard] = Field(default_factory=list)

    def filter_by_level(self, level: str) -> list[KanbanCard]:
        """Return cards matching a level, optionally filtered by board_name."""
        return [c for c in self.cards if c.level == level]

    def filter_by_status(self, status: str) -> list[KanbanCard]:
        """Return cards matching a status column."""
        return [c for c in self.cards if c.status == status]

    def sorted_by_priority(self, cards: list[KanbanCard]) -> list[KanbanCard]:
        """Sort cards by priority (P0 first)."""
        priority_order = {p: i for i, p in enumerate(VALID_PRIORITIES)}
        return sorted(cards, key=lambda c: priority_order.get(c.priority, 99))


def _kanban_path(project_root: Path) -> Path:
    """Return the path to the kanban JSON file."""
    return project_root / KANBAN_DIR / KANBAN_FILE


def load_kanban(project_root: Path) -> KanbanBoard:
    """Load the kanban board from disk, returning empty board if missing."""
    path = _kanban_path(project_root)
    if not path.exists():
        return KanbanBoard()
    try:
        data = json.loads(path.read_text())
        return KanbanBoard.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Corrupt kanban file at %s, returning empty board", path)
        return KanbanBoard()


def save_kanban(project_root: Path, board: KanbanBoard) -> None:
    """Persist the kanban board to disk."""
    path = _kanban_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board.model_dump(), indent=2) + "\n")


def add_card(project_root: Path, card: KanbanCard) -> KanbanCard:
    """Add a card to the board and persist. Returns the card with generated ID."""
    board = load_kanban(project_root)
    # Ensure timestamps are set
    now = _utc_now_iso()
    if not card.created:
        card = card.model_copy(update={"created": now})
    if not card.updated:
        card = card.model_copy(update={"updated": now})
    board.cards.append(card)
    save_kanban(project_root, board)
    return card


def update_card(project_root: Path, card_id: str, updates: dict[str, object]) -> KanbanCard:
    """Update fields on a card. Raises KeyError if card not found."""
    board = load_kanban(project_root)
    for i, card in enumerate(board.cards):
        if card.id == card_id:
            updates["updated"] = _utc_now_iso()
            board.cards[i] = card.model_copy(update=updates)
            save_kanban(project_root, board)
            return board.cards[i]
    msg = f"Card {card_id} not found"
    raise KeyError(msg)


def move_card(project_root: Path, card_id: str, new_status: str) -> KanbanCard:
    """Move a card to a new status column. Raises ValueError for invalid status."""
    if new_status not in VALID_STATUSES:
        msg = f"Invalid status: {new_status}. Must be one of {VALID_STATUSES}"
        raise ValueError(msg)
    return update_card(project_root, card_id, {"status": new_status})


def delete_card(project_root: Path, card_id: str) -> None:
    """Delete a card by ID. Raises KeyError if not found."""
    board = load_kanban(project_root)
    original_count = len(board.cards)
    board.cards = [c for c in board.cards if c.id != card_id]
    if len(board.cards) == original_count:
        msg = f"Card {card_id} not found"
        raise KeyError(msg)
    save_kanban(project_root, board)


def import_from_roadmap(project_root: Path, roadmap_path: Path) -> int:
    """Parse a roadmap.md and create kanban cards from its items.

    Recognizes:
    - ``- [ ] **P0** BUG-001: description`` -> bug, backlog
    - ``- [x] TASK-001: description`` -> feature, done
    - Lines under ``## Wishlist`` -> wishlist
    - Lines under ``## Moonshots`` -> moonshot
    - Skips duplicates (matched by title).

    Returns:
        Number of cards imported.
    """
    if not roadmap_path.exists():
        return 0

    board = load_kanban(project_root)
    existing_titles = {c.title for c in board.cards}

    text = roadmap_path.read_text()
    lines = text.split("\n")

    current_section: str | None = None
    imported = 0

    # Pattern: - [ ] or - [x] with optional **Px** and ID: description
    checkbox_re = re.compile(
        r"^-\s+\[([ xX])\]\s+"  # checkbox
        r"(?:\*\*([Pp][0-3])\*\*\s+)?"  # optional **P0**
        r"(?:(BUG|TASK|WISH|MOON)-\d+:\s+)?"  # optional ID prefix
        r"(.+)$"  # description/title
    )

    for line in lines:
        stripped = line.strip()

        # Detect section headers
        if stripped.startswith("## "):
            header = stripped[3:].strip().lower()
            if "wishlist" in header:
                current_section = "wishlist"
            elif "moonshot" in header:
                current_section = "moonshot"
            elif "bug" in header:
                current_section = "bug"
            else:
                current_section = None
            continue

        match = checkbox_re.match(stripped)
        if not match:
            continue

        checked = match.group(1).lower() == "x"
        priority = (match.group(2) or "P2").upper()
        id_prefix = match.group(3)
        title = match.group(4).strip()

        if title in existing_titles:
            continue

        # Determine card type
        if current_section in ("wishlist", "moonshot"):
            card_type = current_section
        elif id_prefix == "BUG" or current_section == "bug":
            card_type = "bug"
        else:
            card_type = "feature"

        status = "done" if checked else "backlog"

        card = KanbanCard(
            title=title,
            card_type=card_type,
            priority=priority,
            status=status,
            level="framework",
        )
        board.cards.append(card)
        existing_titles.add(title)
        imported += 1

    if imported > 0:
        save_kanban(project_root, board)

    return imported

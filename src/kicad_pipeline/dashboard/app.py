"""NiceGUI review dashboard for the KiCad AI pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Shared log buffer that the API pushes to and the log panel reads from.
_log_buffer: list[str] = []

# Currently selected board directory (set by the selector).
_current_board_dir: Path | None = None


def _discover_boards(output_root: Path) -> list[str]:
    """Return sorted board directory names from the output/ folder.

    Includes both ``train_*`` directories and any other board directory
    containing a ``.kicad_pcb`` file.
    """
    if not output_root.is_dir():
        return []
    boards: list[str] = []
    for child in sorted(output_root.iterdir()):
        if (
            child.is_dir()
            and not child.name.startswith(("_", "."))
            and any(child.glob("*.kicad_pcb"))
        ):
            boards.append(child.name)
    return boards


def _latest_grade(board_dir: Path) -> str:
    """Return the latest letter grade for a board, or '?' if none."""
    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind

    pcbs = list(board_dir.glob("*.kicad_pcb"))
    if not pcbs:
        return "?"
    ledger = load_ledger(pcbs[0])
    score_records = ledger.filter_by_kind(EvidenceKind.SCORE)
    if not score_records:
        return "?"
    latest = score_records[-1]
    return str(latest.details.get("grade", "?")) if latest.details else "?"


def _board_has_evidence(board_dir: Path) -> bool:
    """Return True if the board has any evidence records."""
    evidence_dir = board_dir / ".evidence"
    if not evidence_dir.is_dir():
        return False
    return any(evidence_dir.glob("*.jsonl"))


def _pick_default_board(output_root: Path, boards: list[str]) -> str | None:
    """Choose the best default board: first with evidence, else first alphabetically."""
    if not boards:
        return None
    for name in boards:
        if _board_has_evidence(output_root / name):
            return name
    return boards[0]


def main(
    board_path: str | None = None,
    port: int = 8080,
    output_dir: str | None = None,
) -> None:
    """Launch the NiceGUI review dashboard."""
    from pathlib import Path

    from nicegui import app as nicegui_app
    from nicegui import ui

    from kicad_pipeline.dashboard.api import register_api_routes
    from kicad_pipeline.dashboard.panels import (
        build_context_panel,
        build_image_panel,
        build_log_panel,
    )

    global _current_board_dir

    project_root = Path.cwd()
    output_root = Path(output_dir) if output_dir else project_root / "output"

    # Resolve initial board directory
    if board_path:
        bp = Path(board_path)
        if bp.is_dir():
            _current_board_dir = bp
        elif bp.is_file():
            _current_board_dir = bp.parent
        else:
            # Treat as board name
            candidate = output_root / board_path
            if candidate.is_dir():
                _current_board_dir = candidate

    # Register FastAPI routes
    register_api_routes(nicegui_app, output_root)

    @ui.page("/")
    def index() -> None:
        global _current_board_dir

        board_names = _discover_boards(output_root)

        # Build display labels with latest grade
        board_options: dict[str, str] = {}
        for name in board_names:
            grade = _latest_grade(output_root / name)
            label = f"{name} [{grade}]" if grade != "?" else name
            board_options[name] = label

        # Auto-select default board
        if _current_board_dir is None:
            default_name = _pick_default_board(output_root, board_names)
            if default_name:
                _current_board_dir = output_root / default_name

        ui.dark_mode(True)
        with ui.header().classes("items-center justify-between"):
            ui.label("KiCad Review Dashboard").classes("text-h5 font-bold")
            board_select = ui.select(
                options=board_options,
                value=_current_board_dir.name if _current_board_dir else None,
                label="Board",
            ).classes("w-64")

        # Containers that get rebuilt when board changes
        image_container = ui.element("div")
        log_container = ui.element("div")
        context_container = ui.element("div")

        def _rebuild_panels() -> None:
            image_container.clear()
            with image_container:
                build_image_panel(_current_board_dir)
            log_container.clear()
            with log_container:
                build_log_panel()
            context_container.clear()
            with context_container:
                build_context_panel(_current_board_dir, output_root)

        def _on_board_change(e: object) -> None:
            global _current_board_dir
            # e is a ValueChangeEventArguments with .value
            value = getattr(e, "value", None)
            _current_board_dir = output_root / value if value else None
            _rebuild_panels()

        board_select.on_value_change(_on_board_change)

        with ui.splitter(value=25).classes("w-full h-full") as outer_splitter:
            with outer_splitter.before, image_container:
                build_image_panel(_current_board_dir)

            with (
                outer_splitter.after,
                ui.splitter(value=60).classes("w-full h-full") as inner_splitter,
            ):
                with inner_splitter.before, log_container:
                    build_log_panel()

                with inner_splitter.after, context_container:
                    build_context_panel(_current_board_dir, output_root)

    @ui.page("/kanban")
    def kanban_page() -> None:
        """Multi-level kanban board for tracking bugs, features, and releases."""
        from kicad_pipeline.dashboard.kanban import (
            VALID_PRIORITIES,
            VALID_STATUSES,
            VALID_TYPES,
            KanbanCard,
            add_card,
            delete_card,
            load_kanban,
            move_card,
            update_card,
        )

        ui.dark_mode(True)

        # ---- State ----
        active_level = {"value": "framework"}
        active_board_name = {"value": ""}
        active_type_filters: set[str] = set(VALID_TYPES)

        column_labels = {
            "backlog": "Backlog",
            "in_progress": "In Progress",
            "review": "Review",
            "done": "Done",
        }

        type_colors = {
            "bug": "red",
            "feature": "blue",
            "release": "green",
            "wishlist": "purple",
            "moonshot": "amber",
        }

        priority_colors = {
            "P0": "red",
            "P1": "orange",
            "P2": "blue",
            "P3": "grey",
        }

        board_container = ui.element("div")

        def _get_filtered_cards() -> list[KanbanCard]:
            """Load and filter cards by active level, board, and type."""
            kb = load_kanban(project_root)
            cards = kb.filter_by_level(active_level["value"])
            if active_level["value"] == "board" and active_board_name["value"]:
                cards = [
                    c for c in cards if c.board_name == active_board_name["value"]
                ]
            cards = [c for c in cards if c.card_type in active_type_filters]
            return kb.sorted_by_priority(cards)

        def _rebuild_board() -> None:
            """Rebuild the kanban columns."""
            board_container.clear()
            cards = _get_filtered_cards()
            with board_container, ui.row().classes("w-full gap-4"):
                    for status in VALID_STATUSES:
                        col_cards = [c for c in cards if c.status == status]
                        with ui.card().classes("flex-1 min-w-64"):
                            ui.label(column_labels[status]).classes(
                                "text-h6 font-bold q-mb-sm"
                            )
                            ui.separator()
                            with ui.scroll_area().classes("h-96"):
                                for card in col_cards:
                                    _render_card(card, status)
                            if not col_cards:
                                ui.label("No cards").classes(
                                    "text-grey-6 text-center q-mt-md"
                                )

        def _render_card(card: KanbanCard, current_status: str) -> None:
            """Render a single kanban card with badges and actions."""
            with ui.card().classes("w-full q-mb-sm cursor-pointer"):
                with ui.row().classes("items-center gap-2"):
                    ui.badge(
                        card.card_type,
                        color=type_colors.get(card.card_type, "grey"),
                    )
                    ui.badge(
                        card.priority,
                        color=priority_colors.get(card.priority, "grey"),
                    )
                ui.label(card.title).classes("font-bold")
                if card.description:
                    ui.label(card.description[:80]).classes(
                        "text-caption text-grey-5"
                    )
                if card.board_name:
                    ui.label(f"Board: {card.board_name}").classes(
                        "text-caption text-grey-6"
                    )

                with ui.row().classes("gap-2 q-mt-xs"):
                    other_statuses = [
                        s for s in VALID_STATUSES if s != current_status
                    ]
                    move_select = ui.select(
                        options=other_statuses,
                        label="Move to",
                    ).classes("w-32").props("dense")

                    def _on_move(e: object, cid: str = card.id) -> None:
                        val = getattr(e, "value", None)
                        if val:
                            move_card(project_root, cid, val)
                            _rebuild_board()

                    move_select.on_value_change(_on_move)

                    def _on_edit(_e: object, cid: str = card.id) -> None:
                        _show_edit_dialog(cid)

                    ui.button(icon="edit", on_click=_on_edit).props(
                        "flat dense round size=sm"
                    )

                    def _on_delete(_e: object, cid: str = card.id) -> None:
                        delete_card(project_root, cid)
                        _rebuild_board()

                    ui.button(icon="delete", on_click=_on_delete).props(
                        "flat dense round size=sm color=red"
                    )

        def _show_edit_dialog(card_id: str) -> None:
            """Show an edit dialog for a card."""
            kb = load_kanban(project_root)
            card = next((c for c in kb.cards if c.id == card_id), None)
            if card is None:
                return

            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label("Edit Card").classes("text-h6")
                title_input = ui.input("Title", value=card.title).classes(
                    "w-full"
                )
                desc_input = ui.textarea(
                    "Description", value=card.description
                ).classes("w-full")
                type_select = ui.select(
                    options=list(VALID_TYPES),
                    value=card.card_type,
                    label="Type",
                ).classes("w-full")
                priority_select = ui.select(
                    options=list(VALID_PRIORITIES),
                    value=card.priority,
                    label="Priority",
                ).classes("w-full")

                with ui.row().classes("justify-end gap-2 q-mt-md"):

                    def _save(_e: object) -> None:
                        updates: dict[str, object] = {
                            "title": title_input.value,
                            "description": desc_input.value,
                            "card_type": type_select.value,
                            "priority": priority_select.value,
                        }
                        update_card(project_root, card_id, updates)
                        dialog.close()
                        _rebuild_board()

                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button("Save", on_click=_save).props("color=primary")
            dialog.open()

        def _show_add_dialog() -> None:
            """Show a dialog to add a new card."""
            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label("Add Card").classes("text-h6")
                title_input = ui.input("Title").classes("w-full")
                desc_input = ui.textarea("Description").classes("w-full")
                type_select = ui.select(
                    options=list(VALID_TYPES),
                    value="feature",
                    label="Type",
                ).classes("w-full")
                priority_select = ui.select(
                    options=list(VALID_PRIORITIES),
                    value="P2",
                    label="Priority",
                ).classes("w-full")
                level_select = ui.select(
                    options=["framework", "deployment", "board"],
                    value=active_level["value"],
                    label="Level",
                ).classes("w-full")
                board_input = ui.input(
                    "Board name (if level=board)",
                    value=active_board_name["value"],
                ).classes("w-full")

                with ui.row().classes("justify-end gap-2 q-mt-md"):

                    def _create(_e: object) -> None:
                        if not title_input.value:
                            ui.notify("Title is required", type="warning")
                            return
                        card = KanbanCard(
                            title=title_input.value,
                            description=desc_input.value or "",
                            card_type=type_select.value or "feature",
                            priority=priority_select.value or "P2",
                            level=level_select.value or "framework",
                            board_name=board_input.value or "",
                        )
                        add_card(project_root, card)
                        dialog.close()
                        _rebuild_board()

                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button("Create", on_click=_create).props("color=primary")
            dialog.open()

        # ---- Header ----
        with ui.header().classes("items-center justify-between"):
            ui.label("Kanban Board").classes("text-h5 font-bold")
            ui.button("+ Add Card", on_click=_show_add_dialog).props(
                "color=primary"
            )

        # ---- Level tabs ----
        with ui.row().classes("w-full q-mb-md items-center gap-4"):
            with ui.tabs().classes("w-auto") as level_tabs:
                ui.tab("framework", label="Framework")
                ui.tab("deployment", label="Deployment")
                ui.tab("board", label="Board")

            boards = _discover_boards(output_root)
            board_selector = ui.select(
                options=boards,
                label="Select board",
            ).classes("w-48").bind_visibility_from(
                level_tabs, "value", backward=lambda v: v == "board"
            )

            def _on_board_select(e: object) -> None:
                active_board_name["value"] = getattr(e, "value", "") or ""
                _rebuild_board()

            board_selector.on_value_change(_on_board_select)

        def _on_level_change(e: object) -> None:
            active_level["value"] = (
                getattr(e, "value", "framework") or "framework"
            )
            _rebuild_board()

        level_tabs.on_value_change(_on_level_change)

        # ---- Type filters ----
        with ui.row().classes("q-mb-md gap-2"):
            ui.label("Filter:").classes("text-subtitle1")
            for card_type in VALID_TYPES:

                def _make_toggle(ct: str = card_type) -> None:
                    def _toggle(e: object) -> None:
                        val = getattr(e, "value", False)
                        if val:
                            active_type_filters.add(ct)
                        else:
                            active_type_filters.discard(ct)
                        _rebuild_board()

                    ui.checkbox(ct.title(), value=True, on_change=_toggle)

                _make_toggle()

        # ---- Board columns ----
        _rebuild_board()

    ui.run(port=port, title="KiCad Review Dashboard", reload=False)

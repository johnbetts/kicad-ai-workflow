"""NiceGUI review dashboard for the KiCad AI pipeline.

Multi-user, role-aware dashboard with:
- Shared navigation header with role selector
- Three-panel review page (Images | CLI/Chat | Context)
- Multi-level kanban board (Framework / Deployment / Board)
- Per-tab session state (multiple users/tabs supported)
- Cross-navigation between review and kanban pages
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Shared log buffer — API pushes here, log panel drains it.
_log_buffer: list[str] = []

# Shared notification buffer — API pushes here, panel timer drains it.
# Each entry: {"message": str, "type": str} where type is positive/negative/warning/info.
_notification_buffer: list[dict[str, str]] = []


def _discover_boards(output_root: Path) -> list[str]:
    """Return sorted board directory names from the output/ folder."""
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
    """Choose the best default board."""
    if not boards:
        return None
    for name in boards:
        if _board_has_evidence(output_root / name):
            return name
    return boards[0]


# ---------------------------------------------------------------------------
# Shared navigation header
# ---------------------------------------------------------------------------

ROLE_LABELS: dict[str, str] = {
    "framework": "Framework Developer",
    "deployment": "Deployment Admin",
    "board": "Board User",
}


def _build_nav_header(
    current_page: str,
    session: dict[str, object],
    on_role_change: object | None = None,
) -> None:
    """Shared navigation header with role selector.

    Args:
        current_page: "/" or "/kanban" — highlights the active page.
        session: Per-tab session dict for storing role state.
        on_role_change: Optional callback invoked after role changes.
    """
    from nicegui import ui

    with ui.header().classes("items-center justify-between gap-4"):
        # Left: title + nav links
        with ui.row().classes("items-center gap-6"):
            ui.label("KiCad AI Pipeline").classes("text-h5 font-bold")
            ui.separator().props("vertical")

            review_btn = ui.button(
                "Review",
                on_click=lambda: ui.navigate.to("/"),
            ).props("flat no-caps")
            if current_page == "/":
                review_btn.props("color=primary")

            kanban_btn = ui.button(
                "Kanban",
                on_click=lambda: ui.navigate.to("/kanban"),
            ).props("flat no-caps")
            if current_page == "/kanban":
                kanban_btn.props("color=primary")

            fleet_btn = ui.button(
                "Fleet",
                on_click=lambda: ui.navigate.to("/fleet"),
            ).props("flat no-caps")
            if current_page == "/fleet":
                fleet_btn.props("color=primary")

        # Right: reviewer name + role selector
        with ui.row().classes("items-center gap-4"):
            with ui.row().classes("items-center gap-2"):
                ui.label("Reviewer:").classes("text-subtitle2 text-grey-4")
                reviewer_input = (
                    ui.input(
                        value=str(session.get("reviewer_name", "")),
                        placeholder="Your name",
                    )
                    .classes("w-32")
                    .props("dense borderless")
                )

                def _on_reviewer_change(e: object) -> None:
                    val = getattr(e, "value", "")
                    session["reviewer_name"] = val

                reviewer_input.on("update:model-value", _on_reviewer_change)

            # Role selector
            with ui.row().classes("items-center gap-2"):
                ui.label("Role:").classes("text-subtitle2 text-grey-4")
                role_select = (
                    ui.select(
                        options=ROLE_LABELS,
                        value=session.get("role", "framework"),
                        label="",
                    )
                    .classes("w-52")
                    .props("dense borderless")
                )

                def _on_role_change_handler(e: object) -> None:
                    val = getattr(e, "value", None)
                    if val:
                        session["role"] = val
                        if callable(on_role_change):
                            on_role_change()

                role_select.on_value_change(_on_role_change_handler)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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
        build_board_summary,
        build_context_panel,
        build_image_panel,
        build_log_panel,
        start_notification_drain,
    )

    project_root = Path.cwd()
    output_root = Path(output_dir) if output_dir else project_root / "output"

    # Resolve initial board from CLI arg
    initial_board_dir: Path | None = None
    if board_path:
        bp = Path(board_path)
        if bp.is_dir():
            initial_board_dir = bp
        elif bp.is_file():
            initial_board_dir = bp.parent
        else:
            candidate = output_root / board_path
            if candidate.is_dir():
                initial_board_dir = candidate

    # Register FastAPI/Starlette API routes
    register_api_routes(nicegui_app, output_root)

    # Per-tab storage enabled via storage_secret in ui.run()

    # ------------------------------------------------------------------
    # Review page — three-panel layout
    # ------------------------------------------------------------------

    @ui.page("/")
    async def index(board: str = "") -> None:
        # Must await client connection before accessing storage.tab
        await ui.context.client.connected()
        # Per-tab session state (survives refresh, per-tab isolation)
        session = nicegui_app.storage.tab
        session.setdefault("role", "framework")
        session.setdefault("reviewer_name", "")
        session.setdefault("board_dir", "")

        board_names = _discover_boards(output_root)
        board_options: dict[str, str] = {}
        for name in board_names:
            grade = _latest_grade(output_root / name)
            label = f"{name} [{grade}]" if grade != "?" else name
            board_options[name] = label

        # Determine initial board: query param > session > CLI arg > auto
        if board and board in board_names:
            session["board_dir"] = str(output_root / board)
        elif not session.get("board_dir"):
            if initial_board_dir:
                session["board_dir"] = str(initial_board_dir)
            else:
                default_name = _pick_default_board(output_root, board_names)
                if default_name:
                    session["board_dir"] = str(output_root / default_name)

        current_board = Path(str(session["board_dir"])) if session.get("board_dir") else None

        ui.dark_mode(True)

        # _rebuild_panels is forward-declared; will be defined after containers
        # are created.  We pass a lambda so the header can call it once it exists.
        _rebuild_ref: dict[str, object] = {}

        def _rebuild_from_header() -> None:
            fn = _rebuild_ref.get("fn")
            if callable(fn):
                fn()

        _build_nav_header("/", session, on_role_change=_rebuild_from_header)

        # Board selector toolbar
        with ui.row().classes("w-full items-center gap-4 q-px-md q-pt-sm q-pb-xs"):
            ui.icon("developer_board").classes("text-h6")
            ui.label("Board:").classes("text-subtitle1")
            board_select = ui.select(
                options=board_options,
                value=current_board.name if current_board else None,
                label="",
            ).classes("w-64")

            # Cross-navigation: link to kanban for current board
            def _open_kanban_for_board() -> None:
                ui.navigate.to("/kanban")

            ui.button(
                "Board Issues",
                icon="view_kanban",
                on_click=_open_kanban_for_board,
            ).props("flat dense no-caps size=sm")

        # Board summary header (between selector and panels)
        summary_container = ui.element("div")
        with summary_container:
            build_board_summary(current_board)

        # Toast notifications for evidence events
        start_notification_drain()

        def _get_board_dir() -> Path | None:
            val = session.get("board_dir")
            return Path(str(val)) if val else None

        def _get_role() -> str:
            return str(session.get("role", "framework"))

        # Three-panel layout: Images (left) | Chat (center) | Context (right)
        with (
            ui.row()
            .classes("w-full gap-2")
            .style("height: calc(100vh - 200px); flex-wrap: nowrap")
        ):
            # LEFT PANEL — Images (25%)
            with (
                ui.card()
                .classes("h-full")
                .style("flex: 0 0 25%; max-width: 25%; overflow-y: auto; overflow-x: hidden")
            ):
                image_container = ui.element("div")
                with image_container:
                    build_image_panel(current_board)

            # CENTER PANEL — CLI / Chat (45%)
            with (
                ui.card()
                .classes("h-full")
                .style(
                    "flex: 0 0 45%; max-width: 45%; "
                    "display: flex; flex-direction: column; overflow: hidden"
                )
            ):
                log_container = ui.element("div").style(
                    "flex: 1 1 auto; display: flex; flex-direction: column; "
                    "min-height: 0; width: 100%"
                )
                with log_container:
                    build_log_panel(board_dir=current_board)

            # RIGHT PANEL — Context (30%)
            with (
                ui.card()
                .classes("h-full")
                .style("flex: 0 0 30%; max-width: 30%; overflow-y: auto")
            ):
                context_container = ui.element("div")
                with context_container:
                    build_context_panel(current_board, output_root, role=_get_role())

        def _rebuild_panels() -> None:
            bd = _get_board_dir()
            summary_container.clear()
            with summary_container:
                build_board_summary(bd)
            image_container.clear()
            with image_container:
                build_image_panel(bd)
            log_container.clear()
            with log_container:
                build_log_panel(board_dir=bd)
            context_container.clear()
            with context_container:
                build_context_panel(bd, output_root, role=_get_role())

        # Register so header role-change callback can reach it
        _rebuild_ref["fn"] = _rebuild_panels

        def _on_board_change(e: object) -> None:
            value = getattr(e, "value", None)
            if value:
                session["board_dir"] = str(output_root / value)
            _rebuild_panels()

        board_select.on_value_change(_on_board_change)

    # ------------------------------------------------------------------
    # Kanban page — multi-level board
    # ------------------------------------------------------------------

    @ui.page("/kanban")
    async def kanban_page() -> None:
        """Multi-level kanban board for tracking bugs, features, and releases."""
        await ui.context.client.connected()
        from kicad_pipeline.dashboard.kanban import (
            VALID_PRIORITIES,
            VALID_STATUSES,
            VALID_TYPES,
            KanbanCard,
            _relative_time,
            add_card,
            add_note,
            delete_card,
            load_kanban,
            move_card,
            update_card,
        )

        # Per-tab session state (survives refresh, per-tab isolation)
        session = nicegui_app.storage.tab
        session.setdefault("role", "framework")
        session.setdefault("reviewer_name", "")
        session.setdefault("board_dir", "")

        if "type_filters" not in session:
            session["type_filters"] = list(VALID_TYPES)
        if "kanban_board_name" not in session:
            session["kanban_board_name"] = ""
        active_board_name: dict[str, str] = {
            "value": str(session.get("kanban_board_name", "")),
        }
        active_type_filters: set[str] = set(
            session.get("type_filters", list(VALID_TYPES))  # type: ignore[arg-type]
        )

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

        ui.dark_mode(True)
        _build_nav_header("/kanban", session)

        board_container = ui.element("div")

        def _current_level() -> str:
            return str(session.get("role", "framework"))

        def _get_filtered_cards() -> list[KanbanCard]:
            kb = load_kanban(project_root)
            cards = kb.filter_by_level(_current_level())
            if _current_level() == "board" and active_board_name["value"]:
                cards = [c for c in cards if c.board_name == active_board_name["value"]]
            cards = [c for c in cards if c.card_type in active_type_filters]
            return kb.sorted_by_priority(cards)

        def _rebuild_board() -> None:
            board_container.clear()
            cards = _get_filtered_cards()
            with board_container, ui.row().classes("w-full gap-4"):
                for status in VALID_STATUSES:
                    col_cards = [c for c in cards if c.status == status]
                    with ui.card().classes("flex-1 min-w-64"):
                        ui.label(column_labels[status]).classes("text-h6 font-bold q-mb-sm")
                        ui.separator()
                        with ui.scroll_area().classes("h-96"):
                            for card in col_cards:
                                _render_card(card, status)
                        if not col_cards:
                            ui.label("No cards").classes("text-grey-6 text-center q-mt-md")

        def _render_card(card: KanbanCard, current_status: str) -> None:
            with ui.card().classes("w-full q-mb-sm"):
                with ui.row().classes("items-center gap-2"):
                    ui.badge(
                        card.card_type,
                        color=type_colors.get(card.card_type, "grey"),
                    )
                    ui.badge(
                        card.priority,
                        color=priority_colors.get(card.priority, "grey"),
                    )
                    if card.notes:
                        ui.badge(
                            f"{len(card.notes)} notes",
                            color="teal",
                        ).props("outline")
                ui.label(card.title).classes("font-bold")
                if card.description:
                    ui.label(card.description[:80]).classes("text-caption text-grey-5")
                # Relative timestamps
                updated_rel = _relative_time(card.updated)
                created_rel = _relative_time(card.created)
                time_label = (
                    f"Updated {updated_rel}"
                    if updated_rel != created_rel
                    else f"Created {created_rel}"
                )
                ui.label(time_label).classes("text-caption text-grey-7")

                if card.board_name:
                    with ui.row().classes("items-center gap-2"):
                        ui.label(f"Board: {card.board_name}").classes("text-caption text-grey-6")
                        if card.level == "board":
                            ui.button(
                                "Open Review",
                                on_click=lambda bn=card.board_name: (
                                    ui.navigate.to(f"/?board={bn}")
                                ),
                            ).props("flat dense no-caps size=xs color=accent")

                with ui.row().classes("gap-2 q-mt-xs"):
                    other_statuses = [s for s in VALID_STATUSES if s != current_status]
                    move_select = (
                        ui.select(
                            options=other_statuses,
                            label="Move to",
                        )
                        .classes("w-32")
                        .props("dense")
                    )

                    def _on_move(e: object, cid: str = card.id) -> None:
                        val = getattr(e, "value", None)
                        if val:
                            move_card(project_root, cid, val)
                            _rebuild_board()

                    move_select.on_value_change(_on_move)

                    def _on_edit(_e: object, cid: str = card.id) -> None:
                        _show_edit_dialog(cid)

                    ui.button(icon="edit", on_click=_on_edit).props("flat dense round size=sm")

                    def _on_delete(
                        _e: object,
                        cid: str = card.id,
                        title: str = card.title,
                    ) -> None:
                        with ui.dialog() as confirm_dlg, ui.card():
                            ui.label(f"Delete '{title[:50]}'?").classes("text-subtitle1")
                            with ui.row().classes("justify-end gap-2 q-mt-md"):
                                ui.button("Cancel", on_click=confirm_dlg.close).props("flat")

                                def _confirm(_e2: object, _cid: str = cid) -> None:
                                    delete_card(project_root, _cid)
                                    confirm_dlg.close()
                                    _rebuild_board()

                                ui.button("Delete", on_click=_confirm).props("color=red")
                        confirm_dlg.open()

                    ui.button(icon="delete", on_click=_on_delete).props(
                        "flat dense round size=sm color=red"
                    )

        def _show_edit_dialog(card_id: str) -> None:
            kb = load_kanban(project_root)
            card = next((c for c in kb.cards if c.id == card_id), None)
            if card is None:
                return

            with ui.dialog() as dialog, ui.card().classes("w-96"):
                ui.label("Edit Card").classes("text-h6")
                title_input = ui.input("Title", value=card.title).classes("w-full")
                desc_input = ui.textarea("Description", value=card.description).classes("w-full")
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

                # Full timestamps
                ui.separator().classes("q-my-sm")
                ui.label("Timestamps").classes("text-subtitle2 font-bold")
                ui.label(f"Created: {card.created}").classes("text-caption text-grey-5")
                ui.label(f"Updated: {card.updated}").classes("text-caption text-grey-5")

                # Notes section
                ui.separator().classes("q-my-sm")
                ui.label("Notes").classes("text-subtitle2 font-bold")
                if card.notes:
                    with ui.scroll_area().classes("w-full").style("max-height: 150px"):
                        for note in card.notes:
                            ui.label(note).classes("text-caption text-grey-4 q-mb-xs")
                else:
                    ui.label("No notes yet").classes("text-caption text-grey-6")

                with ui.row().classes("w-full items-center gap-2"):
                    note_input = ui.input(placeholder="Add a note...").classes("flex-grow")

                    def _add_note(_e: object) -> None:
                        if note_input.value:
                            add_note(project_root, card_id, note_input.value)
                            dialog.close()
                            _show_edit_dialog(card_id)

                    ui.button("Add Note", on_click=_add_note).props("dense size=sm")

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
                    value=_current_level(),
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
                        new_card = KanbanCard(
                            title=title_input.value,
                            description=desc_input.value or "",
                            card_type=type_select.value or "feature",
                            priority=priority_select.value or "P2",
                            level=level_select.value or "framework",
                            board_name=board_input.value or "",
                        )
                        add_card(project_root, new_card)
                        dialog.close()
                        _rebuild_board()

                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button("Create", on_click=_create).props("color=primary")
            dialog.open()

        # Toolbar: Add button + level tabs + board selector + type filters
        with ui.row().classes("w-full q-mb-md items-center gap-4 q-px-md q-pt-sm"):
            ui.button("+ Add Card", on_click=_show_add_dialog).props("color=primary")

            with ui.tabs().classes("w-auto") as level_tabs:
                ui.tab("framework", label="Framework")
                ui.tab("deployment", label="Deployment")
                ui.tab("board", label="Board")

            # Sync tabs with role selector
            level_tabs.value = _current_level()

            boards = _discover_boards(output_root)
            board_selector = (
                ui.select(
                    options=boards,
                    label="Select board",
                )
                .classes("w-48")
                .bind_visibility_from(level_tabs, "value", backward=lambda v: v == "board")
            )

            def _on_board_select(e: object) -> None:
                val = getattr(e, "value", "") or ""
                active_board_name["value"] = val
                session["kanban_board_name"] = val
                _rebuild_board()

            board_selector.on_value_change(_on_board_select)

        def _on_level_change(e: object) -> None:
            val = getattr(e, "value", "framework") or "framework"
            session["role"] = val
            _rebuild_board()

        level_tabs.on_value_change(_on_level_change)

        # Type filters
        with ui.row().classes("q-mb-md gap-2 q-px-md"):
            ui.label("Filter:").classes("text-subtitle1")
            for card_type in VALID_TYPES:

                def _make_toggle(ct: str = card_type) -> None:
                    def _toggle(e: object) -> None:
                        val = getattr(e, "value", False)
                        if val:
                            active_type_filters.add(ct)
                        else:
                            active_type_filters.discard(ct)
                        session["type_filters"] = list(active_type_filters)
                        _rebuild_board()

                    ui.checkbox(
                        ct.title(),
                        value=ct in active_type_filters,
                        on_change=_toggle,
                    )

                _make_toggle()

        # Stats bar
        stats_container = ui.element("div")

        def _rebuild_stats() -> None:
            stats_container.clear()
            kb = load_kanban(project_root)
            level_cards = kb.filter_by_level(_current_level())
            with stats_container, ui.row().classes("w-full q-px-md q-mb-sm items-center gap-3"):
                # Cards per type
                for ct in VALID_TYPES:
                    count = sum(1 for c in level_cards if c.card_type == ct)
                    if count > 0:
                        ui.badge(
                            f"{ct}: {count}",
                            color=type_colors.get(ct, "grey"),
                        )

                ui.separator().props("vertical").classes("q-mx-sm")

                # Open vs done
                open_count = sum(1 for c in level_cards if c.status != "done")
                done_count = sum(1 for c in level_cards if c.status == "done")
                ui.label(f"Open: {open_count} | Done: {done_count}").classes("text-caption")

                # P0/P1 warnings
                p0_count = sum(1 for c in level_cards if c.priority == "P0" and c.status != "done")
                p1_count = sum(1 for c in level_cards if c.priority == "P1" and c.status != "done")
                if p0_count > 0:
                    ui.badge(f"P0: {p0_count}", color="red").props("outline")
                if p1_count > 0:
                    ui.badge(f"P1: {p1_count}", color="orange").props("outline")

        # Wrap _rebuild_board to also rebuild stats
        _original_rebuild = _rebuild_board

        def _rebuild_board_with_stats() -> None:
            _rebuild_stats()
            _original_rebuild()

        _rebuild_board = _rebuild_board_with_stats

        # Kanban columns
        _rebuild_board()

    # ------------------------------------------------------------------
    # Fleet page — all boards in a single table with gate status
    # ------------------------------------------------------------------

    @ui.page("/fleet")
    async def fleet_page() -> None:
        """Fleet overview showing all boards and their gate status."""
        from datetime import datetime, timezone

        from kicad_pipeline.evidence.gates import ALL_STAGES, check_gate
        from kicad_pipeline.evidence.ledger import load_ledger

        await ui.context.client.connected()
        session = nicegui_app.storage.tab
        session.setdefault("role", "framework")
        session.setdefault("reviewer_name", "")

        ui.dark_mode(True)
        _build_nav_header("/fleet", session)

        ui.label("Fleet Overview").classes("text-h5 font-bold q-px-md q-pt-md")

        board_names = _discover_boards(output_root)

        if not board_names:
            ui.label("No boards found in output directory").classes("text-grey q-px-md")
            return

        fleet_container = ui.element("div")

        def _build_fleet_rows() -> list[dict[str, str]]:
            rows: list[dict[str, str]] = []
            for name in board_names:
                board_dir = output_root / name
                grade = _latest_grade(board_dir)

                board_pcb_files = list(board_dir.glob("*.kicad_pcb"))
                if not board_pcb_files:
                    continue
                board_pcb = board_pcb_files[0]

                stage_status: dict[str, str] = {}
                blocker_parts: list[str] = []
                for stage in ALL_STAGES:
                    result = check_gate(board_pcb, stage)
                    if result.passed:
                        stage_status[stage] = "pass"
                    else:
                        stage_status[stage] = "fail"
                        if result.missing:
                            for m in result.missing:
                                if "human_approval" in m:
                                    blocker_parts.append(f"{stage}: needs approval")
                                    break
                            else:
                                blocker_parts.append(f"{stage}: {len(result.missing)} missing")

                ledger = load_ledger(board_pcb)
                activity = "—"
                if ledger.records:
                    now = datetime.now(tz=timezone.utc)
                    delta = now - ledger.records[-1].timestamp
                    secs = delta.total_seconds()
                    if secs < 3600:
                        activity = f"{int(secs // 60)}m ago"
                    elif secs < 86400:
                        activity = f"{int(secs // 3600)}h ago"
                    else:
                        activity = f"{int(secs // 86400)}d ago"

                blocker = blocker_parts[0] if blocker_parts else "—"

                rows.append(
                    {
                        "board": name,
                        "grade": grade,
                        "requirements": stage_status.get("requirements", "?"),
                        "schematic": stage_status.get("schematic", "?"),
                        "pcb": stage_status.get("pcb", "?"),
                        "validation": stage_status.get("validation", "?"),
                        "production": stage_status.get("production", "?"),
                        "blocker": blocker,
                        "activity": activity,
                    }
                )
            return rows

        def _render_fleet_table(columns: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
            table = ui.table(
                columns=columns,
                rows=rows,
                row_key="board",
            ).classes("w-full q-mx-md")

            table.on(
                "row-click",
                lambda e: ui.navigate.to(f"/?board={e.args[1]['board']}"),
            )

            with ui.row().classes("q-px-md q-mt-md gap-4"):
                total = len(rows)
                ready = sum(1 for r in rows if r["production"] == "pass")
                blocked = sum(1 for r in rows if "approval" in r.get("blocker", ""))
                ui.badge(f"Total: {total}", color="blue")
                ui.badge(f"Ready: {ready}", color="green")
                if blocked:
                    ui.badge(f"Awaiting approval: {blocked}", color="orange")

        def _rebuild_fleet() -> None:
            columns = [
                {"name": "board", "label": "Board", "field": "board", "sortable": True},
                {"name": "grade", "label": "Grade", "field": "grade", "sortable": True},
                {"name": "requirements", "label": "Req", "field": "requirements"},
                {"name": "schematic", "label": "Sch", "field": "schematic"},
                {"name": "pcb", "label": "PCB", "field": "pcb"},
                {"name": "validation", "label": "Val", "field": "validation"},
                {"name": "production", "label": "Prod", "field": "production"},
                {"name": "blocker", "label": "Blocker", "field": "blocker"},
                {
                    "name": "activity",
                    "label": "Last Activity",
                    "field": "activity",
                    "sortable": True,
                },
            ]
            fleet_container.clear()
            with fleet_container:
                rows = _build_fleet_rows()
                _render_fleet_table(columns, rows)

        _rebuild_fleet()
        ui.timer(5.0, _rebuild_fleet)

    ui.run(
        port=port,
        title="KiCad Review Dashboard",
        reload=True,
        storage_secret="kicad-dashboard",
    )

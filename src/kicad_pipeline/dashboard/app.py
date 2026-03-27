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

        boards = _discover_boards(output_root)

        ui.dark_mode(True)
        with ui.header().classes("items-center justify-between"):
            ui.label("KiCad Review Dashboard").classes("text-h5 font-bold")
            board_select = ui.select(
                options=boards,
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

    ui.run(port=port, title="KiCad Review Dashboard", reload=False)

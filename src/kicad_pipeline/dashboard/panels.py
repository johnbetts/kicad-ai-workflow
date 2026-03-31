"""Panel builders for the NiceGUI review dashboard."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.evidence.models import Issue, ScoreSnapshot

logger = logging.getLogger(__name__)

# Board-level image name prefixes (shown first, in this order).
_BOARD_IMAGE_ORDER = (
    "2d_top",
    "3d_top",
    "3d_iso",
    "3d_isoback",
    "3d_hires_top",
    "3d_padoverlay",
)

# Track which static directories have been registered to avoid collisions.
_registered_static_routes: dict[str, str] = {}

# Module-level callback for cross-panel image navigation (Feature 6).
_image_nav_callback: dict[str, object] = {}


def _sorted_images(board_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (board_images, crop_images) sorted for display."""
    all_pngs = sorted(board_dir.glob("*.png"))
    board_images: list[Path] = []
    other_images: list[Path] = []

    name_to_path = {p.stem: p for p in all_pngs}
    for prefix in _BOARD_IMAGE_ORDER:
        if prefix in name_to_path:
            board_images.append(name_to_path[prefix])
    board_stems = {p.stem for p in board_images}
    for p in all_pngs:
        if p.stem not in board_stems:
            other_images.append(p)

    crops_dir = board_dir / "crops"
    crop_images: list[Path] = []
    if crops_dir.is_dir():
        crop_images = sorted(crops_dir.glob("*.png"))

    return board_images + other_images, crop_images


def _static_route_for(directory: Path) -> str:
    """Register a directory as a static file source and return its URL prefix.

    Each unique directory gets its own route to avoid collisions between
    board images and crop images from different boards.
    """
    from nicegui import app as nicegui_app

    dir_str = str(directory.resolve())
    if dir_str in _registered_static_routes:
        return _registered_static_routes[dir_str]

    # Create a unique short hash for this directory
    dir_hash = hashlib.md5(dir_str.encode()).hexdigest()[:8]
    route = f"/static/{dir_hash}"
    nicegui_app.add_static_files(route, dir_str)
    _registered_static_routes[dir_str] = route
    return route


def _image_url(img_path: Path) -> str:
    """Return the static URL for an image file."""
    route = _static_route_for(img_path.parent)
    return f"{route}/{img_path.name}"


# ---------------------------------------------------------------------------
# Image panel (left)
# ---------------------------------------------------------------------------


def build_image_panel(board_dir: Path | None) -> None:
    """Build the image gallery panel (left side).

    Thumbnails at top; clicking shows the image inline below (no popup).
    """
    from nicegui import ui

    if board_dir is None or not board_dir.is_dir():
        ui.label("No board selected").classes("text-grey")
        return

    # Inline viewer — selected image shown here, not in a popup
    selected_viewer = ui.column().classes("w-full")
    image_column = ui.column().classes("w-full gap-1")
    _selected_path: dict[str, Path | None] = {"current": None}

    def _show_inline(img_path: Path) -> None:
        _selected_path["current"] = img_path
        selected_viewer.clear()
        url = _image_url(img_path)
        with selected_viewer:
            ui.label(img_path.stem).classes("text-subtitle2 font-bold")
            img = ui.image(url).style("width: 100%; max-width: 100%; cursor: grab")
            # Feature 5: Add zoom/pan via CSS transform
            img_id = f"img_{id(img)}"
            img.props(f'id="{img_id}"')
            ui.run_javascript(f"""
                (function() {{
                    const el = document.getElementById("{img_id}");
                    if (!el) return;
                    let scale = 1, panX = 0, panY = 0;
                    let isDragging = false, startX, startY;
                    el.addEventListener("wheel", function(e) {{
                        e.preventDefault();
                        const delta = e.deltaY > 0 ? 0.9 : 1.1;
                        scale = Math.max(0.5, Math.min(10, scale * delta));
                        el.style.transform =
                            "scale(" + scale + ") translate(" + panX + "px, " + panY + "px)";
                    }});
                    el.addEventListener("mousedown", function(e) {{
                        isDragging = true;
                        startX = e.clientX - panX;
                        startY = e.clientY - panY;
                        el.style.cursor = "grabbing";
                    }});
                    document.addEventListener("mousemove", function(e) {{
                        if (!isDragging) return;
                        panX = e.clientX - startX;
                        panY = e.clientY - startY;
                        el.style.transform =
                            "scale(" + scale + ") translate(" + panX + "px, " + panY + "px)";
                    }});
                    document.addEventListener("mouseup", function() {{
                        isDragging = false;
                        el.style.cursor = "grab";
                    }});
                    el.addEventListener("dblclick", function() {{
                        scale = 1; panX = 0; panY = 0;
                        el.style.transform = "";
                    }});
                }})();
            """)

    # Feature 6: Register crop navigation callback
    def _show_crop_by_ref(ref: str) -> None:
        """Navigate to a component's crop image by ref designator."""
        crops_dir = board_dir / "crops"
        if not crops_dir.is_dir():
            return
        for crop in sorted(crops_dir.glob("*.png")):
            if crop.stem.startswith(ref + "_") or crop.stem == ref:
                _show_inline(crop)
                return

    _image_nav_callback["show_crop"] = _show_crop_by_ref

    def _refresh_images() -> None:
        image_column.clear()
        board_images, crop_images = _sorted_images(board_dir)

        with image_column:
            if not board_images and not crop_images:
                ui.label("No images found").classes("text-grey")
                return

            if board_images:
                ui.label("Board Views").classes("text-caption font-bold")
                with ui.row().classes("flex-wrap gap-1"):
                    for img_path in board_images:
                        _make_thumbnail(img_path)

            # Feature 12: Crop search/filter
            if crop_images:
                with ui.expansion(
                    f"Crops ({len(crop_images)})",
                    icon="grid_view",
                ).classes("w-full"):
                    crop_filter = (
                        ui.input(placeholder="Filter by ref (e.g., U1, C3)...")
                        .classes("w-full q-mb-sm")
                        .props("dense clearable")
                    )
                    crop_container = ui.row().classes("flex-wrap gap-1")

                    def _refresh_crops() -> None:
                        crop_container.clear()
                        filter_val = (crop_filter.value or "").upper()
                        with crop_container:
                            for img_path in crop_images:
                                if filter_val and not img_path.stem.upper().startswith(filter_val):
                                    continue
                                with ui.column().classes("items-center"):
                                    _make_thumbnail(img_path)
                                    ui.label(img_path.stem.split("_")[0]).classes(
                                        "text-caption text-center"
                                    )

                    _refresh_crops()
                    crop_filter.on("update:model-value", lambda _: _refresh_crops())

        # Auto-select first board image only on first load or if selection removed
        if board_images:
            current = _selected_path["current"]
            all_paths = board_images + crop_images
            if current is None or current not in all_paths:
                _show_inline(board_images[0])

    def _make_thumbnail(img_path: Path) -> None:
        from nicegui import ui

        url = _image_url(img_path)
        ui.image(url).classes("cursor-pointer").style(
            "width: 100px; height: 75px; object-fit: cover; "
            "border-radius: 4px; border: 1px solid #555"
        ).on("click", lambda _e=None, p=img_path: _show_inline(p)).tooltip(img_path.stem)

    _refresh_images()
    ui.timer(5.0, _refresh_images)

    # Compare iterations button
    ui.button(
        "Compare Iterations",
        on_click=lambda: _open_diff_dialog(board_dir),
        icon="compare",
    ).classes("q-mt-md")


# ---------------------------------------------------------------------------
# Cross-panel navigation helper (Feature 6)
# ---------------------------------------------------------------------------


def _navigate_to_crop(ref: str) -> None:
    """Navigate image panel to the crop for a given ref designator."""
    callback = _image_nav_callback.get("show_crop")
    if callable(callback):
        callback(ref)


# ---------------------------------------------------------------------------
# Log panel (center)
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    """Return a formatted timestamp string for log messages."""
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


def _dispatch_command(
    command: str,
    board_dir: Path | None,
    log_widget: object,
) -> str:
    """Dispatch a slash command and return the response text.

    Args:
        command: The raw command string (with leading ``/``).
        board_dir: Path to the active board directory (may be ``None``).
        log_widget: The ``ui.log`` widget to push messages to.

    Returns:
        A human-readable response string.
    """
    from kicad_pipeline.evidence.gates import ALL_STAGES, check_gate
    from kicad_pipeline.evidence.ledger import append_record, load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind, EvidenceRecord

    parts = command.strip().split(maxsplit=2)
    cmd = parts[0].lower()
    arg1 = parts[1] if len(parts) > 1 else ""
    arg2 = parts[2] if len(parts) > 2 else ""

    if cmd == "/help":
        return (
            "Available commands:\n"
            "  /approve [stage]          - Approve the current stage\n"
            "  /reject [stage] [feedback] - Reject with feedback\n"
            "  /status                   - Show stage status summary\n"
            "  /score                    - Show latest quality score\n"
            "  /boards                   - List available boards\n"
            "  /board <name>             - Switch to a different board\n"
            "  /help                     - Show this help"
        )

    if cmd == "/boards":
        from kicad_pipeline.dashboard.api import _discover_board_names

        names = _discover_board_names()
        if not names:
            return "No boards found."
        return "Boards: " + ", ".join(names)

    if cmd == "/board":
        # Board switching is handled by the caller; just acknowledge here.
        if not arg1:
            return "Usage: /board <name>"
        return f"SWITCH_BOARD:{arg1}"

    # Commands below require an active board_dir.
    if board_dir is None or not board_dir.is_dir():
        return "No board selected. Use /board <name> first."

    board_pcb = _find_board_pcb(board_dir)
    board_name = board_pcb.stem

    if cmd == "/status":
        lines: list[str] = [f"Stage status for {board_name}:"]
        for stage in ALL_STAGES:
            result = check_gate(board_pcb, stage)
            if result.passed:
                status = "PASS"
            elif result.missing:
                status = f"FAIL (missing: {', '.join(result.missing)})"
            else:
                status = "pending"
            lines.append(f"  {stage:14s}  {status}")
        return "\n".join(lines)

    if cmd == "/score":
        ledger = load_ledger(board_pcb)
        score = ledger.latest_score()
        if score is None:
            return "No score data yet."
        breakdown_lines = [f"  {dim}: {val:.3f}" for dim, val in sorted(score.breakdown.items())]
        return (
            f"Quality score for {board_name}: "
            f"{score.grade} ({score.overall_score:.3f})\n" + "\n".join(breakdown_lines)
        )

    if cmd == "/approve":
        stage = arg1 or "pcb"
        if stage not in ALL_STAGES:
            return f"Unknown stage '{stage}'. Valid: {', '.join(ALL_STAGES)}"
        record = EvidenceRecord(
            kind=EvidenceKind.HUMAN_APPROVAL,
            stage=stage,
            step="cli_approval",
            board=board_name,
            passed=True,
            summary=f"Human approved {stage} via CLI",
            producer="human",
        )
        append_record(board_pcb, record)
        return f"Approved stage '{stage}'."

    if cmd == "/reject":
        stage = arg1 or "pcb"
        feedback = arg2 or ""
        if stage not in ALL_STAGES:
            return f"Unknown stage '{stage}'. Valid: {', '.join(ALL_STAGES)}"
        if not feedback:
            return "Usage: /reject <stage> <feedback>"
        record = EvidenceRecord(
            kind=EvidenceKind.HUMAN_REJECTION,
            stage=stage,
            step="cli_rejection",
            board=board_name,
            passed=False,
            summary=f"Human rejected {stage} via CLI",
            feedback=feedback,
            producer="human",
        )
        append_record(board_pcb, record)
        return f"Rejected stage '{stage}' with feedback."

    return f"Unknown command: {cmd}. Type /help for available commands."


def build_log_panel(board_dir: Path | None = None) -> None:
    """Build the chat/CLI panel (center).

    Chat-style interface with scrollable message history, command input
    with Enter-to-submit, and up-arrow command history. Drains the shared
    log buffer for agent messages.

    Args:
        board_dir: Path to the active board directory.  Used by commands
            such as ``/status`` and ``/score`` to read board evidence.
    """
    from nicegui import ui

    from kicad_pipeline.dashboard.app import _log_buffer

    # Command history for up-arrow recall
    _cmd_history: list[str] = []
    _history_idx: dict[str, int] = {"pos": -1}

    # Main flex column filling the card
    with (
        ui.column()
        .classes("w-full h-full")
        .style("display: flex; flex-direction: column; min-height: 0")
    ):
        # Scrollable message area
        with (
            ui.scroll_area().classes("w-full").style("flex: 1 1 auto; min-height: 0")
        ) as chat_scroll:
            chat_column = ui.column().classes("w-full gap-1 q-pa-sm")

        def _add_message(
            text: str,
            sender: str = "system",
        ) -> None:
            """Add a styled message to the chat."""
            with chat_column:
                if sender == "user":
                    with ui.row().classes("w-full justify-end"):
                        ui.chat_message(
                            text=text,
                            name="you",
                            sent=True,
                        ).classes("max-w-3/4")
                elif sender == "agent":
                    with ui.row().classes("w-full"):
                        ui.chat_message(
                            text=text,
                            name="agent",
                            sent=False,
                        ).classes("max-w-3/4")
                else:
                    # System messages: monospace style
                    with ui.row().classes("w-full"):
                        ui.chat_message(
                            text=text,
                            name="system",
                            sent=False,
                        ).classes("max-w-full").style("font-family: monospace; font-size: 0.85em")
            # Auto-scroll to bottom
            chat_scroll.scroll_to(percent=1.0)

        def _push_buffered() -> None:
            while _log_buffer:
                entry = _log_buffer.pop(0)
                _add_message(entry, sender="agent")

        ui.timer(1.0, _push_buffered)

        # Input area pinned to bottom
        with (
            ui.row()
            .classes("w-full items-center gap-2 q-pt-sm")
            .style("flex: 0 0 auto; border-top: 1px solid #444")
        ):
            cmd_input = (
                ui.input(placeholder="Type a /command or message...")
                .classes("flex-grow")
                .props('dense outlined autofocus color="blue-4"')
            )

            def _send_cmd() -> None:
                text = cmd_input.value.strip()
                if not text:
                    return

                # Save to history
                _cmd_history.append(text)
                _history_idx["pos"] = -1
                cmd_input.value = ""

                if text.startswith("/"):
                    _add_message(text, sender="user")
                    response = _dispatch_command(text, board_dir, None)

                    # Handle board-switch sentinel.
                    if response.startswith("SWITCH_BOARD:"):
                        target = response.split(":", 1)[1]
                        _add_message(
                            f"Switch to board '{target}' using the board selector dropdown.",
                            sender="system",
                        )
                    else:
                        _add_message(response, sender="system")
                else:
                    _add_message(text, sender="user")

            def _on_keydown(e: object) -> None:
                key = getattr(getattr(e, "args", {}), "key", "")
                if not key:
                    args = getattr(e, "args", {})
                    if isinstance(args, dict):
                        key = args.get("key", "")
                if key == "ArrowUp" and _cmd_history:
                    if _history_idx["pos"] < 0:
                        _history_idx["pos"] = len(_cmd_history) - 1
                    elif _history_idx["pos"] > 0:
                        _history_idx["pos"] -= 1
                    cmd_input.value = _cmd_history[_history_idx["pos"]]
                elif key == "ArrowDown" and _cmd_history:
                    if _history_idx["pos"] >= 0:
                        _history_idx["pos"] += 1
                        if _history_idx["pos"] >= len(_cmd_history):
                            _history_idx["pos"] = -1
                            cmd_input.value = ""
                        else:
                            cmd_input.value = _cmd_history[_history_idx["pos"]]

            cmd_input.on("keydown.enter", lambda _: _send_cmd())
            cmd_input.on(
                "keydown",
                _on_keydown,
                ["key"],
            )
            ui.button(icon="send", on_click=_send_cmd).props("flat dense round color=blue-4")

        # Welcome message
        _add_message(
            "Type /help for available commands. "
            "Use /status to check pipeline gates, "
            "/approve to approve stages.",
            sender="system",
        )


# ---------------------------------------------------------------------------
# Toast notification drain
# ---------------------------------------------------------------------------


def start_notification_drain() -> None:
    """Start a 2-second timer that drains the notification buffer as toasts.

    Call once per page to receive evidence-event notifications.
    """
    from nicegui import ui

    from kicad_pipeline.dashboard.app import _notification_buffer

    def _drain() -> None:
        while _notification_buffer:
            entry = _notification_buffer.pop(0)
            ui.notify(
                entry["message"],
                type=entry.get("type", "info"),
                position="top-right",
                close_button=True,
            )

    ui.timer(2.0, _drain)


# ---------------------------------------------------------------------------
# Board summary header
# ---------------------------------------------------------------------------


def build_board_summary(board_dir: Path | None) -> None:
    """Render a summary row: board name, grade, evidence count, stage, last activity.

    Place between the board selector and the three-panel layout.
    Auto-refreshes via a 3-second timer.
    """
    from datetime import datetime, timezone

    from nicegui import ui

    if board_dir is None or not board_dir.is_dir():
        return

    summary_row = (
        ui.row()
        .classes("w-full items-center gap-6 q-px-md q-py-xs")
        .style("background: rgba(255,255,255,0.03); border-radius: 4px")
    )

    def _refresh_summary() -> None:
        from kicad_pipeline.evidence.ledger import load_ledger

        board_pcb = _find_board_pcb(board_dir)
        ledger = load_ledger(board_pcb)

        # Latest grade
        score = ledger.latest_score()
        grade = score.grade if score else "?"
        grade_colors = {
            "A": "green",
            "B": "light-green",
            "C": "yellow",
            "D": "orange",
            "F": "red",
        }
        color = grade_colors.get(grade, "grey")

        # Latest stage from most recent record
        latest_stage = "---"
        if ledger.records:
            latest_stage = ledger.records[-1].stage or "---"

        # Time since last activity
        time_ago = "---"
        if ledger.records:
            now = datetime.now(tz=timezone.utc)
            delta = now - ledger.records[-1].timestamp
            secs = delta.total_seconds()
            if secs < 60:
                time_ago = "just now"
            elif secs < 3600:
                time_ago = f"{int(secs // 60)}m ago"
            elif secs < 86400:
                time_ago = f"{int(secs // 3600)}h ago"
            else:
                time_ago = f"{int(secs // 86400)}d ago"

        evidence_count = len(ledger.records)

        summary_row.clear()
        with summary_row:
            # Board name
            with ui.column().classes("items-center"):
                ui.label("Board").classes("text-caption text-grey")
                ui.label(board_dir.name).classes("text-subtitle1 font-bold")

            ui.separator().props("vertical").classes("h-10")

            # Grade (large colored letter)
            with ui.column().classes("items-center"):
                ui.label("Grade").classes("text-caption text-grey")
                ui.label(grade).classes(f"text-h4 font-bold text-{color}")

            ui.separator().props("vertical").classes("h-10")

            # Evidence count
            with ui.column().classes("items-center"):
                ui.label("Evidence").classes("text-caption text-grey")
                ui.label(str(evidence_count)).classes("text-h6 font-bold")

            ui.separator().props("vertical").classes("h-10")

            # Latest stage
            with ui.column().classes("items-center"):
                ui.label("Latest Stage").classes("text-caption text-grey")
                ui.label(latest_stage.capitalize()).classes("text-subtitle1 font-medium")

            ui.separator().props("vertical").classes("h-10")

            # Last activity
            with ui.column().classes("items-center"):
                ui.label("Last Activity").classes("text-caption text-grey")
                ui.label(time_ago).classes("text-subtitle1 font-medium")

    _refresh_summary()
    ui.timer(3.0, _refresh_summary)


# ---------------------------------------------------------------------------
# Context panel (right) -- role-aware
# ---------------------------------------------------------------------------


def build_context_panel(
    board_dir: Path | None,
    output_root: Path,
    role: str = "framework",
) -> None:
    """Build the context/status panel (right side).

    Content adapts based on active role:
    - framework: all cards including known issues and regression details
    - deployment: stage status, quality score, requirements
    - board: simplified view -- score, findings, actions
    """
    from nicegui import ui

    if board_dir is None or not board_dir.is_dir():
        ui.label("No board selected").classes("text-grey")
        return

    context_column = ui.column().classes("w-full gap-2")
    # Track previous state to avoid full rebuild flicker
    _prev_hash: dict[str, str] = {"value": ""}

    def _compute_state_hash() -> str:
        """Quick hash of ledger state to detect changes."""
        from kicad_pipeline.evidence.ledger import load_ledger

        board_pcb = _find_board_pcb(board_dir)
        ledger = load_ledger(board_pcb)
        return f"{len(ledger.records)}:{ledger.records[-1].id if ledger.records else ''}"

    def _refresh_context() -> None:
        current_hash = _compute_state_hash()
        if current_hash == _prev_hash["value"]:
            return  # No changes -- skip rebuild to avoid flicker
        _prev_hash["value"] = current_hash

        context_column.clear()
        with context_column:
            # Feature 1: "Waiting for You" banner
            _render_waiting_banner(board_dir)

            # Feature 11: Rejection feedback loop
            _render_rejection_banner(board_dir)

            # Primary action -- always first for quick access
            _build_actions_card(board_dir)

            # Common cards for all roles
            _build_stage_status_card(board_dir)
            _build_activity_feed_card(board_dir)
            _build_quality_score_card(board_dir)

            # Role-specific cards
            if role in ("framework", "deployment"):
                _build_score_trend_card(board_dir)
                _build_requirements_status_card(board_dir)

            _build_review_findings_card(board_dir)

            # DRC report -- full detail for framework/deployment, badge for board
            _build_drc_report_card(board_dir, role=role)

            # Feature 13: Manufacturing readiness checklist
            if role in ("framework", "deployment"):
                _build_manufacturing_readiness_card(board_dir, output_root)

            if role == "framework":
                _build_known_issues_card(board_dir)

    _refresh_context()
    ui.timer(3.0, _refresh_context)


def _find_board_pcb(board_dir: Path) -> Path:
    """Find the .kicad_pcb file in a board directory."""
    pcbs = list(board_dir.glob("*.kicad_pcb"))
    if pcbs:
        return pcbs[0]
    return board_dir / f"{board_dir.name}.kicad_pcb"


# ---------------------------------------------------------------------------
# Feature 1: "Waiting for You" banner
# ---------------------------------------------------------------------------


def _render_waiting_banner(board_dir: Path) -> None:
    """Show an amber banner if the pipeline is blocked waiting for human approval."""
    from nicegui import ui

    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)

    gate_records = [r for r in ledger.records if r.kind == EvidenceKind.GATE_RESULT]
    if not gate_records:
        return

    latest_gate = gate_records[-1]
    if latest_gate.passed is False:
        missing = latest_gate.details.get("missing", []) if latest_gate.details else []
        if isinstance(missing, list) and "human_approval" in missing:
            stage = latest_gate.stage or "unknown"
            with (
                ui.card().classes("w-full").style("background: #f57c00; color: white"),
                ui.row().classes("items-center gap-2"),
            ):
                ui.icon("hourglass_top").classes("text-h5")
                ui.label(f"Process blocked -- waiting for your approval on {stage}").classes(
                    "font-bold"
                )


# ---------------------------------------------------------------------------
# Feature 11: Rejection feedback loop
# ---------------------------------------------------------------------------


def _render_rejection_banner(board_dir: Path) -> None:
    """Show a red banner if the most recent action is a rejection without a later approval."""
    from nicegui import ui

    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)

    rejection_records = [r for r in ledger.records if r.kind == EvidenceKind.HUMAN_REJECTION]
    if not rejection_records:
        return

    latest_rejection = rejection_records[-1]
    # Check if there's a newer approval that supersedes it
    later_approvals = [
        r
        for r in ledger.records
        if r.kind == EvidenceKind.HUMAN_APPROVAL
        and r.stage == latest_rejection.stage
        and r.timestamp > latest_rejection.timestamp
    ]
    if not later_approvals:
        with ui.card().classes("w-full").style("background: #c62828; color: white"):
            ui.label(f"Rejected: {latest_rejection.stage}").classes("font-bold")
            ui.label(latest_rejection.feedback or latest_rejection.summary).classes("text-body2")


# ---------------------------------------------------------------------------
# Context cards
# ---------------------------------------------------------------------------


def _build_stage_status_card(board_dir: Path) -> None:
    """Card showing gate status for each pipeline stage as a horizontal stepper (Feature 4)."""
    from nicegui import ui

    from kicad_pipeline.evidence.gates import ALL_STAGES, check_gate
    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)

    with ui.card().classes("w-full"):
        ui.label("Pipeline Status").classes("text-subtitle1 font-bold")

        with ui.row().classes("w-full items-center justify-between"):
            for i, stage in enumerate(ALL_STAGES):
                result = check_gate(board_pcb, stage)
                if result.passed:
                    color = "green"
                    icon = "check_circle"
                elif result.missing:
                    color = "red"
                    icon = "cancel"
                else:
                    color = "grey"
                    icon = "radio_button_unchecked"

                with ui.column().classes("items-center"):
                    ui.icon(icon, color=color).classes("text-h5")
                    ui.label(stage.capitalize()).classes("text-caption font-medium")
                    if result.missing:
                        missing_str = ", ".join(result.missing[:2])
                        if len(result.missing) > 2:
                            missing_str += f" +{len(result.missing) - 2}"
                        ui.label(missing_str).classes("text-caption text-grey").style(
                            "font-size: 10px"
                        )

                    # Feature 10: Stale approval warning
                    if result.passed:
                        approval_records = [
                            r
                            for r in ledger.records
                            if r.kind == EvidenceKind.HUMAN_APPROVAL
                            and r.stage == stage
                            and r.passed
                        ]
                        if approval_records:
                            latest_approval = approval_records[-1]
                            newer_evidence = [
                                r
                                for r in ledger.records
                                if r.stage == stage
                                and r.timestamp > latest_approval.timestamp
                                and r.kind != EvidenceKind.HUMAN_APPROVAL
                            ]
                            if newer_evidence:
                                ui.label("Approval may be stale").classes(
                                    "text-caption text-orange"
                                ).style("font-size: 10px")

                # Arrow between stages
                if i < len(ALL_STAGES) - 1:
                    ui.icon("arrow_forward").classes("text-grey-6")


def _build_quality_score_card(board_dir: Path) -> None:
    """Card showing the latest quality score snapshot."""
    from nicegui import ui

    from kicad_pipeline.evidence.ledger import load_ledger

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)
    score = ledger.latest_score()

    with ui.card().classes("w-full"):
        ui.label("Quality Score").classes("text-subtitle1 font-bold")
        if score is None:
            ui.label("No score data yet").classes("text-grey")
            return

        with ui.row().classes("items-center gap-4"):
            grade_colors = {
                "A": "green",
                "B": "light-green",
                "C": "yellow",
                "D": "orange",
                "F": "red",
            }
            color = grade_colors.get(score.grade, "grey")
            ui.label(score.grade).classes(f"text-h3 font-bold text-{color}")
            ui.label(f"{score.overall_score:.3f}").classes("text-h5")

        if score.breakdown:
            with ui.expansion("Score breakdown", icon="analytics"):
                for dimension, value in sorted(score.breakdown.items()):
                    with ui.row().classes("justify-between"):
                        ui.label(dimension).classes("text-caption")
                        ui.label(f"{value:.3f}").classes("text-caption font-mono")


def _score_color(val: float) -> str:
    """Return a hex color for a score value based on grade boundaries."""
    if val >= 0.9:
        return "#4caf50"
    if val >= 0.75:
        return "#ffeb3b"
    if val >= 0.6:
        return "#ff9800"
    return "#f44336"


def _extract_score_series(
    timestamps: list[str], scores: list[float]
) -> list[dict[str, object]]:
    pieces: list[dict[str, object]] = []
    for idx in range(len(scores) - 1):
        color = _score_color(scores[idx])
        pieces.append(
            {
                "type": "line",
                "data": [
                    [timestamps[idx], scores[idx]],
                    [timestamps[idx + 1], scores[idx + 1]],
                ],
                "lineStyle": {"color": color, "width": 2},
                "itemStyle": {"color": color},
                "symbol": "circle",
                "symbolSize": 6,
            }
        )
    if len(scores) == 1:
        color = _score_color(scores[0])
        pieces.append(
            {
                "type": "line",
                "data": [[timestamps[0], scores[0]]],
                "lineStyle": {"color": color, "width": 2},
                "itemStyle": {"color": color},
                "symbol": "circle",
                "symbolSize": 8,
            }
        )
    return pieces


def _build_score_trend_options(
    timestamps: list[str], scores: list[float]
) -> dict[str, object]:
    pieces = _extract_score_series(timestamps, scores)
    return {
        "animation": False,
        "grid": {"left": 45, "right": 15, "top": 20, "bottom": 30},
        "xAxis": {"type": "category", "data": timestamps},
        "yAxis": {"type": "value", "min": 0.0, "max": 1.0},
        "series": [
            *pieces,
            {
                "type": "line",
                "markLine": {
                    "silent": True,
                    "symbol": "none",
                    "lineStyle": {"type": "dashed", "width": 1},
                    "data": [
                        {"yAxis": 0.9, "label": {"formatter": "A"},
                         "lineStyle": {"color": "#4caf50"}},
                        {"yAxis": 0.75, "label": {"formatter": "B"},
                         "lineStyle": {"color": "#ffeb3b"}},
                        {"yAxis": 0.6, "label": {"formatter": "C"},
                         "lineStyle": {"color": "#ff9800"}},
                    ],
                },
                "data": [],
            },
        ],
        "tooltip": {"trigger": "axis"},
    }


def _build_score_trend_card(board_dir: Path) -> None:
    """Card showing score history as a line chart."""
    from nicegui import ui

    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)
    score_records = ledger.filter_by_kind(EvidenceKind.SCORE)

    with ui.card().classes("w-full"):
        ui.label("Score History").classes("text-subtitle1 font-bold")

        if not score_records:
            ui.label("No scores recorded yet").classes("text-grey")
            return

        timestamps: list[str] = []
        scores: list[float] = []
        for rec in score_records:
            timestamps.append(rec.timestamp.strftime("%H:%M"))
            details = rec.details
            overall = details.get("overall_score", 0.0) if details else 0.0
            scores.append(float(overall))

        options = _build_score_trend_options(timestamps, scores)
        ui.chart(options).classes("w-full h-48")


def _load_requirements(board_dir: Path) -> list[dict[str, object]] | None:
    """Load features list from requirements.json."""
    import json as _json

    req_path = board_dir / "requirements.json"
    if not req_path.exists():
        return None
    try:
        data = _json.loads(req_path.read_text(encoding="utf-8"))
        return data.get("features", [])
    except (ValueError, KeyError):
        return None


def _derive_board_status(board_dir: Path) -> str:
    """Derive overall board status from evidence ledger."""
    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)

    score_records = [r for r in ledger.filter_by_kind(EvidenceKind.SCORE) if r.passed is True]
    if score_records:
        latest = score_records[-1]
        grade = latest.details.get("grade", "") if latest.details else ""
        if grade in ("A", "B"):
            return "Ready"
        return "Needs Work"

    if ledger.filter_by_kind(EvidenceKind.REVIEW):
        return "Needs Work"

    return "Not Started"


def _build_requirements_status_card(board_dir: Path) -> None:
    """Card showing requirements/features tracking."""
    from nicegui import ui

    features = _load_requirements(board_dir)

    with ui.card().classes("w-full"):
        ui.label("Requirements Status").classes("text-subtitle1 font-bold")

        if features is None:
            ui.label("No requirements.json found").classes("text-grey")
            return
        if not features:
            ui.label("No features defined").classes("text-grey")
            return

        board_status = _derive_board_status(board_dir)
        status_colors = {"Ready": "green", "Needs Work": "orange", "Not Started": "grey"}

        columns = [
            {"name": "feature", "label": "Feature", "field": "feature"},
            {"name": "components", "label": "Components", "field": "components"},
            {"name": "status", "label": "Status", "field": "status"},
        ]
        rows = []
        for feat in features:
            name = str(feat.get("name", ""))
            comps = feat.get("components", [])
            comp_count = len(comps) if isinstance(comps, list) else 0
            rows.append({"feature": name, "components": str(comp_count), "status": board_status})

        ui.table(columns=columns, rows=rows).classes("w-full")

        with ui.row().classes("gap-2 q-mt-sm"):
            ui.badge(
                f"{len(features)} features | {board_status}",
                color=status_colors.get(board_status, "grey"),
            )


def _get_review_personas(
    review_records: list[object],
) -> tuple[object, list[object], list[object]]:
    latest = review_records[-1]  # type: ignore[index]
    current_stage = latest.stage  # type: ignore[union-attr]
    stage_reviews = [r for r in review_records if r.stage == current_stage]  # type: ignore[union-attr]
    fab_reviews = [r for r in stage_reviews if "fab" in (r.producer or "").lower()]  # type: ignore[union-attr]
    ee_reviews = [r for r in stage_reviews if "ee" in (r.producer or "").lower()]  # type: ignore[union-attr]
    return latest, fab_reviews, ee_reviews


def _collect_all_review_issues(
    latest: object,
    fab_reviews: list[object],
    ee_reviews: list[object],
) -> list[object]:
    if fab_reviews and ee_reviews:
        return list(fab_reviews[-1].issues) + list(ee_reviews[-1].issues)  # type: ignore[union-attr]
    return list(latest.issues)  # type: ignore[union-attr]


def _render_dual_persona(
    fab_reviews: list[object],
    ee_reviews: list[object],
    severity_order: dict[object, int],
    severity_colors: dict[object, str],
) -> None:
    from nicegui import ui

    fab_latest = fab_reviews[-1]
    ee_latest = ee_reviews[-1]
    fab_issues = sorted(fab_latest.issues, key=lambda i: severity_order.get(i.severity, 99))  # type: ignore[union-attr]
    ee_issues = sorted(ee_latest.issues, key=lambda i: severity_order.get(i.severity, 99))  # type: ignore[union-attr]
    fab_refs = {i.ref for i in fab_issues if i.ref}  # type: ignore[union-attr]
    ee_refs = {i.ref for i in ee_issues if i.ref}  # type: ignore[union-attr]
    consensus_refs = fab_refs & ee_refs
    if consensus_refs:
        with ui.row().classes("w-full gap-2 q-mb-sm flex-wrap"):
            for ref in sorted(consensus_refs):
                ui.badge(f"Consensus: {ref}", color="amber").classes("text-black")
    with ui.row().classes("w-full gap-4"):
        with ui.column().classes("flex-1"):
            ui.label("Fabricator Review").classes("text-subtitle2 font-bold")
            _render_issues_table(fab_issues, severity_colors)  # type: ignore[arg-type]
        with ui.column().classes("flex-1"):
            ui.label("EE Review").classes("text-subtitle2 font-bold")
            _render_issues_table(ee_issues, severity_colors)  # type: ignore[arg-type]


def _build_review_findings_card(board_dir: Path) -> None:
    """Card showing issues from review evidence.

    Feature 3: Dual-persona review diff view -- if both fab and EE reviews exist
    for the current stage, show them side-by-side with cross-validated consensus.
    Feature 6: Clickable ref designators navigate to crop images.
    """
    from nicegui import ui

    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind, Severity

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)
    review_records = ledger.filter_by_kind(EvidenceKind.REVIEW)

    severity_order: dict[object, int] = {
        Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2, Severity.INFO: 3
    }
    severity_colors: dict[object, str] = {
        Severity.CRITICAL: "red", Severity.MAJOR: "orange",
        Severity.MINOR: "yellow", Severity.INFO: "blue",
    }

    with ui.card().classes("w-full"):
        ui.label("Review Findings").classes("text-subtitle1 font-bold")

        if not review_records:
            ui.label("No reviews yet").classes("text-grey")
            return

        latest, fab_reviews, ee_reviews = _get_review_personas(review_records)
        all_issues = _collect_all_review_issues(latest, fab_reviews, ee_reviews)

        if fab_reviews and ee_reviews:
            _render_dual_persona(fab_reviews, ee_reviews, severity_order, severity_colors)
        else:
            if not latest.issues:  # type: ignore[union-attr]
                ui.label("No issues found").classes("text-green")
                return
            sorted_issues = sorted(latest.issues, key=lambda i: severity_order.get(i.severity, 99))  # type: ignore[union-attr]
            _render_issues_table(sorted_issues, severity_colors)  # type: ignore[arg-type]

        with ui.row().classes("gap-2 q-mt-sm"):
            for sev in Severity:
                count = sum(1 for i in all_issues if i.severity == sev)  # type: ignore[union-attr]
                if count > 0:
                    ui.badge(f"{sev.value}: {count}", color=severity_colors.get(sev, "grey"))

        board_name = board_dir.name if board_dir else ""

        def _report_issue_from_finding(issue_desc: str) -> None:
            from kicad_pipeline.dashboard.kanban import KanbanCard, add_card

            project_root = board_dir.parent.parent
            new_card = KanbanCard(
                title=issue_desc[:80], description=issue_desc,
                card_type="bug", level="board", board_name=board_name,
            )
            add_card(project_root, new_card)
            ui.notify(f"Issue reported to kanban: {issue_desc[:50]}...", type="positive")

        if all_issues:

            def _report_all(_e: object = None) -> None:
                for issue in all_issues:
                    _report_issue_from_finding(issue.description)  # type: ignore[union-attr]
                ui.notify(f"Reported {len(all_issues)} issues to kanban", type="positive")

            ui.button(
                f"Report {len(all_issues)} issues to Kanban",
                on_click=_report_all,
                icon="bug_report",
            ).props("flat dense no-caps size=sm color=red").classes("q-mt-sm")


def _render_issues_table(
    issues: list[Issue],
    severity_colors: dict[object, str],
) -> None:
    """Render an issues table with clickable ref designators (Feature 6)."""
    from nicegui import ui

    if not issues:
        ui.label("No issues found").classes("text-green")
        return

    for issue in issues:
        sev_color = severity_colors.get(issue.severity, "grey")
        with (
            ui.row()
            .classes("items-center gap-2 q-py-xs")
            .style("border-bottom: 1px solid rgba(128,128,128,0.2)")
        ):
            ui.badge(issue.severity.value, color=sev_color).style("min-width: 60px")
            if issue.ref:
                ui.label(issue.ref).classes("cursor-pointer text-blue font-bold").on(
                    "click", lambda _e=None, r=issue.ref: _navigate_to_crop(r)
                )
            else:
                ui.label("--").classes("text-grey")
            ui.label(issue.description).classes("text-caption")


_KIND_ICONS: dict[str, str] = {
    "render": "image",
    "review": "rate_review",
    "score": "speed",
    "verification": "verified",
    "human_approval": "thumb_up",
    "human_rejection": "thumb_down",
    "gate_result": "security",
    "drc_report": "assignment",
    "known_issue_check": "bug_report",
}


def _build_activity_feed_card(board_dir: Path) -> None:
    """Card showing the last 10 evidence records chronologically (newest first)."""
    from datetime import datetime, timezone

    from nicegui import ui

    from kicad_pipeline.evidence.ledger import load_ledger

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)

    with ui.card().classes("w-full"):
        ui.label("Recent Activity").classes("text-subtitle1 font-bold")

        if not ledger.records:
            ui.label("No activity yet").classes("text-grey")
            return

        recent = list(reversed(ledger.records[-10:]))
        now = datetime.now(tz=timezone.utc)

        for rec in recent:
            icon_name = _KIND_ICONS.get(rec.kind.value, "info")
            delta = now - rec.timestamp
            if delta.total_seconds() < 60:
                ago = "just now"
            elif delta.total_seconds() < 3600:
                mins = int(delta.total_seconds() // 60)
                ago = f"{mins}m ago"
            elif delta.total_seconds() < 86400:
                hours = int(delta.total_seconds() // 3600)
                ago = f"{hours}h ago"
            else:
                days = int(delta.total_seconds() // 86400)
                ago = f"{days}d ago"

            ts_str = rec.timestamp.strftime("%H:%M:%S")
            summary = rec.summary or rec.kind.value.replace("_", " ").title()

            if rec.passed is True:
                badge_color = "green"
                badge_text = "pass"
            elif rec.passed is False:
                badge_color = "red"
                badge_text = "fail"
            else:
                badge_color = "grey"
                badge_text = "info"

            with ui.expansion(
                text=f"{ts_str}  {summary}",
                icon=icon_name,
            ).classes("w-full"):
                with ui.row().classes("items-center gap-2 q-mb-xs"):
                    ui.badge(badge_text, color=badge_color)
                    ui.label(ago).classes("text-caption text-grey")
                with ui.column().classes("gap-1"):
                    ui.label(f"Kind: {rec.kind.value}").classes("text-caption")
                    ui.label(f"Stage: {rec.stage}").classes("text-caption")
                    ui.label(f"Step: {rec.step}").classes("text-caption")
                    ui.label(f"Producer: {rec.producer}").classes("text-caption")
                    if rec.feedback:
                        ui.label(f"Feedback: {rec.feedback}").classes("text-caption text-orange")
                    if rec.details:
                        ui.label(f"Details: {rec.details}").classes("text-caption text-grey")
                    if rec.artifacts:
                        ui.label(f"Artifacts: {', '.join(rec.artifacts)}").classes(
                            "text-caption text-grey"
                        )


def _load_drc_report(board_dir: Path) -> dict[str, object] | None:
    """Load drc_report.json from a board directory, returning None if missing."""
    import json as _json

    drc_path = board_dir / "drc_report.json"
    if not drc_path.exists():
        return None
    try:
        return _json.loads(drc_path.read_text(encoding="utf-8"))
    except (ValueError, KeyError):
        return None


def _extract_drc_ref(item: dict[str, object]) -> str:
    sub_items = item.get("items", [])
    if isinstance(sub_items, list):
        for si in sub_items:
            if isinstance(si, dict):
                desc = str(si.get("description", ""))
                if " of " in desc:
                    ref_candidate = desc.split(" of ")[-1].strip()
                    if ref_candidate and ref_candidate[0].isalpha():
                        return ref_candidate
    return ""


def _render_drc_violations(all_items: list[dict[str, object]], total: int) -> None:
    from nicegui import ui

    type_counts: dict[str, int] = {}
    for item in all_items:
        vtype = str(item.get("type", "unknown"))
        type_counts[vtype] = type_counts.get(vtype, 0) + 1

    with ui.expansion(f"Violations ({total})", icon="warning").classes("w-full"):
        with ui.row().classes("gap-2 q-mb-sm flex-wrap"):
            for vtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                ui.badge(f"{vtype}: {count}", color="grey")

        for item in all_items:
            severity = str(item.get("severity", ""))
            description = str(item.get("description", ""))
            vtype = str(item.get("type", ""))
            ref = _extract_drc_ref(item)
            sev_color = "red" if severity == "error" else "orange"
            with (
                ui.row()
                .classes("items-center gap-2 q-py-xs")
                .style("border-bottom: 1px solid rgba(128,128,128,0.2)")
            ):
                ui.badge(severity, color=sev_color).style("min-width: 50px")
                ui.label(vtype).classes("text-caption font-bold").style("min-width: 80px")
                if ref:
                    ui.label(ref).classes("cursor-pointer text-blue font-bold").on(
                        "click", lambda _e=None, r=ref: _navigate_to_crop(r)
                    )
                ui.label(description).classes("text-caption")


def _build_drc_report_card(board_dir: Path, role: str = "framework") -> None:
    """Card showing DRC report violations from drc_report.json.

    For framework/deployment roles: full expandable detail view.
    For board role: simplified badge showing error count.
    Feature 6: Clickable ref designators navigate to crop images.
    """
    from nicegui import ui

    drc = _load_drc_report(board_dir)
    if drc is None:
        with (
            ui.card().classes("w-full"),
            ui.row().classes("items-center gap-2"),
        ):
            ui.icon("assignment_late", color="grey").classes("text-lg")
            ui.label("DRC Report").classes("text-subtitle1 font-bold")
            ui.badge("Not run", color="grey")
        return

    violations: list[dict[str, object]] = drc.get("violations", [])  # type: ignore[assignment]
    unconnected: list[dict[str, object]] = drc.get("unconnected_items", [])  # type: ignore[assignment]
    all_items = [*violations, *unconnected]
    error_count = sum(1 for item in all_items if item.get("severity") == "error")
    warning_count = sum(1 for item in all_items if item.get("severity") == "warning")
    total = len(all_items)

    if role == "board":
        with (
            ui.card().classes("w-full"),
            ui.row().classes("items-center gap-2"),
        ):
            ui.icon("assignment").classes("text-lg")
            if error_count == 0 and warning_count == 0:
                ui.badge("DRC clean", color="green")
            else:
                ui.badge(f"DRC: {error_count} errors", color="red" if error_count > 0 else "green")
        return

    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-2"):
            ui.label("DRC Report").classes("text-subtitle1 font-bold")
            if total == 0:
                ui.badge("DRC clean", color="green")
            else:
                if error_count > 0:
                    ui.badge(f"{error_count} errors", color="red")
                if warning_count > 0:
                    ui.badge(f"{warning_count} warnings", color="orange")

        if not all_items:
            ui.label("No violations found").classes("text-green")
            return

        _render_drc_violations(all_items, total)


def _build_known_issues_card(board_dir: Path) -> None:
    """Card showing known issues (framework developer role only)."""
    from nicegui import ui

    review_dir = board_dir / ".pcb-review"
    lessons_path = review_dir / "lessons.md"

    with ui.card().classes("w-full"):
        ui.label("Known Issues / Lessons").classes("text-subtitle1 font-bold")
        if lessons_path.exists():
            content = lessons_path.read_text(encoding="utf-8")
            with ui.expansion("Show details", icon="info"):
                ui.markdown(content).classes("w-full")
        else:
            ui.label("No lessons file found").classes("text-grey")


def _build_actions_card(board_dir: Path) -> None:
    """Card with approve/reject action buttons."""
    from nicegui import ui

    from kicad_pipeline.evidence.gates import ALL_STAGES, check_gate
    from kicad_pipeline.evidence.ledger import append_record
    from kicad_pipeline.evidence.models import EvidenceKind, EvidenceRecord

    board_pcb = _find_board_pcb(board_dir)
    board_name = board_pcb.stem

    # Auto-detect the stage that needs approval
    pending_stage = "pcb"
    for stage in ALL_STAGES:
        result = check_gate(board_pcb, stage)
        if not result.passed and EvidenceKind.HUMAN_APPROVAL.value in result.missing:
            pending_stage = stage
            break

    with ui.card().classes("w-full"):
        ui.label("Actions").classes("text-subtitle1 font-bold")

        stage_select = ui.select(
            options=list(ALL_STAGES),
            value=pending_stage,
            label="Stage",
        ).classes("w-full")

        with ui.row().classes("gap-2 q-mt-sm"):

            def _approve() -> None:
                record = EvidenceRecord(
                    kind=EvidenceKind.HUMAN_APPROVAL,
                    stage=stage_select.value,
                    step="dashboard_approval",
                    board=board_name,
                    passed=True,
                    summary=f"Human approved {stage_select.value} via dashboard",
                    producer="human",
                )
                append_record(board_pcb, record)
                ui.notify(f"Approved {stage_select.value}!", type="positive")

            ui.button("Approve", on_click=_approve, color="green", icon="check")

            def _show_reject_dialog() -> None:
                with ui.dialog() as dlg, ui.card().classes("w-96"):
                    ui.label("Reject -- provide feedback").classes("text-h6")
                    feedback = ui.textarea("What needs to change?").classes("w-full")

                    def _submit_reject() -> None:
                        if not feedback.value:
                            ui.notify("Feedback required", type="warning")
                            return
                        record = EvidenceRecord(
                            kind=EvidenceKind.HUMAN_REJECTION,
                            stage=stage_select.value,
                            step="dashboard_rejection",
                            board=board_name,
                            passed=False,
                            summary=f"Human rejected {stage_select.value}",
                            feedback=feedback.value,
                            producer="human",
                        )
                        append_record(board_pcb, record)
                        ui.notify("Rejection recorded", type="negative")
                        dlg.close()

                    with ui.row().classes("justify-end gap-2 q-mt-md"):
                        ui.button("Cancel", on_click=dlg.close).props("flat")
                        ui.button("Reject", on_click=_submit_reject, color="red")
                dlg.open()

            ui.button("Reject", on_click=_show_reject_dialog, color="red", icon="close")


# ---------------------------------------------------------------------------
# Feature 13: Manufacturing readiness checklist
# ---------------------------------------------------------------------------


def _gather_readiness_checks(
    board_dir: Path, board_pcb: Path,
) -> list[tuple[str, bool]]:
    from kicad_pipeline.evidence.gates import ALL_STAGES, check_gate
    from kicad_pipeline.evidence.ledger import load_ledger

    checks: list[tuple[str, bool]] = []
    checks.append(("All pipeline gates passed",
                    all(check_gate(board_pcb, s).passed for s in ALL_STAGES)))
    gerber_dir = board_dir / "gerber"
    checks.append(("Gerber files generated",
                    gerber_dir.is_dir() and any(gerber_dir.glob("*.gbr"))))
    checks.append(("BOM file present",
                    any(board_dir.glob("*bom*.*")) or any(board_dir.glob("*BOM*.*"))))
    checks.append(("Pick-and-place file present",
                    any(board_dir.glob("*cpl*.*")) or any(board_dir.glob("*CPL*.*"))
                    or any(board_dir.glob("*pos*.*"))))
    drc = _load_drc_report(board_dir)
    drc_clean = False
    if drc:
        drc_violations: list[dict[str, object]] = drc.get("violations", [])  # type: ignore[assignment]
        drc_clean = sum(1 for v in drc_violations if v.get("severity") == "error") == 0
    checks.append(("DRC clean (0 errors)", drc_clean))
    ledger = load_ledger(board_pcb)
    score = ledger.latest_score()
    score_grade = score.grade if score else "?"
    checks.append((f"Quality score >= B ({score_grade})",
                    score is not None and score.grade in ("A", "B")))
    checks.append(("Validation approved", check_gate(board_pcb, "validation").passed))
    checks.append(("Production approved", check_gate(board_pcb, "production").passed))
    return checks


def _build_manufacturing_readiness_card(board_dir: Path, output_root: Path) -> None:
    """Card showing manufacturing readiness checklist."""
    from nicegui import ui

    board_pcb = _find_board_pcb(board_dir)
    checks = _gather_readiness_checks(board_dir, board_pcb)
    ready_count = sum(1 for _, ok in checks if ok)
    total = len(checks)

    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-2"):
            ui.label("Manufacturing Readiness").classes("text-subtitle1 font-bold")
            if ready_count == total:
                ui.badge("Ready to order", color="green")
            else:
                ui.badge(f"{ready_count}/{total}", color="orange")

        for label, passed in checks:
            with ui.row().classes("items-center gap-2"):
                icon = "check_circle" if passed else "radio_button_unchecked"
                color = "green" if passed else "grey"
                ui.icon(icon, color=color)
                ui.label(label).classes("text-body2")


# ---------------------------------------------------------------------------
# Iteration Diff View
# ---------------------------------------------------------------------------


def _collect_iterations(board_dir: Path) -> list[dict[str, object]]:
    """Group RENDER, SCORE, and REVIEW evidence into iteration snapshots."""
    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind, ScoreSnapshot

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)

    render_records = ledger.filter_by_kind(EvidenceKind.RENDER)
    if not render_records:
        return []

    score_records = ledger.filter_by_kind(EvidenceKind.SCORE)
    review_records = ledger.filter_by_kind(EvidenceKind.REVIEW)

    iterations: list[dict[str, object]] = []
    for render_rec in render_records:
        ts = render_rec.timestamp
        label = ts.strftime("%Y-%m-%d %H:%M:%S")

        score_snap: ScoreSnapshot | None = None
        grade = "?"
        for sr in score_records:
            if abs((sr.timestamp - ts).total_seconds()) < 60:
                try:
                    score_snap = ScoreSnapshot.model_validate(sr.details)
                    grade = score_snap.grade
                except Exception as exc:
                    logger.debug("ScoreSnapshot validation failed for review record: %s", exc)
                break

        issues: list[Issue] = []
        for rr in review_records:
            if abs((rr.timestamp - ts).total_seconds()) < 60:
                issues = list(rr.issues)
                break

        iterations.append(
            {
                "label": label,
                "timestamp": ts,
                "artifacts": list(render_rec.artifacts),
                "score": score_snap,
                "issues": issues,
                "grade": grade,
            }
        )

    return iterations


def _open_diff_dialog(board_dir: Path) -> None:
    """Open a dialog comparing two render iterations side by side."""
    from nicegui import ui

    iterations = _collect_iterations(board_dir)

    with ui.dialog() as dlg, ui.card().classes("w-full max-w-6xl"):
        ui.label("Iteration Compare").classes("text-h5 font-bold q-mb-md")

        if len(iterations) < 2:
            ui.label("Not enough iterations to compare").classes("text-grey text-subtitle1")
            ui.button("Close", on_click=dlg.close)
            dlg.open()
            return

        labels = [str(it["label"]) for it in iterations]

        with ui.row().classes("w-full items-center gap-4 q-mb-md"):
            before_select = ui.select(
                options=labels,
                value=labels[-2],
                label="Before",
            ).classes("w-64")
            ui.icon("arrow_forward").classes("text-h5")
            after_select = ui.select(
                options=labels,
                value=labels[-1],
                label="After",
            ).classes("w-64")

        compare_container = ui.column().classes("w-full")

        def _rebuild_comparison() -> None:
            compare_container.clear()
            before_it = next((it for it in iterations if it["label"] == before_select.value), None)
            after_it = next((it for it in iterations if it["label"] == after_select.value), None)
            if not before_it or not after_it:
                return

            with compare_container:
                _render_side_by_side(
                    before_it,
                    after_it,
                    str(before_select.value),
                    str(after_select.value),
                )
                ui.separator().classes("q-my-md")
                _render_change_summary(before_it, after_it)

        _rebuild_comparison()
        before_select.on_value_change(lambda _: _rebuild_comparison())
        after_select.on_value_change(lambda _: _rebuild_comparison())

        ui.button("Close", on_click=dlg.close).classes("q-mt-md")

    dlg.open()


def _render_side_by_side(
    before_it: dict[str, object],
    after_it: dict[str, object],
    before_label: str,
    after_label: str,
) -> None:
    """Render before/after images side by side."""
    from pathlib import Path as _Path

    from nicegui import ui

    before_arts: list[str] = before_it.get("artifacts", [])  # type: ignore[assignment]
    after_arts: list[str] = after_it.get("artifacts", [])  # type: ignore[assignment]

    with ui.row().classes("w-full gap-4"):
        with ui.column().classes("flex-1"):
            ui.label(f"Before -- {before_label}").classes("text-subtitle1 font-bold")
            for art_str in before_arts:
                art = _Path(art_str)
                if art.exists() and art.suffix.lower() == ".png":
                    ui.image(_image_url(art)).classes("w-full max-h-80 object-contain")

        with ui.column().classes("flex-1"):
            ui.label(f"After -- {after_label}").classes("text-subtitle1 font-bold")
            for art_str in after_arts:
                art = _Path(art_str)
                if art.exists() and art.suffix.lower() == ".png":
                    ui.image(_image_url(art)).classes("w-full max-h-80 object-contain")


def _render_change_summary(
    before_it: dict[str, object],
    after_it: dict[str, object],
) -> None:
    """Render the change summary: grade, score delta, issues diff."""
    from nicegui import ui

    ui.label("Change Summary").classes("text-subtitle1 font-bold q-mb-sm")

    before_score: ScoreSnapshot | None = before_it.get("score")  # type: ignore[assignment]
    after_score: ScoreSnapshot | None = after_it.get("score")  # type: ignore[assignment]
    before_grade = str(before_it.get("grade", "?"))
    after_grade = str(after_it.get("grade", "?"))
    before_issues: list[Issue] = before_it.get("issues", [])  # type: ignore[assignment]
    after_issues: list[Issue] = after_it.get("issues", [])  # type: ignore[assignment]

    grade_colors = {"A": "green", "B": "light-green", "C": "yellow", "D": "orange", "F": "red"}

    with ui.row().classes("items-center gap-2"):
        ui.label("Grade:").classes("font-bold")
        if before_grade != after_grade:
            ui.badge(before_grade, color=grade_colors.get(before_grade, "grey"))
            ui.icon("arrow_forward")
            ui.badge(after_grade, color=grade_colors.get(after_grade, "grey"))
        else:
            ui.label(f"{before_grade} (unchanged)").classes("text-grey")

    if before_score and after_score:
        delta = after_score.overall_score - before_score.overall_score
        sign = "+" if delta >= 0 else ""
        delta_color = "green" if delta >= 0 else "red"
        with ui.row().classes("items-center gap-2"):
            ui.label("Score:").classes("font-bold")
            ui.label(
                f"{before_score.overall_score:.3f} -> {after_score.overall_score:.3f}"
            ).classes("font-mono")
            ui.label(f"({sign}{delta:.3f})").classes(f"font-mono text-{delta_color}")

    # Feature 7: Per-dimension breakdown comparison
    if before_score and after_score and before_score.breakdown and after_score.breakdown:
        all_dims = sorted(set(before_score.breakdown) | set(after_score.breakdown))
        with ui.card().classes("w-full q-mt-sm"):
            ui.label("Per-Dimension Breakdown").classes("text-subtitle2 font-bold")
            columns = [
                {"name": "dim", "label": "Dimension", "field": "dim"},
                {"name": "before", "label": "Before", "field": "before"},
                {"name": "after", "label": "After", "field": "after"},
                {"name": "delta", "label": "Delta", "field": "delta"},
            ]
            rows = []
            for dim in all_dims:
                bv = before_score.breakdown.get(dim, 0.0)
                av = after_score.breakdown.get(dim, 0.0)
                d = av - bv
                dim_sign = "+" if d >= 0 else ""
                rows.append(
                    {
                        "dim": dim,
                        "before": f"{bv:.3f}",
                        "after": f"{av:.3f}",
                        "delta": f"{dim_sign}{d:.3f}",
                    }
                )
            ui.table(columns=columns, rows=rows).classes("w-full")

    before_descs = {i.description for i in before_issues}
    after_descs = {i.description for i in after_issues}
    resolved = before_descs - after_descs
    new_issues = after_descs - before_descs

    with ui.row().classes("items-center gap-2 q-mt-sm"):
        ui.label("Issues:").classes("font-bold")
        ui.label(f"{len(before_issues)} -> {len(after_issues)}").classes("font-mono")
        if resolved:
            ui.badge(f"-{len(resolved)} resolved", color="green")
        if new_issues:
            ui.badge(f"+{len(new_issues)} new", color="red")

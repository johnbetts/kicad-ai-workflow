"""Panel builders for the NiceGUI review dashboard."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Board-level image name prefixes (shown first, in this order).
_BOARD_IMAGE_ORDER = ("2d_top", "3d_top", "3d_iso", "3d_isoback", "3d_hires_top", "3d_padoverlay")


def _sorted_images(board_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return (board_images, crop_images) sorted for display."""
    all_pngs = sorted(board_dir.glob("*.png"))
    board_images: list[Path] = []
    other_images: list[Path] = []

    # Separate board-level images in preferred order
    name_to_path = {p.stem: p for p in all_pngs}
    for prefix in _BOARD_IMAGE_ORDER:
        if prefix in name_to_path:
            board_images.append(name_to_path[prefix])
    # Any remaining top-level PNGs
    board_stems = {p.stem for p in board_images}
    for p in all_pngs:
        if p.stem not in board_stems:
            other_images.append(p)

    # Crop images from crops/ subdirectory
    crops_dir = board_dir / "crops"
    crop_images: list[Path] = []
    if crops_dir.is_dir():
        crop_images = sorted(crops_dir.glob("*.png"))

    return board_images + other_images, crop_images


def build_image_panel(board_dir: Path | None) -> None:
    """Build the image gallery panel (left side)."""
    from nicegui import ui

    ui.label("Images").classes("text-h6 font-bold q-mb-sm")

    if board_dir is None or not board_dir.is_dir():
        ui.label("No board selected").classes("text-grey")
        return

    image_column = ui.column().classes("w-full gap-2")

    def _refresh_images() -> None:
        image_column.clear()
        board_images, crop_images = _sorted_images(board_dir)

        with image_column:
            if not board_images and not crop_images:
                ui.label("No images found").classes("text-grey")
                return

            if board_images:
                ui.label("Board Views").classes("text-subtitle2 font-bold")
                with ui.row().classes("flex-wrap gap-2"):
                    for img_path in board_images:
                        _make_thumbnail(img_path)

            if crop_images:
                ui.label("Component Crops").classes("text-subtitle2 font-bold q-mt-md")
                with ui.row().classes("flex-wrap gap-2"):
                    for img_path in crop_images:
                        _make_thumbnail(img_path)

    def _make_thumbnail(img_path: Path) -> None:
        from nicegui import ui

        img_url = f"/static/images/{img_path.name}"
        # Serve the file via app.add_static_files if not already done
        _ensure_static(img_path)

        with ui.card().classes("cursor-pointer").on("click", lambda p=img_path: _show_fullsize(p)):
            ui.image(img_url).classes("w-32 h-24 object-cover")
            ui.label(img_path.stem).classes("text-caption text-center")

    def _show_fullsize(img_path: Path) -> None:
        from nicegui import ui

        img_url = f"/static/images/{img_path.name}"
        with ui.dialog() as dlg, ui.card().classes("w-full max-w-4xl"):
            ui.label(img_path.stem).classes("text-h6")
            ui.image(img_url).classes("w-full")
            ui.button("Close", on_click=dlg.close)
        dlg.open()

    _refresh_images()
    ui.timer(5.0, _refresh_images)


def _ensure_static(img_path: Path) -> None:
    """Register the image's parent directory as a static file source."""
    from nicegui import app as nicegui_app

    # Use a per-directory approach: serve the board dir as /static/images/
    parent = img_path.parent
    route = "/static/images"
    # NiceGUI's add_static_files is idempotent for the same route
    nicegui_app.add_static_files(route, str(parent))


def build_log_panel() -> None:
    """Build the log stream panel (center)."""
    from nicegui import ui

    from kicad_pipeline.dashboard.app import _log_buffer

    ui.label("Log Stream").classes("text-h6 font-bold q-mb-sm")
    log_widget = ui.log(max_lines=500).classes("w-full h-full")

    def _push_buffered() -> None:
        while _log_buffer:
            entry = _log_buffer.pop(0)
            log_widget.push(entry)

    ui.timer(1.0, _push_buffered)


def build_context_panel(board_dir: Path | None, output_root: Path) -> None:
    """Build the context/status panel (right side)."""
    from nicegui import ui

    ui.label("Context").classes("text-h6 font-bold q-mb-sm")

    if board_dir is None or not board_dir.is_dir():
        ui.label("No board selected").classes("text-grey")
        return

    context_column = ui.column().classes("w-full gap-2")

    def _refresh_context() -> None:
        context_column.clear()
        with context_column:
            _build_stage_status_card(board_dir)
            _build_quality_score_card(board_dir)
            _build_review_findings_card(board_dir)
            _build_known_issues_card(board_dir)
            _build_actions_card(board_dir)

    _refresh_context()
    ui.timer(3.0, _refresh_context)


def _find_board_pcb(board_dir: Path) -> Path:
    """Find the .kicad_pcb file in a board directory."""
    pcbs = list(board_dir.glob("*.kicad_pcb"))
    if pcbs:
        return pcbs[0]
    # Fallback: construct from directory name
    return board_dir / f"{board_dir.name}.kicad_pcb"


def _build_stage_status_card(board_dir: Path) -> None:
    """Card showing gate status for each pipeline stage."""
    from nicegui import ui

    from kicad_pipeline.evidence.gates import ALL_STAGES, check_gate

    board_pcb = _find_board_pcb(board_dir)

    with ui.card().classes("w-full"):
        ui.label("Stage Status").classes("text-subtitle1 font-bold")
        for stage in ALL_STAGES:
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
            with ui.row().classes("items-center gap-2"):
                ui.icon(icon, color=color).classes("text-lg")
                ui.label(stage.capitalize()).classes("font-medium")
                if result.missing:
                    ui.label(f"missing: {', '.join(result.missing)}").classes(
                        "text-caption text-grey"
                    )


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
            with ui.element("table").classes("w-full q-mt-sm"):
                for dimension, value in sorted(score.breakdown.items()):
                    with ui.row().classes("justify-between"):
                        ui.label(dimension).classes("text-caption")
                        ui.label(f"{value:.3f}").classes("text-caption font-mono")


def _build_review_findings_card(board_dir: Path) -> None:
    """Card showing issues from the latest review evidence."""
    from nicegui import ui

    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind, Severity

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)

    # Collect issues from all REVIEW records
    review_records = ledger.filter_by_kind(EvidenceKind.REVIEW)

    with ui.card().classes("w-full"):
        ui.label("Review Findings").classes("text-subtitle1 font-bold")

        if not review_records:
            ui.label("No reviews yet").classes("text-grey")
            return

        # Take issues from the latest review
        latest = review_records[-1]
        if not latest.issues:
            ui.label("No issues found").classes("text-green")
            return

        # Sort by severity: critical first
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.MAJOR: 1,
            Severity.MINOR: 2,
            Severity.INFO: 3,
        }
        sorted_issues = sorted(latest.issues, key=lambda i: severity_order.get(i.severity, 99))

        severity_colors = {
            Severity.CRITICAL: "red",
            Severity.MAJOR: "orange",
            Severity.MINOR: "yellow",
            Severity.INFO: "blue",
        }

        columns = [
            {"name": "ref", "label": "Ref", "field": "ref"},
            {"name": "rule", "label": "Rule", "field": "rule"},
            {"name": "severity", "label": "Severity", "field": "severity"},
            {"name": "description", "label": "Description", "field": "description"},
        ]
        rows = [
            {
                "ref": issue.ref,
                "rule": issue.rule,
                "severity": issue.severity.value,
                "description": issue.description,
            }
            for issue in sorted_issues
        ]
        ui.table(columns=columns, rows=rows).classes("w-full")

        # Summary badges
        with ui.row().classes("gap-2 q-mt-sm"):
            for sev in Severity:
                count = sum(1 for i in sorted_issues if i.severity == sev)
                if count > 0:
                    color = severity_colors.get(sev, "grey")
                    ui.badge(f"{sev.value}: {count}", color=color)


def _build_known_issues_card(board_dir: Path) -> None:
    """Card showing known issues from the .pcb-review directory."""
    from nicegui import ui

    review_dir = board_dir / ".pcb-review"
    lessons_path = review_dir / "lessons.md"

    with ui.card().classes("w-full"):
        ui.label("Known Issues / Lessons").classes("text-subtitle1 font-bold")
        if lessons_path.exists():
            content = lessons_path.read_text(encoding="utf-8")
            ui.markdown(content).classes("w-full")
        else:
            ui.label("No lessons file found").classes("text-grey")


def _build_actions_card(board_dir: Path) -> None:
    """Card with approve/reject action buttons."""
    from nicegui import ui

    from kicad_pipeline.evidence.ledger import append_record
    from kicad_pipeline.evidence.models import EvidenceKind, EvidenceRecord

    board_pcb = _find_board_pcb(board_dir)
    board_name = board_pcb.stem

    with ui.card().classes("w-full"):
        ui.label("Actions").classes("text-subtitle1 font-bold")
        stage_input = ui.input(label="Stage", value="pcb").classes("w-full")

        def _approve() -> None:
            record = EvidenceRecord(
                kind=EvidenceKind.HUMAN_APPROVAL,
                stage=stage_input.value,
                step="dashboard_approval",
                board=board_name,
                passed=True,
                summary=f"Human approved {stage_input.value} stage via dashboard",
                producer="dashboard",
            )
            append_record(board_pcb, record)
            ui.notify("Approved!", type="positive")

        ui.button("Approve", on_click=_approve, color="green").classes("q-mr-sm")

        # Reject with feedback
        feedback_input = ui.textarea(label="Rejection feedback").classes("w-full")

        def _reject() -> None:
            if not feedback_input.value:
                ui.notify("Please provide feedback", type="warning")
                return
            record = EvidenceRecord(
                kind=EvidenceKind.HUMAN_REJECTION,
                stage=stage_input.value,
                step="dashboard_rejection",
                board=board_name,
                passed=False,
                summary=f"Human rejected {stage_input.value} stage via dashboard",
                feedback=feedback_input.value,
                producer="dashboard",
            )
            append_record(board_pcb, record)
            ui.notify("Rejection recorded", type="negative")
            feedback_input.value = ""

        ui.button("Reject", on_click=_reject, color="red")

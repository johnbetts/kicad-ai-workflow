"""Panel builders for the NiceGUI review dashboard."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.evidence.models import Issue, ScoreSnapshot

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

    # "Compare Iterations" button — only shown when there are iterations to compare
    ui.button(
        "Compare Iterations",
        on_click=lambda: _open_diff_dialog(board_dir),
        icon="compare",
    ).classes("q-mt-md")


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
            _build_score_trend_card(board_dir)
            _build_requirements_status_card(board_dir)
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


def _score_color(val: float) -> str:
    """Return a hex color for a score value based on grade boundaries."""
    if val >= 0.9:
        return "#4caf50"  # green
    if val >= 0.75:
        return "#ffeb3b"  # yellow
    if val >= 0.6:
        return "#ff9800"  # orange
    return "#f44336"  # red


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

        # Extract timestamps and scores
        timestamps: list[str] = []
        scores: list[float] = []
        for rec in score_records:
            timestamps.append(rec.timestamp.strftime("%H:%M"))
            details = rec.details
            overall = details.get("overall_score", 0.0) if details else 0.0
            scores.append(float(overall))

        # Build colored line segments based on score value
        pieces: list[dict[str, object]] = []

        for idx in range(len(scores) - 1):
            color = _score_color(scores[idx])
            segment_data: list[list[object]] = [
                [timestamps[idx], scores[idx]],
                [timestamps[idx + 1], scores[idx + 1]],
            ]
            pieces.append(
                {
                    "type": "line",
                    "data": segment_data,
                    "lineStyle": {"color": color, "width": 2},
                    "itemStyle": {"color": color},
                    "symbol": "circle",
                    "symbolSize": 6,
                }
            )

        # Single data point: show as a dot
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

        options: dict[str, object] = {
            "animation": False,
            "grid": {"left": 45, "right": 15, "top": 20, "bottom": 30},
            "xAxis": {"type": "category", "data": timestamps},
            "yAxis": {
                "type": "value",
                "min": 0.0,
                "max": 1.0,
                "splitLine": {"show": True},
            },
            "series": [
                *pieces,
                # Grade boundary reference lines
                {
                    "type": "line",
                    "markLine": {
                        "silent": True,
                        "symbol": "none",
                        "lineStyle": {"type": "dashed", "width": 1},
                        "data": [
                            {
                                "yAxis": 0.9,
                                "label": {"formatter": "A", "position": "end"},
                                "lineStyle": {"color": "#4caf50"},
                            },
                            {
                                "yAxis": 0.75,
                                "label": {"formatter": "B", "position": "end"},
                                "lineStyle": {"color": "#ffeb3b"},
                            },
                            {
                                "yAxis": 0.6,
                                "label": {"formatter": "C", "position": "end"},
                                "lineStyle": {"color": "#ff9800"},
                            },
                        ],
                    },
                    "data": [],
                },
            ],
            "tooltip": {"trigger": "axis"},
        }

        ui.chart(options).classes("w-full h-48")


def _load_requirements(board_dir: Path) -> list[dict[str, object]] | None:
    """Load features list from requirements.json, or None if missing."""
    import json as _json

    req_path = board_dir / "requirements.json"
    if not req_path.exists():
        return None
    try:
        data = _json.loads(req_path.read_text(encoding="utf-8"))
        features: list[dict[str, object]] = data.get("features", [])
        return features
    except (ValueError, KeyError):
        return None


def _derive_board_status(board_dir: Path) -> str:
    """Derive overall board status from evidence ledger."""
    from kicad_pipeline.evidence.ledger import load_ledger
    from kicad_pipeline.evidence.models import EvidenceKind

    board_pcb = _find_board_pcb(board_dir)
    ledger = load_ledger(board_pcb)

    # Check for passing SCORE records
    score_records = [
        r for r in ledger.filter_by_kind(EvidenceKind.SCORE) if r.passed is True
    ]
    if score_records:
        latest = score_records[-1]
        grade = latest.details.get("grade", "") if latest.details else ""
        if grade in ("A", "B"):
            return "Ready"
        return "Needs Work"

    # Check for any REVIEW records
    review_records = ledger.filter_by_kind(EvidenceKind.REVIEW)
    if review_records:
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

        status_colors = {
            "Ready": "green",
            "Needs Work": "orange",
            "Not Started": "grey",
        }
        status_color = status_colors.get(board_status, "grey")

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
            rows.append(
                {
                    "feature": name,
                    "components": str(comp_count),
                    "status": board_status,
                }
            )

        ui.table(columns=columns, rows=rows).classes("w-full")

        # Summary badge
        with ui.row().classes("gap-2 q-mt-sm"):
            ui.badge(
                f"{len(features)} features | {board_status}",
                color=status_color,
            )


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


# ---------------------------------------------------------------------------
# Iteration Diff View
# ---------------------------------------------------------------------------


def _collect_iterations(board_dir: Path) -> list[dict[str, object]]:
    """Group RENDER, SCORE, and REVIEW evidence records into iteration snapshots.

    Each iteration is a dict with keys:
        label (str): Human-readable timestamp label
        timestamp (datetime): For sorting
        artifacts (list[str]): PNG paths from the RENDER record
        score (ScoreSnapshot | None): From a nearby SCORE record
        issues (list[Issue]): From a nearby REVIEW record
        grade (str): Letter grade or "?"
    """
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

        # Find the closest SCORE record within 60 seconds of this render
        score_snap: ScoreSnapshot | None = None
        grade = "?"
        for sr in score_records:
            delta = abs((sr.timestamp - ts).total_seconds())
            if delta < 60:
                try:
                    score_snap = ScoreSnapshot.model_validate(sr.details)
                    grade = score_snap.grade
                except Exception:
                    pass
                break

        # Find the closest REVIEW record within 60 seconds
        issues: list[Issue] = []
        for rr in review_records:
            delta = abs((rr.timestamp - ts).total_seconds())
            if delta < 60:
                issues = list(rr.issues)
                break

        iterations.append({
            "label": label,
            "timestamp": ts,
            "artifacts": list(render_rec.artifacts),
            "score": score_snap,
            "issues": issues,
            "grade": grade,
        })

    return iterations


def _open_diff_dialog(board_dir: Path) -> None:
    """Open a dialog comparing two render iterations side by side."""
    from nicegui import ui

    iterations = _collect_iterations(board_dir)

    with ui.dialog() as dlg, ui.card().classes("w-full max-w-6xl"):
        ui.label("Iteration Compare").classes("text-h5 font-bold q-mb-md")

        if len(iterations) < 2:
            ui.label("Not enough iterations to compare").classes(
                "text-grey text-subtitle1"
            )
            ui.button("Close", on_click=dlg.close)
            dlg.open()
            return

        labels = [str(it["label"]) for it in iterations]

        # Selectors
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

        # Container that rebuilds when selections change
        compare_container = ui.column().classes("w-full")

        def _rebuild_comparison() -> None:
            compare_container.clear()

            before_label = before_select.value
            after_label = after_select.value
            if not before_label or not after_label:
                return

            before_it = next(
                (it for it in iterations if it["label"] == before_label), None
            )
            after_it = next(
                (it for it in iterations if it["label"] == after_label), None
            )
            if not before_it or not after_it:
                return

            with compare_container:
                # Side-by-side images
                _render_side_by_side(
                    board_dir, before_it, after_it, before_label, after_label
                )

                ui.separator().classes("q-my-md")

                # Change summary
                _render_change_summary(before_it, after_it)

        _rebuild_comparison()
        before_select.on_value_change(lambda _: _rebuild_comparison())
        after_select.on_value_change(lambda _: _rebuild_comparison())

        ui.button("Close", on_click=dlg.close).classes("q-mt-md")

    dlg.open()


def _render_side_by_side(
    board_dir: Path,
    before_it: dict[str, object],
    after_it: dict[str, object],
    before_label: str,
    after_label: str,
) -> None:
    """Render before/after images side by side in the current UI context."""
    from pathlib import Path as _Path

    from nicegui import ui

    before_arts: list[str] = before_it.get("artifacts", [])  # type: ignore[assignment]
    after_arts: list[str] = after_it.get("artifacts", [])  # type: ignore[assignment]

    with ui.row().classes("w-full gap-4"):
        # Before column
        with ui.column().classes("flex-1"):
            ui.label(f"Before — {before_label}").classes("text-subtitle1 font-bold")
            if before_arts:
                for art_path_str in before_arts:
                    art_path = _Path(art_path_str)
                    if art_path.exists() and art_path.suffix.lower() == ".png":
                        _ensure_static(art_path)
                        url = f"/static/images/{art_path.name}"
                        ui.image(url).classes("w-full max-h-80 object-contain")
                        ui.label(art_path.stem).classes("text-caption text-center")
            else:
                ui.label("No render artifacts").classes("text-grey")

        # After column
        with ui.column().classes("flex-1"):
            ui.label(f"After — {after_label}").classes("text-subtitle1 font-bold")
            if after_arts:
                for art_path_str in after_arts:
                    art_path = _Path(art_path_str)
                    if art_path.exists() and art_path.suffix.lower() == ".png":
                        _ensure_static(art_path)
                        url = f"/static/images/{art_path.name}"
                        ui.image(url).classes("w-full max-h-80 object-contain")
                        ui.label(art_path.stem).classes("text-caption text-center")
            else:
                ui.label("No render artifacts").classes("text-grey")


def _render_change_summary(
    before_it: dict[str, object],
    after_it: dict[str, object],
) -> None:
    """Render the change summary section: score delta, grade change, issue diff."""
    from nicegui import ui

    ui.label("Change Summary").classes("text-subtitle1 font-bold q-mb-sm")

    before_score: ScoreSnapshot | None = before_it.get("score")  # type: ignore[assignment]
    after_score: ScoreSnapshot | None = after_it.get("score")  # type: ignore[assignment]
    before_grade: str = str(before_it.get("grade", "?"))
    after_grade: str = str(after_it.get("grade", "?"))
    before_issues: list[Issue] = before_it.get("issues", [])  # type: ignore[assignment]
    after_issues: list[Issue] = after_it.get("issues", [])  # type: ignore[assignment]

    # Grade change
    if before_grade != after_grade:
        grade_colors = {"A": "green", "B": "light-green", "C": "yellow", "D": "orange", "F": "red"}
        before_color = grade_colors.get(before_grade, "grey")
        after_color = grade_colors.get(after_grade, "grey")
        with ui.row().classes("items-center gap-2"):
            ui.label("Grade:").classes("font-bold")
            ui.badge(before_grade, color=before_color)
            ui.icon("arrow_forward")
            ui.badge(after_grade, color=after_color)
    else:
        with ui.row().classes("items-center gap-2"):
            ui.label("Grade:").classes("font-bold")
            ui.label(f"{before_grade} (unchanged)").classes("text-grey")

    # Score delta
    if before_score and after_score:
        delta = after_score.overall_score - before_score.overall_score
        sign = "+" if delta >= 0 else ""
        delta_color = "green" if delta >= 0 else "red"
        with ui.row().classes("items-center gap-2"):
            ui.label("Score:").classes("font-bold")
            ui.label(f"{before_score.overall_score:.3f}").classes("font-mono")
            ui.icon("arrow_forward")
            ui.label(f"{after_score.overall_score:.3f}").classes("font-mono")
            ui.label(f"({sign}{delta:.3f})").classes(f"font-mono text-{delta_color}")

        # Per-dimension deltas for dimensions that changed
        changed_dims: list[tuple[str, float, float]] = []
        all_dims = set(before_score.breakdown.keys()) | set(after_score.breakdown.keys())
        for dim in sorted(all_dims):
            bv = before_score.breakdown.get(dim, 0.0)
            av = after_score.breakdown.get(dim, 0.0)
            if abs(av - bv) > 0.001:
                changed_dims.append((dim, bv, av))

        if changed_dims:
            with ui.expansion("Dimension changes", icon="analytics").classes("w-full"):
                for dim, bv, av in changed_dims:
                    d = av - bv
                    s = "+" if d >= 0 else ""
                    dc = "green" if d >= 0 else "red"
                    with ui.row().classes("justify-between"):
                        ui.label(dim).classes("text-caption")
                        ui.label(f"{bv:.3f} -> {av:.3f} ({s}{d:.3f})").classes(
                            f"text-caption font-mono text-{dc}"
                        )
    elif before_score or after_score:
        ui.label("Score data available for only one iteration").classes("text-grey")
    else:
        ui.label("No score data for either iteration").classes("text-grey")

    # Issue diff
    before_descs = {i.description for i in before_issues}
    after_descs = {i.description for i in after_issues}
    resolved = before_descs - after_descs
    new_issues = after_descs - before_descs

    with ui.row().classes("items-center gap-4 q-mt-sm"):
        ui.label("Issues:").classes("font-bold")
        ui.label(f"{len(before_issues)} before").classes("text-caption")
        ui.icon("arrow_forward")
        ui.label(f"{len(after_issues)} after").classes("text-caption")

    if resolved:
        with ui.expansion(
            f"Resolved ({len(resolved)})", icon="check_circle"
        ).classes("w-full"):
            for desc in sorted(resolved):
                ui.label(f"  {desc}").classes("text-caption text-green")

    if new_issues:
        with ui.expansion(
            f"New issues ({len(new_issues)})", icon="warning"
        ).classes("w-full"):
            for desc in sorted(new_issues):
                ui.label(f"  {desc}").classes("text-caption text-red")

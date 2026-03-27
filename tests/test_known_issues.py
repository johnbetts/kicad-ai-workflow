"""Tests for the known issues injector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.evidence.known_issues import load_known_issues

if TYPE_CHECKING:
    from pathlib import Path


def _setup_project(tmp_path: Path) -> Path:
    """Create a minimal project structure with known issues files."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "known_issues.md").write_text(
        "# Known Issues\n"
        "| ID | Status | Description | Guard |\n"
        "|---|---|---|---|\n"
        "| KI-001 | FIXED | Off-board components | check_bounds |\n"
        "| KI-005 | OPEN | Decoupling too far from IC | pcb general |\n"
        "| KI-010 | OPEN | Connector placement | all stages |\n",
        encoding="utf-8",
    )

    pcb_review = tmp_path / ".pcb-review"
    pcb_review.mkdir()
    (pcb_review / "lessons.md").write_text(
        "## What Works\n- 4-view rendering\n- Edge-to-edge distance\n",
        encoding="utf-8",
    )
    (pcb_review / "failed_approaches.jsonl").write_text(
        '{"board": "mcu", "description": "Moving C1 to edge", "reason": "collision"}\n'
        '{"board": "relay", "description": "Rotating relay 90deg", "reason": "pad overlap"}\n',
        encoding="utf-8",
    )

    return tmp_path


class TestLoadKnownIssues:
    def test_loads_all_sources(self, tmp_path: Path) -> None:
        root = _setup_project(tmp_path)
        result = load_known_issues(project_root=root)
        assert "KNOWN ISSUES" in result
        assert "LESSONS LEARNED" in result
        assert "FAILED APPROACHES" in result

    def test_includes_open_issues(self, tmp_path: Path) -> None:
        root = _setup_project(tmp_path)
        result = load_known_issues(project_root=root)
        assert "KI-005" in result
        assert "KI-010" in result

    def test_filters_failed_approaches_by_board(self, tmp_path: Path) -> None:
        root = _setup_project(tmp_path)
        result = load_known_issues(project_root=root, board="mcu")
        assert "Moving C1 to edge" in result
        assert "Rotating relay 90deg" not in result

    def test_missing_files_graceful(self, tmp_path: Path) -> None:
        result = load_known_issues(project_root=tmp_path)
        assert result == "" or "KNOWN ISSUES" not in result

    def test_output_has_expected_headers(self, tmp_path: Path) -> None:
        root = _setup_project(tmp_path)
        result = load_known_issues(project_root=root)
        assert "## KNOWN ISSUES" in result
        assert "## LESSONS LEARNED" in result
        assert "## FAILED APPROACHES" in result

    def test_board_specific_lessons(self, tmp_path: Path) -> None:
        root = _setup_project(tmp_path)
        board_dir = root / "output" / "train_mcu"
        board_dir.mkdir(parents=True)
        pcb_review = board_dir / ".pcb-review"
        pcb_review.mkdir()
        (pcb_review / "lessons.md").write_text(
            "## MCU-specific\n- Keep antenna clear\n",
            encoding="utf-8",
        )
        result = load_known_issues(project_root=root, board_dir=board_dir)
        assert "MCU-specific" in result
        assert "this board" in result

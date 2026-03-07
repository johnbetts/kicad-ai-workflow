"""Tests for KiCad-CLI DRC runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.exceptions import DRCError, ValidationError
from kicad_pipeline.validation.kicad_drc import (
    DRCReport,
    categorize_violations,
    parse_drc_json,
    run_drc,
    summarize_drc,
)

if TYPE_CHECKING:
    from pathlib import Path

# Fixture: realistic kicad-cli DRC JSON output.
DRC_JSON_FIXTURE = json.dumps({
    "$schema_version": 1,
    "source": "test_board.kicad_pcb",
    "coordinate_units": "mm",
    "violations": [
        {
            "type": "clearance",
            "severity": "error",
            "description": "Clearance violation (0.15mm < 0.2mm)",
            "excluded": False,
            "items": [
                {
                    "description": "Track on F.Cu",
                    "uuid": "track-1",
                    "pos": {"x": 10.0, "y": 20.0},
                },
                {
                    "description": "Pad R1-1 on F.Cu",
                    "uuid": "pad-r1-1",
                    "pos": {"x": 10.2, "y": 20.0},
                },
            ],
        },
        {
            "type": "silk_over_copper",
            "severity": "warning",
            "description": "Silk text over copper pad",
            "excluded": False,
            "items": [
                {
                    "description": "Text 'R1' on F.SilkS",
                    "uuid": "silk-r1",
                    "pos": {"x": 15.0, "y": 25.0},
                },
            ],
        },
        {
            "type": "copper_edge_clearance",
            "severity": "error",
            "description": "Copper to board edge clearance violation",
            "excluded": True,
            "items": [],
        },
    ],
    "unconnected_items": [
        {
            "type": "unconnected_items",
            "severity": "error",
            "description": "Unconnected: GND pad U1-4",
            "excluded": False,
            "items": [
                {
                    "description": "Pad U1-4",
                    "uuid": "pad-u1-4",
                    "pos": {"x": 30.0, "y": 40.0},
                },
            ],
        },
    ],
})


class TestParseDrcJson:
    def test_parses_violations(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        assert len(report.violations) == 3
        assert report.violations[0].type == "clearance"
        assert report.violations[0].severity == "error"
        assert len(report.violations[0].items) == 2

    def test_parses_unconnected(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        assert len(report.unconnected) == 1
        assert report.unconnected[0].description == "Unconnected: GND pad U1-4"

    def test_error_count_excludes_excluded(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        # 1 clearance error (not excluded) + silk is warning = 1 error.
        assert report.error_count == 1

    def test_warning_count(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        assert report.warning_count == 1

    def test_unconnected_count(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        assert report.unconnected_count == 1

    def test_passed_false_with_errors(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        assert report.passed is False

    def test_passed_true_when_clean(self) -> None:
        clean = json.dumps({
            "violations": [],
            "unconnected_items": [],
        })
        report = parse_drc_json(clean)
        assert report.passed is True

    def test_position_parsing(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        item = report.violations[0].items[0]
        assert item.pos is not None
        assert item.pos.x == pytest.approx(10.0)
        assert item.pos.y == pytest.approx(20.0)

    def test_schema_version(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        assert report.schema_version == 1

    def test_invalid_json(self) -> None:
        with pytest.raises(ValidationError, match="Failed to parse"):
            parse_drc_json("not json at all")


class TestSummarizeDrc:
    def test_summary_contains_status(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        summary = summarize_drc(report)
        assert "FAILED" in summary

    def test_summary_contains_counts(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        summary = summarize_drc(report)
        assert "Errors:" in summary
        assert "Warnings:" in summary
        assert "Unconnected:" in summary

    def test_summary_passed(self) -> None:
        clean = json.dumps({"violations": [], "unconnected_items": []})
        report = parse_drc_json(clean)
        summary = summarize_drc(report)
        assert "PASSED" in summary


class TestCategorizeViolations:
    def test_auto_fixable(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        cats = categorize_violations(report)
        auto_types = {v.type for v in cats["auto_fixable"]}
        assert "silk_over_copper" in auto_types

    def test_manual(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        cats = categorize_violations(report)
        manual_types = {v.type for v in cats["manual"]}
        assert "clearance" in manual_types

    def test_excluded_not_categorized(self) -> None:
        report = parse_drc_json(DRC_JSON_FIXTURE)
        cats = categorize_violations(report)
        all_types = {v.type for v in cats["auto_fixable"] + cats["manual"]}
        # copper_edge_clearance was excluded — should not appear.
        assert "copper_edge_clearance" not in all_types


class TestRunDrc:
    def test_missing_pcb_file(self) -> None:
        with pytest.raises(DRCError, match="not found"):
            run_drc("/nonexistent/board.kicad_pcb")

    def test_kicad_cli_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pcb = tmp_path / "test.kicad_pcb"
        pcb.write_text("(kicad_pcb)")
        monkeypatch.delenv("KICAD_CLI", raising=False)
        monkeypatch.setattr(
            "kicad_pipeline.validation.kicad_drc._KICAD_CLI_DEFAULT",
            "/nonexistent/kicad-cli",
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            with pytest.raises(DRCError, match="Cannot find"):
                run_drc(str(pcb))


class TestDRCReport:
    def test_empty_report_passes(self) -> None:
        report = DRCReport()
        assert report.passed is True
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.unconnected_count == 0

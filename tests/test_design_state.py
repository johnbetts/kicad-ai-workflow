"""Tests for design_state — persistent conversational workflow state."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — used at runtime

import pytest

from kicad_pipeline.design_state import (
    PHASES,
    append_design_decision,
    get_current_phase,
    list_research,
    lookup_research,
    read_project_state,
    save_research,
    search_research,
    write_checklist_results,
    write_drc_history,
    write_parts_selection,
    write_project_readme,
    write_requirements,
)
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
    Recommendation,
)
from kicad_pipeline.parts.jlcpcb_db import JLCPCBPart
from kicad_pipeline.parts.selector import PartSuggestion
from kicad_pipeline.validation.checklist import ChecklistReport, CheckResult, CheckStatus
from kicad_pipeline.validation.drc import DRCReport, DRCViolation, Severity
from kicad_pipeline.validation.drc_interpreter import DRCSuggestion

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Provide a temporary project directory."""
    return tmp_path


def _make_requirements() -> ProjectRequirements:
    """Create a minimal ProjectRequirements for testing."""
    return ProjectRequirements(
        project=ProjectInfo(
            name="Test Board",
            description="A test project",
            author="Tester",
            revision="v0.1",
        ),
        features=(
            FeatureBlock(
                name="Power",
                description="5V to 3.3V regulation",
                components=("U1", "C1", "C2"),
                nets=("+5V", "+3V3", "GND"),
                subcircuits=("ldo",),
            ),
        ),
        components=(
            Component(
                ref="U1",
                value="AMS1117-3.3",
                footprint="SOT-223",
                lcsc="C6186",
                pins=(
                    Pin(number="1", name="GND", pin_type="passive"),
                    Pin(number="2", name="OUT", pin_type="output"),
                    Pin(number="3", name="IN", pin_type="input"),
                ),
            ),
            Component(ref="C1", value="10uF", footprint="C_0805", lcsc="C15850"),
            Component(ref="C2", value="100nF", footprint="C_0805", lcsc="C49678"),
            Component(ref="R1", value="10k", footprint="R_0805", lcsc="C17414"),
        ),
        nets=(
            Net(
                name="+3V3",
                connections=(
                    NetConnection(ref="U1", pin="2"),
                    NetConnection(ref="C2", pin="1"),
                ),
            ),
            Net(
                name="GND",
                connections=(
                    NetConnection(ref="U1", pin="1"),
                    NetConnection(ref="C1", pin="2"),
                    NetConnection(ref="C2", pin="2"),
                ),
            ),
        ),
        mechanical=MechanicalConstraints(
            board_width_mm=50.0,
            board_height_mm=30.0,
            board_template="generic",
            notes="Compact layout",
        ),
        power_budget=PowerBudget(
            rails=(
                PowerRail(name="+5V", voltage=5.0, current_ma=500.0, source_ref="J1"),
                PowerRail(name="+3V3", voltage=3.3, current_ma=300.0, source_ref="U1"),
            ),
            total_current_ma=500.0,
            notes=("ESP32 peak draw ~300mA",),
        ),
        recommendations=(
            Recommendation(
                severity="info",
                category="power",
                message="Consider bulk cap on input",
                affected_refs=("C1",),
            ),
        ),
    )


def _make_jlcpcb_part(
    lcsc: str = "C6186",
    mfr: str = "AMS",
    mfr_part: str = "AMS1117-3.3",
    package: str = "SOT-223",
    basic: bool = True,
    stock: int = 23000,
    price: float = 0.03,
) -> JLCPCBPart:
    return JLCPCBPart(
        lcsc=lcsc,
        mfr=mfr,
        mfr_part=mfr_part,
        description="3.3V LDO",
        package=package,
        category="Power",
        subcategory="LDO",
        solder_joints=4,
        stock=stock,
        price=price,
        basic=basic,
    )


def _make_suggestion(
    ref: str = "U1",
    value: str = "AMS1117-3.3",
    quality: str = "exact",
) -> PartSuggestion:
    part = _make_jlcpcb_part()
    return PartSuggestion(
        component_ref=ref,
        component_value=value,
        candidates=(part,),
        preferred=part,
        match_quality=quality,
        notes="",
    )


def _make_drc_report(errors: int = 0, warnings: int = 2) -> DRCReport:
    violations: list[DRCViolation] = []
    for i in range(errors):
        violations.append(
            DRCViolation(
                rule="clearance",
                message=f"Clearance error {i + 1}",
                severity=Severity.ERROR,
                ref=f"R{i + 1}",
            )
        )
    for i in range(warnings):
        violations.append(
            DRCViolation(
                rule="silk_overlap",
                message=f"Silk overlap {i + 1}",
                severity=Severity.WARNING,
                ref=f"C{i + 1}",
            )
        )
    return DRCReport(violations=tuple(violations))


def _make_suggestions_list() -> list[DRCSuggestion]:
    return [
        DRCSuggestion(
            violation_type="clearance",
            severity="error",
            auto_fixable=False,
            description="Track too close to pad",
            fix_action="Increase clearance in KiCad",
            affected_refs=("R1",),
        ),
        DRCSuggestion(
            violation_type="silk_overlap",
            severity="warning",
            auto_fixable=True,
            description="Silk text overlaps courtyard",
            fix_action=None,
            affected_refs=("C1",),
        ),
    ]


def _make_checklist_report() -> ChecklistReport:
    return ChecklistReport(
        results=(
            CheckResult(
                category="Electrical",
                name="Net connectivity",
                status=CheckStatus.PASS,
                message="All nets connected",
            ),
            CheckResult(
                category="Electrical",
                name="Decoupling caps",
                status=CheckStatus.WARN,
                message="U2 missing 100nF cap",
                details="Add cap within 5mm of VCC pin",
            ),
            CheckResult(
                category="Manufacturing",
                name="Min trace width",
                status=CheckStatus.PASS,
                message="All traces >= 0.15mm",
            ),
            CheckResult(
                category="Manufacturing",
                name="Drill sizes",
                status=CheckStatus.FAIL,
                message="Via drill 0.1mm below JLCPCB minimum",
                details="Minimum via drill is 0.3mm",
            ),
        )
    )


# ---------------------------------------------------------------------------
# Tests — Writers
# ---------------------------------------------------------------------------


class TestWriteProjectReadme:
    def test_creates_design_dir(self, project_dir: Path) -> None:
        write_project_readme(project_dir, "My Board", "test desc", "Project Setup")
        assert (project_dir / "design").is_dir()

    def test_contains_project_name(self, project_dir: Path) -> None:
        path = write_project_readme(project_dir, "Relay Board", "4ch relay", "PCB Layout")
        content = path.read_text()
        assert "# Project: Relay Board" in content
        assert "**Phase**: PCB Layout" in content

    def test_checkmarks_reflect_phase(self, project_dir: Path) -> None:
        path = write_project_readme(
            project_dir, "B", "d", "Parts Selection", component_count=5, basic_count=3
        )
        content = path.read_text()
        assert "[x] Project Setup" in content
        assert "[x] Requirements Gathering" in content
        assert "[x] Parts Selection (5 components, 3 JLCPCB basic)" in content
        assert "[ ] Schematic Generation" in content

    def test_overwrites_on_update(self, project_dir: Path) -> None:
        write_project_readme(project_dir, "B", "d", "Project Setup")
        write_project_readme(project_dir, "B", "d", "DRC Iteration")
        content = (project_dir / "design" / "README.md").read_text()
        assert "[x] DRC Iteration" in content


class TestWriteRequirements:
    def test_writes_components_table(self, project_dir: Path) -> None:
        reqs = _make_requirements()
        path = write_requirements(project_dir, reqs)
        content = path.read_text()
        assert "| U1 | AMS1117-3.3 | SOT-223 | C6186 |" in content
        assert "| R1 | 10k | R_0805 | C17414 |" in content

    def test_writes_feature_blocks(self, project_dir: Path) -> None:
        reqs = _make_requirements()
        path = write_requirements(project_dir, reqs)
        content = path.read_text()
        assert "### Power" in content
        assert "5V to 3.3V regulation" in content

    def test_writes_mechanical(self, project_dir: Path) -> None:
        reqs = _make_requirements()
        path = write_requirements(project_dir, reqs)
        content = path.read_text()
        assert "50.0 x 30.0 mm" in content
        assert "Template: generic" in content

    def test_writes_nets(self, project_dir: Path) -> None:
        reqs = _make_requirements()
        path = write_requirements(project_dir, reqs)
        content = path.read_text()
        assert "**+3V3**: U1.2, C2.1" in content
        assert "**GND**: U1.1, C1.2, C2.2" in content

    def test_writes_power_budget(self, project_dir: Path) -> None:
        reqs = _make_requirements()
        path = write_requirements(project_dir, reqs)
        content = path.read_text()
        assert "Total current: 500.0 mA" in content
        assert "+5V: 5.0V, 500.0mA" in content

    def test_writes_recommendations(self, project_dir: Path) -> None:
        reqs = _make_requirements()
        path = write_requirements(project_dir, reqs)
        content = path.read_text()
        assert "[info] (power) Consider bulk cap on input" in content


class TestWritePartsSelection:
    def test_writes_markdown_table(self, project_dir: Path) -> None:
        parts = {"U1": _make_jlcpcb_part()}
        suggestions = {"U1": _make_suggestion()}
        md_path, json_path = write_parts_selection(project_dir, suggestions, parts)
        content = md_path.read_text()
        assert "C6186" in content
        assert "SOT-223" in content
        assert "Yes" in content  # basic

    def test_writes_valid_json(self, project_dir: Path) -> None:
        parts = {"U1": _make_jlcpcb_part(), "R1": _make_jlcpcb_part(lcsc="C17414")}
        suggestions = {"U1": _make_suggestion()}
        _, json_path = write_parts_selection(project_dir, suggestions, parts)
        import json

        data = json.loads(json_path.read_text())
        assert "U1" in data
        assert data["U1"]["lcsc"] == "C6186"
        assert data["U1"]["basic"] is True

    def test_match_quality_section(self, project_dir: Path) -> None:
        parts = {"U1": _make_jlcpcb_part()}
        suggestions = {"U1": _make_suggestion(quality="close")}
        md_path, _ = write_parts_selection(project_dir, suggestions, parts)
        content = md_path.read_text()
        assert "close match" in content


class TestAppendDesignDecision:
    def test_creates_file_on_first_call(self, project_dir: Path) -> None:
        path = append_design_decision(
            project_dir,
            phase="Parts Selection",
            decision="Use AMS1117-3.3",
            rationale="Basic part, cheap, SOT-223",
        )
        content = path.read_text()
        assert "# Design Decisions" in content
        assert "Use AMS1117-3.3" in content

    def test_appends_multiple_decisions(self, project_dir: Path) -> None:
        append_design_decision(project_dir, "Parts", "Decision 1", "Reason 1")
        append_design_decision(project_dir, "PCB", "Decision 2", "Reason 2")
        content = (project_dir / "design" / "design_decisions.md").read_text()
        assert "Decision 1" in content
        assert "Decision 2" in content

    def test_includes_alternatives(self, project_dir: Path) -> None:
        path = append_design_decision(
            project_dir,
            phase="Parts",
            decision="Use AMS1117",
            rationale="Cheap",
            alternatives=("AP2112K", "ME6211"),
        )
        content = path.read_text()
        assert "AP2112K; ME6211" in content


class TestWriteDrcHistory:
    def test_writes_iteration(self, project_dir: Path) -> None:
        report = _make_drc_report(errors=3, warnings=2)
        suggestions = _make_suggestions_list()
        path = write_drc_history(project_dir, report, suggestions, fixes_applied=1, iteration=1)
        content = path.read_text()
        assert "## Iteration 1" in content
        assert "**Status**: FAIL" in content
        assert "**Errors**: 3" in content
        assert "**Warnings**: 2" in content

    def test_appends_iterations(self, project_dir: Path) -> None:
        report1 = _make_drc_report(errors=5, warnings=3)
        report2 = _make_drc_report(errors=0, warnings=1)
        suggestions = _make_suggestions_list()
        write_drc_history(project_dir, report1, suggestions, fixes_applied=0, iteration=1)
        write_drc_history(project_dir, report2, suggestions, fixes_applied=3, iteration=2)
        content = (project_dir / "design" / "drc_history.md").read_text()
        assert "## Iteration 1" in content
        assert "## Iteration 2" in content
        assert "**Status**: PASS" in content

    def test_violation_type_counts(self, project_dir: Path) -> None:
        report = _make_drc_report(errors=2, warnings=2)
        path = write_drc_history(project_dir, report, [], fixes_applied=0, iteration=1)
        content = path.read_text()
        assert "clearance: 2" in content
        assert "silk_overlap: 2" in content

    def test_manual_fix_suggestions(self, project_dir: Path) -> None:
        report = _make_drc_report(errors=1)
        suggestions = _make_suggestions_list()
        path = write_drc_history(project_dir, report, suggestions, fixes_applied=0, iteration=1)
        content = path.read_text()
        assert "Track too close to pad" in content
        assert "Increase clearance in KiCad" in content


class TestWriteChecklistResults:
    def test_writes_pass_fail(self, project_dir: Path) -> None:
        report = _make_checklist_report()
        path = write_checklist_results(project_dir, report)
        content = path.read_text()
        assert "FAIL (1 failures)" in content

    def test_groups_by_category(self, project_dir: Path) -> None:
        report = _make_checklist_report()
        path = write_checklist_results(project_dir, report)
        content = path.read_text()
        assert "## Electrical" in content
        assert "## Manufacturing" in content

    def test_status_icons(self, project_dir: Path) -> None:
        report = _make_checklist_report()
        path = write_checklist_results(project_dir, report)
        content = path.read_text()
        assert "[+]" in content  # pass
        assert "[~]" in content  # warn
        assert "[-]" in content  # fail

    def test_includes_details(self, project_dir: Path) -> None:
        report = _make_checklist_report()
        path = write_checklist_results(project_dir, report)
        content = path.read_text()
        assert "Minimum via drill is 0.3mm" in content


# ---------------------------------------------------------------------------
# Tests — Readers
# ---------------------------------------------------------------------------


class TestResearchCache:
    def test_save_and_lookup(self, project_dir: Path) -> None:
        save_research(
            project_dir,
            topic="ESP32-S3 Pinout",
            content="GPIO0-GPIO21 available.\nStrapping pins: GPIO0, GPIO45, GPIO46.",
            source="https://docs.espressif.com/esp32-s3",
            tags=("pinout", "esp32"),
        )
        result = lookup_research(project_dir, "ESP32-S3 Pinout")
        assert result is not None
        assert "GPIO0-GPIO21" in result
        assert "**Source**: https://docs.espressif.com" in result
        assert "**Tags**: pinout, esp32" in result

    def test_lookup_miss_returns_none(self, project_dir: Path) -> None:
        assert lookup_research(project_dir, "nonexistent topic") is None

    def test_lookup_no_design_dir(self, project_dir: Path) -> None:
        assert lookup_research(project_dir, "anything") is None

    def test_slugify_normalizes(self, project_dir: Path) -> None:
        save_research(project_dir, "AMS1117-3.3 Thermal Limits", content="Tj max 125°C")
        # Should be accessible with same topic
        assert lookup_research(project_dir, "AMS1117-3.3 Thermal Limits") is not None
        # Check file is reasonably named
        files = list_research(project_dir)
        assert len(files) == 1
        assert "ams1117" in files[0]

    def test_search_keyword(self, project_dir: Path) -> None:
        save_research(project_dir, "ESP32 Pinout", content="GPIO0 is a strapping pin")
        save_research(project_dir, "AMS1117 Datasheet", content="SOT-223, max 800mA")
        save_research(project_dir, "Relay Driver", content="Use ULN2003 for relay coils")
        results = search_research(project_dir, "strapping")
        assert len(results) == 1
        assert "esp32-pinout.md" in results

    def test_search_case_insensitive(self, project_dir: Path) -> None:
        save_research(project_dir, "ESP32 Info", content="WiFi + BLE combo")
        results = search_research(project_dir, "WIFI")
        assert len(results) == 1

    def test_search_empty_dir(self, project_dir: Path) -> None:
        assert search_research(project_dir, "anything") == {}

    def test_list_research_empty(self, project_dir: Path) -> None:
        assert list_research(project_dir) == ()

    def test_list_research_populated(self, project_dir: Path) -> None:
        save_research(project_dir, "Topic A", content="data a")
        save_research(project_dir, "Topic B", content="data b")
        files = list_research(project_dir)
        assert len(files) == 2

    def test_overwrite_existing_research(self, project_dir: Path) -> None:
        save_research(project_dir, "ESP32 Pinout", content="v1: basic info")
        save_research(project_dir, "ESP32 Pinout", content="v2: updated with GPIO map")
        result = lookup_research(project_dir, "ESP32 Pinout")
        assert result is not None
        assert "v2" in result
        assert "v1" not in result

    def test_read_project_state_includes_research(self, project_dir: Path) -> None:
        write_project_readme(project_dir, "B", "d", "Project Setup")
        save_research(project_dir, "ESP32 Pinout", content="GPIO data")
        state = read_project_state(project_dir)
        assert "README.md" in state
        assert "research/esp32-pinout.md" in state
        assert "GPIO data" in state["research/esp32-pinout.md"]


class TestReadProjectState:
    def test_empty_when_no_design_dir(self, project_dir: Path) -> None:
        assert read_project_state(project_dir) == {}

    def test_reads_all_files(self, project_dir: Path) -> None:
        reqs = _make_requirements()
        write_project_readme(project_dir, "B", "d", "Parts Selection")
        write_requirements(project_dir, reqs)
        state = read_project_state(project_dir)
        assert "README.md" in state
        assert "requirements.md" in state
        assert "Project: B" in state["README.md"]

    def test_includes_json_files(self, project_dir: Path) -> None:
        parts = {"U1": _make_jlcpcb_part()}
        suggestions = {"U1": _make_suggestion()}
        write_parts_selection(project_dir, suggestions, parts)
        state = read_project_state(project_dir)
        assert "parts_selection.json" in state
        assert "parts_selection.md" in state

    def test_ignores_non_md_json(self, project_dir: Path) -> None:
        d = project_dir / "design"
        d.mkdir()
        (d / "notes.txt").write_text("not included")
        (d / "README.md").write_text("included")
        state = read_project_state(project_dir)
        assert "README.md" in state
        assert "notes.txt" not in state


class TestGetCurrentPhase:
    def test_default_when_no_dir(self, project_dir: Path) -> None:
        assert get_current_phase(project_dir) == "Project Setup"

    def test_detects_requirements(self, project_dir: Path) -> None:
        d = project_dir / "design"
        d.mkdir()
        (d / "README.md").write_text("x")
        (d / "requirements.md").write_text("x")
        assert get_current_phase(project_dir) == "Requirements Gathering"

    def test_detects_parts(self, project_dir: Path) -> None:
        d = project_dir / "design"
        d.mkdir()
        (d / "README.md").write_text("x")
        (d / "requirements.md").write_text("x")
        (d / "parts_selection.json").write_text("{}")
        assert get_current_phase(project_dir) == "Parts Selection"

    def test_detects_drc(self, project_dir: Path) -> None:
        d = project_dir / "design"
        d.mkdir()
        (d / "drc_history.md").write_text("x")
        assert get_current_phase(project_dir) == "DRC Iteration"

    def test_detects_validation(self, project_dir: Path) -> None:
        d = project_dir / "design"
        d.mkdir()
        (d / "checklist.md").write_text("x")
        assert get_current_phase(project_dir) == "Validation"

    def test_phases_tuple_is_ordered(self) -> None:
        assert PHASES[0] == "Project Setup"
        assert PHASES[-1] == "Design Review"
        assert len(PHASES) == 10

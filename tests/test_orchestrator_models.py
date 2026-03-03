"""Tests for orchestrator data models."""

from __future__ import annotations

from kicad_pipeline.orchestrator.models import (
    DEFAULT_PACKAGE_STRATEGIES,
    STAGE_ORDER,
    PackageStrategy,
    ProjectManifest,
    RevisionRecord,
    StageId,
    StageRecord,
    StageState,
    VariantRecord,
    VariantStatus,
    default_stages,
    get_strategy_by_name,
)

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestVariantStatus:
    def test_all_values(self) -> None:
        assert VariantStatus.DRAFT.value == "draft"
        assert VariantStatus.REVIEWING.value == "reviewing"
        assert VariantStatus.APPROVED.value == "approved"
        assert VariantStatus.RELEASED.value == "released"
        assert VariantStatus.ARCHIVED.value == "archived"

    def test_roundtrip(self) -> None:
        for status in VariantStatus:
            assert VariantStatus(status.value) is status


class TestStageId:
    def test_all_values(self) -> None:
        assert StageId.REQUIREMENTS.value == "requirements"
        assert StageId.SCHEMATIC.value == "schematic"
        assert StageId.PCB.value == "pcb"
        assert StageId.VALIDATION.value == "validation"
        assert StageId.PRODUCTION.value == "production"

    def test_stage_order_contains_all(self) -> None:
        assert set(STAGE_ORDER) == set(StageId)

    def test_stage_order_length(self) -> None:
        assert len(STAGE_ORDER) == 5


class TestStageState:
    def test_all_values(self) -> None:
        assert StageState.PENDING.value == "pending"
        assert StageState.GENERATED.value == "generated"
        assert StageState.REVIEWING.value == "reviewing"
        assert StageState.APPROVED.value == "approved"
        assert StageState.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# PackageStrategy
# ---------------------------------------------------------------------------


class TestPackageStrategy:
    def test_default_values(self) -> None:
        ps = PackageStrategy(name="test")
        assert ps.resistor_package == "0805"
        assert ps.capacitor_package == "0805"
        assert ps.led_package == "0805"
        assert ps.prefer_smd is True
        assert ps.notes == ""

    def test_custom_values(self) -> None:
        ps = PackageStrategy(
            name="compact",
            resistor_package="0402",
            capacitor_package="0402",
            led_package="0402",
            prefer_smd=True,
            notes="Ultra compact",
        )
        assert ps.name == "compact"
        assert ps.resistor_package == "0402"

    def test_through_hole(self) -> None:
        ps = PackageStrategy(
            name="tht",
            resistor_package="Axial_DIN0207",
            capacitor_package="C_Disc_D5.0mm",
            led_package="LED_D3.0mm",
            prefer_smd=False,
        )
        assert ps.prefer_smd is False
        assert ps.resistor_package == "Axial_DIN0207"

    def test_frozen(self) -> None:
        ps = PackageStrategy(name="test")
        try:
            ps.name = "changed"  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# StageRecord
# ---------------------------------------------------------------------------


class TestStageRecord:
    def test_defaults(self) -> None:
        sr = StageRecord(stage=StageId.REQUIREMENTS)
        assert sr.state == StageState.PENDING
        assert sr.generated_at is None
        assert sr.approved_at is None
        assert sr.generation_count == 0
        assert sr.notes == ()

    def test_with_values(self) -> None:
        sr = StageRecord(
            stage=StageId.PCB,
            state=StageState.GENERATED,
            generated_at="2026-03-03T10:00:00",
            generation_count=2,
            notes=("widened power traces",),
        )
        assert sr.stage == StageId.PCB
        assert sr.generation_count == 2
        assert len(sr.notes) == 1


# ---------------------------------------------------------------------------
# RevisionRecord
# ---------------------------------------------------------------------------


class TestRevisionRecord:
    def test_minimal(self) -> None:
        rr = RevisionRecord(
            number=1,
            created_at="2026-03-03T10:00:00",
            git_tag="standard-0805/rev1",
            commit_hash="abc123",
        )
        assert rr.number == 1
        assert rr.sent_to_fab is False
        assert rr.fab_order_id is None

    def test_with_fab_info(self) -> None:
        rr = RevisionRecord(
            number=3,
            created_at="2026-03-03T12:00:00",
            git_tag="standard-0805/rev3",
            commit_hash="def456",
            notes="Final production revision",
            sent_to_fab=True,
            fab_order_id="JLCPCB-12345",
        )
        assert rr.sent_to_fab is True
        assert rr.fab_order_id == "JLCPCB-12345"


# ---------------------------------------------------------------------------
# VariantRecord
# ---------------------------------------------------------------------------


class TestVariantRecord:
    def test_minimal(self) -> None:
        vr = VariantRecord(
            name="standard-0805",
            display_name="Standard 0805",
            description="Standard size passives",
            status=VariantStatus.DRAFT,
            package_strategy=PackageStrategy(name="0805"),
        )
        assert vr.name == "standard-0805"
        assert vr.stages == ()
        assert vr.revisions == ()
        assert vr.released_tag is None

    def test_with_stages_and_revisions(self) -> None:
        stages = default_stages()
        vr = VariantRecord(
            name="compact-0603",
            display_name="Compact 0603",
            description="Compact passives",
            status=VariantStatus.REVIEWING,
            package_strategy=PackageStrategy(name="0603"),
            stages=stages,
            revisions=(
                RevisionRecord(
                    number=1,
                    created_at="2026-03-03T10:00:00",
                    git_tag="compact-0603/rev1",
                    commit_hash="abc",
                ),
            ),
        )
        assert len(vr.stages) == 5
        assert len(vr.revisions) == 1


# ---------------------------------------------------------------------------
# ProjectManifest
# ---------------------------------------------------------------------------


class TestProjectManifest:
    def test_defaults(self) -> None:
        pm = ProjectManifest()
        assert pm.schema_version == 1
        assert pm.project_name == ""
        assert pm.variants == ()
        assert pm.active_variant is None

    def test_with_data(self) -> None:
        pm = ProjectManifest(
            project_name="my-board",
            description="Test board",
            original_spec="spec.md",
            created_at="2026-03-03T10:00:00",
            updated_at="2026-03-03T10:00:00",
            active_variant="standard-0805",
            variants=(
                VariantRecord(
                    name="standard-0805",
                    display_name="Standard 0805",
                    description="Standard",
                    status=VariantStatus.DRAFT,
                    package_strategy=PackageStrategy(name="0805"),
                    stages=default_stages(),
                ),
            ),
        )
        assert pm.project_name == "my-board"
        assert pm.active_variant == "standard-0805"
        assert len(pm.variants) == 1
        assert pm.variants[0].name == "standard-0805"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestDefaultStages:
    def test_returns_all_stages(self) -> None:
        stages = default_stages()
        assert len(stages) == 5
        stage_ids = [s.stage for s in stages]
        assert stage_ids == list(STAGE_ORDER)

    def test_all_pending(self) -> None:
        stages = default_stages()
        for s in stages:
            assert s.state == StageState.PENDING

    def test_returns_new_tuples(self) -> None:
        s1 = default_stages()
        s2 = default_stages()
        assert s1 == s2
        assert s1 is not s2


class TestGetStrategyByName:
    def test_finds_0805(self) -> None:
        s = get_strategy_by_name("0805")
        assert s is not None
        assert s.name == "0805"
        assert s.resistor_package == "0805"

    def test_finds_0603(self) -> None:
        s = get_strategy_by_name("0603")
        assert s is not None
        assert s.resistor_package == "0603"

    def test_finds_through_hole(self) -> None:
        s = get_strategy_by_name("through-hole")
        assert s is not None
        assert s.prefer_smd is False

    def test_returns_none_for_unknown(self) -> None:
        assert get_strategy_by_name("1812") is None


class TestDefaultPackageStrategies:
    def test_has_four_strategies(self) -> None:
        assert len(DEFAULT_PACKAGE_STRATEGIES) == 4

    def test_names_unique(self) -> None:
        names = [s.name for s in DEFAULT_PACKAGE_STRATEGIES]
        assert len(names) == len(set(names))

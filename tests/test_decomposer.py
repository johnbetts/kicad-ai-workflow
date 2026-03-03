"""Tests for kicad_pipeline.requirements.decomposer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.exceptions import ComponentError, RequirementsError
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MCUPinMap,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinAssignment,
    PinFunction,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    Recommendation,
)
from kicad_pipeline.requirements.decomposer import (
    RequirementsBuilder,
    load_requirements,
    requirements_from_dict,
    requirements_to_dict,
    save_requirements,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_component(ref: str = "U1") -> Component:
    """Return a minimal valid component."""
    return Component(
        ref=ref,
        value="ESP32-S3",
        footprint="ESP32-S3-WROOM-1",
        pins=(
            Pin(
                number="1",
                name="VCC",
                pin_type=PinType.POWER_IN,
                function=PinFunction.VCC,
                net="+3V3",
            ),
        ),
    )


def _make_net(name: str = "+3V3", ref: str = "U1", pin: str = "1") -> Net:
    """Return a minimal valid net."""
    return Net(name=name, connections=(NetConnection(ref=ref, pin=pin),))


def _minimal_builder() -> RequirementsBuilder:
    """Return a builder with one component and one net already added."""
    builder = RequirementsBuilder(ProjectInfo(name="test-board"))
    builder.add_component(_make_component("U1"))
    builder.add_net(_make_net("+3V3", "U1", "1"))
    return builder


# ---------------------------------------------------------------------------
# RequirementsBuilder
# ---------------------------------------------------------------------------


def test_builder_add_and_build() -> None:
    """Build minimal requirements with one component and one net."""
    builder = _minimal_builder()
    req = builder.build()

    assert req.project.name == "test-board"
    assert len(req.components) == 1
    assert req.components[0].ref == "U1"
    assert len(req.nets) == 1
    assert req.nets[0].name == "+3V3"


def test_builder_duplicate_ref_raises() -> None:
    """Adding a component with a duplicate ref raises ComponentError."""
    builder = RequirementsBuilder(ProjectInfo(name="dup-test"))
    builder.add_component(_make_component("U1"))

    with pytest.raises(ComponentError, match="U1"):
        builder.add_component(_make_component("U1"))


def test_builder_duplicate_net_raises() -> None:
    """Adding a net with a duplicate name raises RequirementsError."""
    builder = RequirementsBuilder(ProjectInfo(name="net-dup"))
    builder.add_component(_make_component("U1"))
    builder.add_net(_make_net("+3V3", "U1", "1"))

    with pytest.raises(RequirementsError, match=r"\+3V3"):
        builder.add_net(_make_net("+3V3", "U1", "1"))


def test_builder_validate_missing_net_ref() -> None:
    """Net referencing unknown component ref produces a validation error."""
    builder = RequirementsBuilder(ProjectInfo(name="bad-ref"))
    # Do NOT add component "U99", but reference it in a net
    builder.add_component(_make_component("U1"))

    # Bypass add_net duplicate check; insert raw to test validate() directly
    builder._nets.append(
        Net(name="BAD_NET", connections=(NetConnection(ref="U99", pin="1"),))
    )

    errors = builder.validate()
    assert any("U99" in e for e in errors), f"Expected U99 in errors, got: {errors}"


def test_builder_build_with_validation_error_raises() -> None:
    """build() raises RequirementsError when validation fails."""
    builder = RequirementsBuilder(ProjectInfo(name="empty"))
    # No components → validation should fail

    with pytest.raises(RequirementsError):
        builder.build()


def test_builder_set_mechanical() -> None:
    """set_mechanical stores MechanicalConstraints correctly."""
    builder = _minimal_builder()
    mech = MechanicalConstraints(
        board_width_mm=100.0,
        board_height_mm=80.0,
        enclosure="Hammond 1455L",
        mounting_hole_diameter_mm=3.2,
        mounting_hole_positions=((5.0, 5.0), (95.0, 75.0)),
        notes="M3 screws",
    )
    builder.set_mechanical(mech)
    req = builder.build()

    assert req.mechanical is not None
    assert req.mechanical.board_width_mm == pytest.approx(100.0)
    assert req.mechanical.board_height_mm == pytest.approx(80.0)
    assert req.mechanical.enclosure == "Hammond 1455L"
    assert len(req.mechanical.mounting_hole_positions) == 2


def test_builder_add_recommendation() -> None:
    """Recommendations accumulate correctly."""
    builder = _minimal_builder()
    rec1 = Recommendation(
        severity="warning",
        category="power",
        message="Add bulk capacitor near U1",
        affected_refs=("U1",),
    )
    rec2 = Recommendation(
        severity="info",
        category="signal",
        message="SPI traces should be length-matched",
    )
    builder.add_recommendation(rec1)
    builder.add_recommendation(rec2)
    req = builder.build()

    assert len(req.recommendations) == 2
    assert req.recommendations[0].severity == "warning"
    assert req.recommendations[1].category == "signal"


def test_builder_set_pin_map() -> None:
    """set_pin_map stores MCUPinMap."""
    builder = _minimal_builder()
    pm = MCUPinMap(
        mcu_ref="U1",
        assignments=(
            PinAssignment(
                mcu_ref="U1",
                pin_number="IO4",
                pin_name="IO4",
                function=PinFunction.SPI_CLK,
                net="SPI_CLK",
            ),
        ),
        unassigned_gpio=("IO5", "IO6"),
    )
    builder.set_pin_map(pm)
    req = builder.build()

    assert req.pin_map is not None
    assert req.pin_map.mcu_ref == "U1"
    assert len(req.pin_map.assignments) == 1
    assert req.pin_map.assignments[0].function == PinFunction.SPI_CLK
    assert req.pin_map.unassigned_gpio == ("IO5", "IO6")


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


def _make_full_requirements() -> object:
    """Build a ProjectRequirements with all optional fields populated."""
    builder = RequirementsBuilder(
        ProjectInfo(
            name="full-board",
            author="Test Author",
            revision="v1.0",
            description="Full test board",
        )
    )

    comp = Component(
        ref="U1",
        value="ESP32-S3-WROOM-1",
        footprint="ESP32-S3-WROOM-1",
        lcsc="C2913202",
        description="MCU module",
        datasheet="https://example.com/esp32.pdf",
        pins=(
            Pin(
                number="1",
                name="VCC",
                pin_type=PinType.POWER_IN,
                function=PinFunction.VCC,
                net="+3V3",
            ),
        ),
    )
    builder.add_component(comp)
    builder.add_net(_make_net("+3V3", "U1", "1"))
    builder.add_feature(
        FeatureBlock(
            name="MCU",
            description="Main microcontroller",
            components=("U1",),
            nets=("+3V3",),
            subcircuits=("decoupling",),
        )
    )
    builder.add_recommendation(
        Recommendation(
            severity="info",
            category="power",
            message="All good",
            affected_refs=("U1",),
        )
    )
    builder.set_pin_map(
        MCUPinMap(
            mcu_ref="U1",
            assignments=(
                PinAssignment(
                    mcu_ref="U1",
                    pin_number="IO4",
                    pin_name="IO4",
                    function=PinFunction.GPIO,
                    net="LED",
                    notes="Status LED",
                ),
            ),
            unassigned_gpio=("IO5",),
        )
    )
    builder.set_power_budget(
        PowerBudget(
            rails=(PowerRail(name="+3V3", voltage=3.3, current_ma=500.0, source_ref="U1"),),
            total_current_ma=500.0,
            notes=("Estimated",),
        )
    )
    builder.set_mechanical(
        MechanicalConstraints(
            board_width_mm=100.0,
            board_height_mm=80.0,
            enclosure=None,
            mounting_hole_positions=((5.0, 5.0),),
        )
    )
    return builder.build()


def test_requirements_to_dict_roundtrip() -> None:
    """requirements_from_dict(requirements_to_dict(req)) == req for a complete case."""
    from kicad_pipeline.models.requirements import ProjectRequirements

    req: ProjectRequirements = _make_full_requirements()  # type: ignore[assignment]
    data = requirements_to_dict(req)
    req2 = requirements_from_dict(data)

    assert req2 == req


def test_save_and_load_requirements(tmp_path: Path) -> None:
    """save_requirements / load_requirements roundtrip via a JSON file."""
    from kicad_pipeline.models.requirements import ProjectRequirements

    req: ProjectRequirements = _make_full_requirements()  # type: ignore[assignment]
    dest = tmp_path / "requirements.json"

    save_requirements(req, dest)
    assert dest.exists()

    loaded = load_requirements(dest)
    assert loaded == req


def test_load_requirements_missing_file(tmp_path: Path) -> None:
    """load_requirements raises RequirementsError for a missing file."""
    with pytest.raises(RequirementsError, match="Cannot read"):
        load_requirements(tmp_path / "nonexistent.json")


def test_requirements_from_dict_malformed() -> None:
    """requirements_from_dict raises RequirementsError on malformed input."""
    with pytest.raises(RequirementsError):
        requirements_from_dict({"project": "not-a-dict"})

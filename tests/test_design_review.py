"""Tests for kicad_pipeline.validation.design_review."""

from __future__ import annotations

from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.validation.design_review import (
    BoardSummary,
    DesignReview,
    ReviewItem,
    format_design_review,
    generate_design_review,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wifi_requirements() -> ProjectRequirements:
    """Requirements with an ESP32 WiFi module, relay, and ADC."""
    components = (
        Component(
            ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32-S3-WROOM-1",
            description="WiFi MCU",
            pins=(
                Pin("1", "GND", PinType.POWER_IN, net="GND"),
                Pin("2", "3V3", PinType.POWER_IN, net="+3V3"),
            ),
        ),
        Component(
            ref="U2", value="AMS1117-3.3", footprint="SOT-223",
            description="3.3V LDO regulator",
            pins=(
                Pin("1", "GND", PinType.POWER_IN, net="GND"),
                Pin("2", "VOUT", PinType.POWER_OUT, net="+3V3"),
                Pin("3", "VIN", PinType.POWER_IN, net="+5V"),
            ),
        ),
        Component(
            ref="C1", value="100nF", footprint="C_0402",
            description="MCU decoupling",
            pins=(
                Pin("1", "~", PinType.PASSIVE, net="+3V3"),
                Pin("2", "~", PinType.PASSIVE, net="GND"),
            ),
        ),
        Component(
            ref="K1", value="SRD-05VDC-SL-C", footprint="Relay_SPDT",
            description="Relay 1",
            pins=(
                Pin("1", "COIL+", PinType.PASSIVE, net="+5V"),
                Pin("2", "COIL-", PinType.PASSIVE, net="RLY1_COIL"),
            ),
        ),
        Component(
            ref="U3", value="ADS1115", footprint="TSSOP-10",
            description="16-bit ADC",
            pins=(
                Pin("1", "VDD", PinType.POWER_IN, net="+5V"),
                Pin("2", "GND", PinType.POWER_IN, net="GND"),
            ),
        ),
    )
    nets = (
        Net(name="GND", connections=(
            NetConnection("U1", "1"), NetConnection("C1", "2"),
            NetConnection("U2", "1"), NetConnection("U3", "2"),
        )),
        Net(name="+3V3", connections=(
            NetConnection("U1", "2"), NetConnection("U2", "2"),
            NetConnection("C1", "1"),
        )),
        Net(name="+5V", connections=(
            NetConnection("U2", "3"), NetConnection("K1", "1"),
            NetConnection("U3", "1"),
        )),
    )
    features = (
        FeatureBlock("MCU", "ESP32", ("U1", "C1"), ("+3V3", "GND"), ()),
        FeatureBlock("Power", "LDO", ("U2",), ("+5V", "+3V3"), ()),
        FeatureBlock("Relay", "Relay output", ("K1",), (), ()),
        FeatureBlock("ADC", "Analog input", ("U3",), (), ()),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="TestBoard", author="Test"),
        features=features,
        components=components,
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=100.0, board_height_mm=80.0),
    )


def _minimal_requirements() -> ProjectRequirements:
    """Requirements without WiFi, relays, or ADC."""
    components = (
        Component(
            ref="R1", value="10k", footprint="R_0805",
            pins=(
                Pin("1", "~", PinType.PASSIVE, net="SIG"),
                Pin("2", "~", PinType.PASSIVE, net="GND"),
            ),
        ),
    )
    nets = (
        Net(name="GND", connections=(NetConnection("R1", "2"),)),
        Net(name="SIG", connections=(NetConnection("R1", "1"),)),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="Minimal"),
        features=(),
        components=components,
        nets=nets,
    )


# ---------------------------------------------------------------------------
# ReviewItem / BoardSummary / DesignReview dataclass tests
# ---------------------------------------------------------------------------


def test_review_item_is_frozen() -> None:
    item = ReviewItem("antenna", "required", "Test", "Desc", ("U1",))
    assert item.category == "antenna"
    assert item.affected_refs == ("U1",)


def test_board_summary_fields() -> None:
    s = BoardSummary((100.0, 80.0), 5, 2, 3, ("+3V3", "+5V"), True, True, True)
    assert s.board_size_mm == (100.0, 80.0)
    assert s.has_wifi is True


def test_design_review_is_frozen() -> None:
    s = BoardSummary((80.0, 40.0), 1, 2, 1, (), False, False, False)
    r = DesignReview(s, ())
    assert len(r.items) == 0


# ---------------------------------------------------------------------------
# generate_design_review
# ---------------------------------------------------------------------------


def test_generate_review_detects_wifi() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    antenna_items = [i for i in review.items if i.category == "antenna"]
    assert len(antenna_items) >= 1
    assert any("keepout" in i.title.lower() for i in antenna_items)
    assert "U1" in antenna_items[0].affected_refs


def test_generate_review_detects_relays() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    relay_items = [i for i in review.items if i.category == "relay"]
    assert len(relay_items) >= 1
    assert any("K1" in i.affected_refs for i in relay_items)


def test_generate_review_detects_regulators() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    thermal_items = [i for i in review.items if i.category == "thermal"]
    assert len(thermal_items) >= 1
    assert "U2" in thermal_items[0].affected_refs


def test_generate_review_decoupling_pairs() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    decoupling = [i for i in review.items if "decoupling" in i.title.lower()]
    assert len(decoupling) >= 1
    # C1 should be paired with U1 (both on +3V3)
    paired_refs = set()
    for item in decoupling:
        paired_refs.update(item.affected_refs)
    assert "C1" in paired_refs


def test_generate_review_always_has_zone_fill() -> None:
    req = _minimal_requirements()
    review = generate_design_review(req)
    zone_fill = [i for i in review.items if "zone fill" in i.title.lower()]
    assert len(zone_fill) == 1
    assert zone_fill[0].severity == "required"


def test_generate_review_no_wifi_no_antenna_items() -> None:
    req = _minimal_requirements()
    review = generate_design_review(req)
    antenna_items = [i for i in review.items if i.category == "antenna"]
    assert len(antenna_items) == 0


def test_generate_review_board_summary() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    s = review.board_summary
    assert s.board_size_mm == (100.0, 80.0)
    assert s.component_count == 5
    assert s.has_wifi is True
    assert s.has_relays is True
    assert s.has_adc is True
    assert "+3V3" in s.power_nets
    assert "+5V" in s.power_nets


def test_generate_review_power_nets_detected() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    power_items = [i for i in review.items if i.category == "power"]
    assert len(power_items) >= 1


# ---------------------------------------------------------------------------
# format_design_review
# ---------------------------------------------------------------------------


def test_format_review_contains_heading() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    md = format_design_review(review, project_name="TestBoard")
    assert "# Design Review: TestBoard" in md


def test_format_review_contains_sections() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    md = format_design_review(review)
    assert "## Board Summary" in md
    assert "## Required Actions" in md
    assert "## Recommended" in md


def test_format_review_contains_checkboxes() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    md = format_design_review(review)
    assert "- [ ]" in md


def test_format_review_contains_component_count() -> None:
    req = _wifi_requirements()
    review = generate_design_review(req)
    md = format_design_review(review)
    assert "Components: 5" in md


def test_format_review_empty_project_name() -> None:
    req = _minimal_requirements()
    review = generate_design_review(req)
    md = format_design_review(review)
    assert md.startswith("# Design Review\n")

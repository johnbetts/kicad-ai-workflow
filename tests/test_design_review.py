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
        Net(name="RLY1_COIL", connections=(
            NetConnection("K1", "2"),
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


def test_generate_review_wifi_edge_clearance() -> None:
    """WiFi → antenna edge clearance reminder (keepout is auto-generated)."""
    req = _wifi_requirements()
    review = generate_design_review(req)
    antenna_items = [i for i in review.items if i.category == "antenna"]
    assert len(antenna_items) >= 1
    assert any("edge" in i.title.lower() for i in antenna_items)
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


def test_no_automated_items_in_review() -> None:
    """Items handled by the framework should NOT appear as recommendations.

    The PCB builder auto-handles:
    - Antenna keepout zones (_make_antenna_keepout)
    - Power trace widths (netclasses classify_nets)
    - Decoupling cap placement (constraints NEAR)
    - Relay trace widths (netclasses)
    """
    req = _wifi_requirements()
    review = generate_design_review(req)
    titles = [i.title.lower() for i in review.items]
    # These should NOT appear — they're automated
    assert not any("high-current trace" in t for t in titles)
    assert not any("decoupling cap" in t for t in titles)
    assert not any(t == "antenna keepout zone" for t in titles)
    assert not any("relay trace width" in t for t in titles)


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


# ---------------------------------------------------------------------------
# Component grouping
# ---------------------------------------------------------------------------


def test_component_groups_from_features() -> None:
    """Each feature block should produce a ComponentGroup."""
    req = _wifi_requirements()
    review = generate_design_review(req)
    assert len(review.component_groups) == 4
    group_names = {g.name for g in review.component_groups}
    assert "MCU" in group_names
    assert "Power" in group_names
    assert "Relay" in group_names
    assert "ADC" in group_names


def test_component_groups_contain_refs() -> None:
    """Each group should list the refs from its feature block."""
    req = _wifi_requirements()
    review = generate_design_review(req)
    mcu_group = next(g for g in review.component_groups if g.name == "MCU")
    assert "U1" in mcu_group.refs
    assert "C1" in mcu_group.refs


def test_component_groups_have_subgroups() -> None:
    """IC + decoupling cap pairs should appear as subgroups."""
    req = _wifi_requirements()
    review = generate_design_review(req)
    mcu_group = next(g for g in review.component_groups if g.name == "MCU")
    # U1 (ESP32) + C1 (decoupling) should be a subgroup
    assert len(mcu_group.subgroups) >= 1
    u1_sub = next(
        (s for s in mcu_group.subgroups if "U1" in s.refs), None
    )
    assert u1_sub is not None
    assert "C1" in u1_sub.refs


def test_format_review_includes_component_groups() -> None:
    """Formatted review should include Component Groups section."""
    req = _wifi_requirements()
    review = generate_design_review(req)
    md = format_design_review(review, project_name="TestBoard")
    assert "## Component Groups" in md
    assert "### MCU" in md
    assert "### Power" in md


def test_no_groups_when_no_features() -> None:
    """Requirements without features produce no component groups."""
    req = _minimal_requirements()
    review = generate_design_review(req)
    assert len(review.component_groups) == 0


# ---------------------------------------------------------------------------
# Connectivity validation
# ---------------------------------------------------------------------------


def _unconnected_component_requirements() -> ProjectRequirements:
    """Requirements with a fully unconnected component."""
    components = (
        Component(
            ref="R1", value="10k", footprint="R_0805",
            pins=(
                Pin("1", "~", PinType.PASSIVE, net="SIG"),
                Pin("2", "~", PinType.PASSIVE, net="GND"),
            ),
        ),
        Component(
            ref="R2", value="10k", footprint="R_0805",
            pins=(
                Pin("1", "~", PinType.PASSIVE),
                Pin("2", "~", PinType.PASSIVE),
            ),
        ),
    )
    nets = (
        Net(name="GND", connections=(NetConnection("R1", "2"),)),
        Net(name="SIG", connections=(NetConnection("R1", "1"),)),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="UnconnectedTest"),
        features=(
            FeatureBlock("Main", "Test", ("R1",), ("GND", "SIG"), ()),
        ),
        components=components,
        nets=nets,
    )


def test_detects_unconnected_component() -> None:
    """A component with no net connections should be flagged as required."""
    req = _unconnected_component_requirements()
    review = generate_design_review(req)
    unconnected = [
        i for i in review.items
        if i.category == "connectivity" and "unconnected component" in i.title.lower()
    ]
    assert len(unconnected) == 1
    assert "R2" in unconnected[0].affected_refs
    assert unconnected[0].severity == "required"


def test_detects_orphaned_component() -> None:
    """A component not in any feature block should be flagged."""
    req = _unconnected_component_requirements()
    review = generate_design_review(req)
    orphaned = [
        i for i in review.items
        if "orphaned" in i.title.lower()
    ]
    # R2 is not in any feature block
    assert len(orphaned) == 1
    assert "R2" in orphaned[0].affected_refs


def test_detects_dead_end_net() -> None:
    """A signal net with only one connection should be flagged."""
    components = (
        Component(
            ref="U1", value="MCU", footprint="QFP-48",
            pins=(
                Pin("1", "VCC", PinType.POWER_IN, net="+3V3"),
                Pin("2", "OUT", PinType.OUTPUT, net="ORPHAN_SIG"),
            ),
        ),
    )
    nets = (
        Net(name="+3V3", connections=(NetConnection("U1", "1"),)),
        Net(name="ORPHAN_SIG", connections=(NetConnection("U1", "2"),)),
    )
    req = ProjectRequirements(
        project=ProjectInfo(name="DeadEndTest"),
        features=(),
        components=components,
        nets=nets,
    )
    review = generate_design_review(req)
    dead_end = [i for i in review.items if "dead-end" in i.title.lower()]
    assert len(dead_end) == 1
    assert "ORPHAN_SIG" in dead_end[0].description
    assert dead_end[0].severity == "required"


def test_no_false_positive_power_dead_end() -> None:
    """Power nets with one connection should NOT be flagged as dead-end."""
    components = (
        Component(
            ref="U1", value="MCU", footprint="QFP-48",
            pins=(
                Pin("1", "VCC", PinType.POWER_IN, net="+3V3"),
                Pin("2", "GND", PinType.POWER_IN, net="GND"),
            ),
        ),
    )
    nets = (
        Net(name="+3V3", connections=(NetConnection("U1", "1"),)),
        Net(name="GND", connections=(NetConnection("U1", "2"),)),
    )
    req = ProjectRequirements(
        project=ProjectInfo(name="PowerTest"),
        features=(),
        components=components,
        nets=nets,
    )
    review = generate_design_review(req)
    dead_end = [i for i in review.items if "dead-end" in i.title.lower()]
    assert len(dead_end) == 0


def test_connected_components_no_warnings() -> None:
    """Properly connected components should not trigger connectivity warnings."""
    req = _wifi_requirements()
    review = generate_design_review(req)
    unconnected = [
        i for i in review.items
        if i.category == "connectivity" and "unconnected component" in i.title.lower()
    ]
    assert len(unconnected) == 0


# ---------------------------------------------------------------------------
# Subcircuit design notes
# ---------------------------------------------------------------------------


def test_subcircuit_notes_relay_driver() -> None:
    """Relay components trigger relay driver design notes."""
    req = _wifi_requirements()
    review = generate_design_review(req)
    relay_notes = [
        i for i in review.items
        if i.category == "subcircuit" and "relay" in i.title.lower()
    ]
    assert len(relay_notes) >= 1
    assert "flyback" in relay_notes[0].description.lower()
    assert "K1" in relay_notes[0].affected_refs


def test_subcircuit_notes_ldo() -> None:
    """Regulator components trigger LDO design notes."""
    req = _wifi_requirements()
    review = generate_design_review(req)
    ldo_notes = [
        i for i in review.items
        if i.category == "subcircuit" and "ldo" in i.title.lower()
    ]
    assert len(ldo_notes) >= 1
    assert "thermal" in ldo_notes[0].description.lower()
    assert "U2" in ldo_notes[0].affected_refs


def test_subcircuit_notes_adc() -> None:
    """ADC components trigger voltage divider routing notes."""
    req = _wifi_requirements()
    review = generate_design_review(req)
    adc_notes = [
        i for i in review.items
        if i.category == "subcircuit" and "adc" in i.title.lower()
    ]
    assert len(adc_notes) >= 1
    assert "guard ring" in adc_notes[0].description.lower()


def test_subcircuit_notes_declared_in_feature() -> None:
    """Subcircuit types from FeatureBlock.subcircuits are detected."""
    components = (
        Component(
            ref="K1", value="SRD-05VDC-SL-C", footprint="Relay_SPDT",
            pins=(
                Pin("1", "COIL+", PinType.PASSIVE, net="+5V"),
                Pin("2", "COIL-", PinType.PASSIVE, net="RLY1_COIL"),
            ),
        ),
    )
    nets = (
        Net(name="+5V", connections=(NetConnection("K1", "1"),)),
        Net(name="RLY1_COIL", connections=(NetConnection("K1", "2"),)),
    )
    features = (
        FeatureBlock("Relays", "4x relay outputs", ("K1",), (), ("relay_driver",)),
    )
    req = ProjectRequirements(
        project=ProjectInfo(name="SubTest"),
        features=features,
        components=components,
        nets=nets,
    )
    review = generate_design_review(req)
    relay_notes = [
        i for i in review.items
        if i.category == "subcircuit" and "relay" in i.title.lower()
    ]
    assert len(relay_notes) >= 1


def test_no_subcircuit_notes_for_minimal() -> None:
    """Minimal design without subcircuits has no subcircuit notes."""
    req = _minimal_requirements()
    review = generate_design_review(req)
    subcircuit_items = [i for i in review.items if i.category == "subcircuit"]
    assert len(subcircuit_items) == 0

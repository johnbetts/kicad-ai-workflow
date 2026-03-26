"""Tests for the layout and routing guide generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.models.requirements import (
    Component,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.research.layout_guide import (
    _detect_ic_subtype,
    _ref_type,
    _trace_width_mm,
    generate_layout_guide,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pin(num: str) -> Pin:
    return Pin(number=num, name=num, pin_type=PinType.PASSIVE)


def _comp(ref: str, value: str, **kwargs: object) -> Component:
    return Component(
        ref=ref,
        value=value,
        footprint=str(kwargs.get("footprint", "R_0805")),
        description=str(kwargs.get("description", "")),
        pins=(_pin("1"), _pin("2")),
    )


def _make_req(
    components: tuple[Component, ...] = (),
    nets: tuple[Net, ...] = (),
    power_budget: PowerBudget | None = None,
    mechanical: MechanicalConstraints | None = None,
) -> ProjectRequirements:
    return ProjectRequirements(
        project=ProjectInfo(name="test-board"),
        features=(),
        components=components,
        nets=nets,
        power_budget=power_budget,
        mechanical=mechanical,
    )


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


class TestRefType:
    def test_resistor(self) -> None:
        assert _ref_type("R1") == "resistor"

    def test_ic(self) -> None:
        assert _ref_type("U3") == "ic"

    def test_connector(self) -> None:
        assert _ref_type("J2") == "connector"

    def test_relay(self) -> None:
        assert _ref_type("K1") == "relay"

    def test_unknown(self) -> None:
        assert _ref_type("X1") == "unknown"


class TestDetectIcSubtype:
    def test_esp32(self) -> None:
        comp = _comp("U1", "ESP32-S3-WROOM-1")
        assert _detect_ic_subtype(comp) == "wifi_ble"

    def test_ads1115(self) -> None:
        comp = _comp("U2", "ADS1115", description="16-bit ADC")
        assert _detect_ic_subtype(comp) == "adc"

    def test_ams1117(self) -> None:
        comp = _comp("U3", "AMS1117-3.3")
        assert _detect_ic_subtype(comp) == "ldo"

    def test_generic_ic(self) -> None:
        comp = _comp("U4", "SomeCustomIC")
        assert _detect_ic_subtype(comp) is None


class TestTraceWidth:
    def test_zero_current(self) -> None:
        assert _trace_width_mm(0) == 0.15

    def test_small_current(self) -> None:
        width = _trace_width_mm(0.1)
        assert 0.15 <= width <= 0.5

    def test_high_current(self) -> None:
        width = _trace_width_mm(2.0)
        assert width >= 0.5

    def test_monotonic(self) -> None:
        """Higher current should need wider trace."""
        w1 = _trace_width_mm(0.5)
        w2 = _trace_width_mm(2.0)
        assert w2 >= w1


# ---------------------------------------------------------------------------
# Integration: full guide generation
# ---------------------------------------------------------------------------


class TestGenerateLayoutGuide:
    def test_minimal_requirements(self, tmp_path: Path) -> None:
        """Generates a guide even with minimal requirements."""
        req = _make_req(
            components=(_comp("R1", "10k"),),
        )
        path = generate_layout_guide(req, tmp_path / "guide.md")
        assert path.exists()
        content = path.read_text()
        assert "Layout & Routing Guide" in content
        assert "test-board" in content

    def test_includes_board_overview(self, tmp_path: Path) -> None:
        req = _make_req(
            components=(_comp("R1", "10k"),),
            mechanical=MechanicalConstraints(
                board_width_mm=65.0,
                board_height_mm=56.0,
                board_template="rpi_hat",
            ),
        )
        content = generate_layout_guide(req, tmp_path / "guide.md").read_text()
        assert "65.0 x 56.0 mm" in content
        assert "rpi_hat" in content

    def test_ic_placement_rules(self, tmp_path: Path) -> None:
        """ICs get specific placement guidance based on type."""
        req = _make_req(
            components=(
                _comp("U1", "ESP32-S3-WROOM-1"),
                _comp("U2", "ADS1115"),
            ),
        )
        content = generate_layout_guide(req, tmp_path / "guide.md").read_text()
        assert "antenna" in content.lower()
        assert "keepout" in content.lower() or "keep" in content.lower()
        assert "analog GND" in content or "ADC" in content

    def test_power_trace_widths(self, tmp_path: Path) -> None:
        """Power budget generates trace width recommendations."""
        req = _make_req(
            components=(_comp("R1", "10k"),),
            power_budget=PowerBudget(
                rails=(
                    PowerRail(name="+5V", voltage=5.0, current_ma=2000.0, source_ref="J1"),
                    PowerRail(name="+3V3", voltage=3.3, current_ma=500.0, source_ref="U1"),
                ),
                total_current_ma=2500.0,
                notes=(),
            ),
        )
        content = generate_layout_guide(req, tmp_path / "guide.md").read_text()
        assert "+5V" in content
        assert "+3V3" in content
        assert "2000mA" in content

    def test_diff_pair_detection(self, tmp_path: Path) -> None:
        """USB D+/D- nets are flagged as differential pairs."""
        req = _make_req(
            components=(_comp("J1", "USB-C"),),
            nets=(
                Net(name="USB_DP", connections=(NetConnection(ref="J1", pin="1"),)),
                Net(name="USB_DM", connections=(NetConnection(ref="J1", pin="2"),)),
            ),
        )
        content = generate_layout_guide(req, tmp_path / "guide.md").read_text()
        assert "Differential" in content
        assert "90Ω" in content

    def test_keepaway_rules(self, tmp_path: Path) -> None:
        """Relay + ADC combo triggers keep-away rule."""
        req = _make_req(
            components=(
                _comp("K1", "HFD4/005-S"),
                _comp("U1", "ADS1115"),
            ),
        )
        content = generate_layout_guide(req, tmp_path / "guide.md").read_text()
        assert "Keep-Away" in content
        assert "15mm" in content

    def test_decoupling_section(self, tmp_path: Path) -> None:
        """ADC IC gets specific decoupling guidance."""
        req = _make_req(
            components=(_comp("U1", "ADS1115"),),
        )
        content = generate_layout_guide(req, tmp_path / "guide.md").read_text()
        assert "Decoupling" in content
        assert "100nF" in content

    def test_thermal_section(self, tmp_path: Path) -> None:
        """LDO gets thermal management notes."""
        req = _make_req(
            components=(_comp("U1", "AMS1117-3.3"),),
        )
        content = generate_layout_guide(req, tmp_path / "guide.md").read_text()
        assert "Thermal" in content
        assert "thermal pad" in content.lower() or "Thermal pad" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Output directory is created if it doesn't exist."""
        req = _make_req(components=(_comp("R1", "10k"),))
        path = generate_layout_guide(req, tmp_path / "nested" / "dir" / "guide.md")
        assert path.exists()

    def test_connector_placement(self, tmp_path: Path) -> None:
        """Connectors get edge placement guidance."""
        req = _make_req(
            components=(
                _comp("J1", "USB-C", description="USB Type-C connector"),
            ),
        )
        content = generate_layout_guide(req, tmp_path / "guide.md").read_text()
        assert "edge" in content.lower()


# ---------------------------------------------------------------------------
# Tests for layout_templates module
# ---------------------------------------------------------------------------

from kicad_pipeline.models.pcb import Footprint, Pad  # noqa: E402
from kicad_pipeline.models.pcb import Point as PcbPoint  # noqa: E402
from kicad_pipeline.pcb.layout_templates import (  # noqa: E402
    CardinalSide,
    ComponentRole,
    ICTemplate,
    PinFunction,
    PinGroup,
    SubcircuitTemplate,
    TemplateSlot,
    auto_generate_ic_template,
    get_ic_template,
    get_subcircuit_template,
    register_ic_template,
    register_subcircuit_template,
)


def _make_ic_pad(number: str, x: float, y: float) -> Pad:
    return Pad(
        number=number, pad_type="smd", shape="rect",
        position=PcbPoint(x=x, y=y), size_x=0.5, size_y=0.3,
        layers=("F.Cu",),
    )


def _make_ic_footprint(ref: str, value: str, pads: tuple[Pad, ...]) -> Footprint:
    return Footprint(
        lib_id="test:test", ref=ref, value=value,
        position=PcbPoint(x=50.0, y=50.0), pads=pads,
    )


# ---------------------------------------------------------------------------
# get_ic_template
# ---------------------------------------------------------------------------


class TestGetIcTemplate:
    """Tests for get_ic_template() glob matching."""

    def test_esp32_match(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        assert "ESP32" in tmpl.ic_pattern

    def test_ads1115_match(self) -> None:
        tmpl = get_ic_template("ADS1115IDGSR")
        assert tmpl is not None

    def test_w5500_match(self) -> None:
        tmpl = get_ic_template("W5500-QFN48")
        assert tmpl is not None

    def test_lan8720_match(self) -> None:
        tmpl = get_ic_template("LAN8720A-CP-TR")
        assert tmpl is not None

    def test_unknown_ic_returns_none(self) -> None:
        assert get_ic_template("TOTALLY_UNKNOWN_IC_XYZ") is None

    def test_empty_string_returns_none(self) -> None:
        assert get_ic_template("") is None


# ---------------------------------------------------------------------------
# get_subcircuit_template
# ---------------------------------------------------------------------------


class TestGetSubcircuitTemplate:
    """Tests for get_subcircuit_template() lookup."""

    def test_voltage_divider(self) -> None:
        tmpl = get_subcircuit_template("voltage_divider")
        assert tmpl is not None
        assert tmpl.name == "Voltage Divider"

    def test_buck_converter(self) -> None:
        tmpl = get_subcircuit_template("buck_converter")
        assert tmpl is not None
        assert len(tmpl.slots) >= 5

    def test_relay_driver(self) -> None:
        tmpl = get_subcircuit_template("relay_driver")
        assert tmpl is not None

    def test_crystal_osc(self) -> None:
        tmpl = get_subcircuit_template("crystal_osc")
        assert tmpl is not None
        assert tmpl.flow_direction == "radial"

    def test_decoupling(self) -> None:
        tmpl = get_subcircuit_template("decoupling")
        assert tmpl is not None

    def test_rc_filter(self) -> None:
        tmpl = get_subcircuit_template("rc_filter")
        assert tmpl is not None

    def test_adc_channel(self) -> None:
        tmpl = get_subcircuit_template("adc_channel")
        assert tmpl is not None

    def test_ldo_regulator(self) -> None:
        tmpl = get_subcircuit_template("ldo_regulator")
        assert tmpl is not None

    def test_unknown_returns_none(self) -> None:
        assert get_subcircuit_template("nonexistent_circuit") is None

    def test_empty_string_returns_none(self) -> None:
        assert get_subcircuit_template("") is None


# ---------------------------------------------------------------------------
# ICTemplate methods
# ---------------------------------------------------------------------------


class TestICTemplateMethods:
    """Tests for ICTemplate query methods."""

    def test_groups_on_side_at_0(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        west_groups = tmpl.groups_on_side(CardinalSide.WEST)
        assert len(west_groups) >= 1

    def test_groups_on_side_rotated(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        # WEST at 0 becomes NORTH at 90
        north_at_90 = tmpl.groups_on_side(CardinalSide.NORTH, rotation=90.0)
        west_at_0 = tmpl.groups_on_side(CardinalSide.WEST, rotation=0.0)
        assert len(north_at_90) == len(west_at_0)

    def test_groups_on_side_empty(self) -> None:
        tmpl = ICTemplate(ic_pattern="TEST*", pin_groups=())
        assert tmpl.groups_on_side(CardinalSide.NORTH) == ()

    def test_preferred_side_for_function_found(self) -> None:
        tmpl = get_ic_template("ADS1115IDGSR")
        assert tmpl is not None
        side = tmpl.preferred_side_for_function(PinFunction.ANALOG_IN)
        assert side is not None

    def test_preferred_side_for_function_not_found(self) -> None:
        tmpl = ICTemplate(
            ic_pattern="TEST*",
            pin_groups=(
                PinGroup("A", CardinalSide.WEST, ("1",), (PinFunction.GPIO,)),
            ),
        )
        assert tmpl.preferred_side_for_function(PinFunction.USB) is None

    def test_preferred_side_rotated(self) -> None:
        tmpl = ICTemplate(
            ic_pattern="TEST*",
            pin_groups=(
                PinGroup("A", CardinalSide.WEST, ("1",), (PinFunction.SPI,)),
            ),
        )
        side = tmpl.preferred_side_for_function(PinFunction.SPI, rotation=90.0)
        assert side == CardinalSide.NORTH

    def test_antenna_side_esp32(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        assert tmpl.antenna_side == CardinalSide.NORTH

    def test_thermal_pad_esp32(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        assert tmpl.thermal_pad == "41"


# ---------------------------------------------------------------------------
# auto_generate_ic_template
# ---------------------------------------------------------------------------


class TestAutoGenerateIcTemplate:
    """Tests for auto_generate_ic_template()."""

    def test_generates_from_dual_row(self) -> None:
        """Dual-row IC produces groups on WEST and EAST."""
        pads = tuple(
            _make_ic_pad(str(i), -3.0, -1.5 + (i - 1) * 1.0)
            for i in range(1, 5)
        ) + tuple(
            _make_ic_pad(str(i), 3.0, -1.5 + (8 - i) * 1.0)
            for i in range(5, 9)
        )
        fp = _make_ic_footprint("U1", "MyIC", pads)
        tmpl = auto_generate_ic_template(fp)
        assert tmpl.version == "auto-1.0"
        sides = {pg.side for pg in tmpl.pin_groups}
        assert CardinalSide.WEST in sides
        assert CardinalSide.EAST in sides

    def test_custom_value_pattern(self) -> None:
        pads = (
            _make_ic_pad("1", -2.0, 0.0),
            _make_ic_pad("2", 2.0, 0.0),
        )
        fp = _make_ic_footprint("U1", "SomeChip", pads)
        tmpl = auto_generate_ic_template(fp, value_pattern="Custom*")
        assert tmpl.ic_pattern == "Custom*"

    def test_default_pattern_from_value(self) -> None:
        pads = (_make_ic_pad("1", -2.0, 0.0), _make_ic_pad("2", 2.0, 0.0))
        fp = _make_ic_footprint("U1", "ATMega328", pads)
        tmpl = auto_generate_ic_template(fp)
        assert tmpl.ic_pattern == "ATMega328*"

    def test_no_function_assignment(self) -> None:
        """Auto-generated templates have empty function tuples."""
        pads = (_make_ic_pad("1", -2.0, 0.0), _make_ic_pad("2", 2.0, 0.0))
        fp = _make_ic_footprint("U1", "Test", pads)
        tmpl = auto_generate_ic_template(fp)
        for pg in tmpl.pin_groups:
            assert pg.functions == ()

    def test_skips_center_pads(self) -> None:
        """Thermal/center pads should not appear in pin groups."""
        pads = (
            _make_ic_pad("1", -3.0, 0.0),
            _make_ic_pad("2", 3.0, 0.0),
            _make_ic_pad("EP", 0.0, 0.0),  # thermal pad at center
        )
        fp = _make_ic_footprint("U1", "QFN", pads)
        tmpl = auto_generate_ic_template(fp)
        all_pad_nums = set()
        for pg in tmpl.pin_groups:
            all_pad_nums.update(pg.pin_numbers)
        assert "EP" not in all_pad_nums

    def test_empty_pads(self) -> None:
        """Footprint with no pads produces no pin groups."""
        fp = _make_ic_footprint("U1", "Empty", ())
        tmpl = auto_generate_ic_template(fp)
        assert tmpl.pin_groups == ()


# ---------------------------------------------------------------------------
# register / custom templates
# ---------------------------------------------------------------------------


class TestTemplateRegistration:
    """Tests for register_ic_template and register_subcircuit_template."""

    def test_register_custom_ic_template(self) -> None:
        custom = ICTemplate(
            ic_pattern="CUSTOM_IC_12345*",
            pin_groups=(),
            version="test-1.0",
        )
        register_ic_template(custom)
        result = get_ic_template("CUSTOM_IC_12345_REV_A")
        assert result is not None
        assert result.version == "test-1.0"

    def test_register_overwrites_existing(self) -> None:
        custom_v1 = ICTemplate(ic_pattern="OVERWRITE_TEST*", pin_groups=(), version="1.0")
        custom_v2 = ICTemplate(ic_pattern="OVERWRITE_TEST*", pin_groups=(), version="2.0")
        register_ic_template(custom_v1)
        register_ic_template(custom_v2)
        result = get_ic_template("OVERWRITE_TEST_CHIP")
        assert result is not None
        assert result.version == "2.0"

    def test_register_custom_subcircuit(self) -> None:
        custom = SubcircuitTemplate(
            circuit_type_name="custom_filter_test",
            name="Custom Filter",
            flow_direction="left_to_right",
            slots=(),
            version="test-1.0",
        )
        register_subcircuit_template(custom)
        result = get_subcircuit_template("custom_filter_test")
        assert result is not None
        assert result.name == "Custom Filter"


# ---------------------------------------------------------------------------
# SubcircuitTemplate slot access
# ---------------------------------------------------------------------------


class TestSubcircuitTemplateSlots:
    """Test SubcircuitTemplate slot data."""

    def test_voltage_divider_has_two_slots(self) -> None:
        tmpl = get_subcircuit_template("voltage_divider")
        assert tmpl is not None
        assert len(tmpl.slots) == 2

    def test_voltage_divider_slots_face_each_other(self) -> None:
        tmpl = get_subcircuit_template("voltage_divider")
        assert tmpl is not None
        r_top = next(s for s in tmpl.slots if s.ref_pattern == "R_top")
        r_bot = next(s for s in tmpl.slots if s.ref_pattern == "R_bot")
        assert r_top.pad_face_toward == "R_bot"
        assert r_bot.pad_face_toward == "R_top"

    def test_buck_converter_has_anchor(self) -> None:
        tmpl = get_subcircuit_template("buck_converter")
        assert tmpl is not None
        anchor_slots = [s for s in tmpl.slots if s.role == ComponentRole.ANCHOR]
        assert len(anchor_slots) == 1

    def test_relay_driver_flow_direction(self) -> None:
        tmpl = get_subcircuit_template("relay_driver")
        assert tmpl is not None
        assert tmpl.flow_direction == "top_to_bottom"

    def test_template_slot_is_frozen(self) -> None:
        import pytest as _pt
        slot = TemplateSlot(
            role=ComponentRole.SERIES, ref_pattern="R1",
            offset_x=0.0, offset_y=0.0,
        )
        with _pt.raises(AttributeError):
            slot.offset_x = 5.0  # type: ignore[misc]

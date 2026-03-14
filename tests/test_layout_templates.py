"""Tests for pcb.layout_templates — IC and subcircuit layout templates."""

from __future__ import annotations

from kicad_pipeline.models.pcb import Footprint, Pad, Point
from kicad_pipeline.pcb.layout_templates import (
    ComponentRole,
    ICTemplate,
    PinFunction,
    SubcircuitTemplate,
    auto_generate_ic_template,
    get_ic_template,
    get_subcircuit_template,
    get_subcircuit_template_by_type,
    register_ic_template,
    register_subcircuit_template,
)
from kicad_pipeline.pcb.pin_map import CardinalSide

# ---------------------------------------------------------------------------
# IC Template Lookup
# ---------------------------------------------------------------------------


class TestICTemplateLookup:
    """Test IC template registry lookups."""

    def test_esp32_match(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        assert "ESP32" in tmpl.ic_pattern

    def test_esp32_wroom_variant(self) -> None:
        tmpl = get_ic_template("ESP32-WROOM-32E")
        assert tmpl is not None

    def test_ads1115_match(self) -> None:
        tmpl = get_ic_template("ADS1115IDGSR")
        assert tmpl is not None

    def test_no_match_returns_none(self) -> None:
        assert get_ic_template("NONEXISTENT_IC_XYZ") is None

    def test_w5500_match(self) -> None:
        tmpl = get_ic_template("W5500-LQFP48")
        assert tmpl is not None

    def test_lan8720_match(self) -> None:
        tmpl = get_ic_template("LAN8720A")
        assert tmpl is not None


# ---------------------------------------------------------------------------
# IC Template Functionality
# ---------------------------------------------------------------------------


class TestICTemplateFunction:
    """Test IC template pin group queries."""

    def test_esp32_spi_on_south_at_0(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        groups = tmpl.groups_on_side(CardinalSide.SOUTH, rotation=0.0)
        names = [g.name for g in groups]
        assert "SPI_GPIO" in names

    def test_esp32_spi_on_west_at_90(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        # SOUTH at 0 -> WEST at 90
        groups = tmpl.groups_on_side(CardinalSide.WEST, rotation=90.0)
        names = [g.name for g in groups]
        assert "SPI_GPIO" in names

    def test_preferred_side_for_spi(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        side = tmpl.preferred_side_for_function(PinFunction.SPI)
        assert side == CardinalSide.SOUTH

    def test_preferred_side_for_spi_rotated(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        side = tmpl.preferred_side_for_function(PinFunction.SPI, rotation=180.0)
        assert side == CardinalSide.NORTH

    def test_preferred_side_not_found(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        side = tmpl.preferred_side_for_function(PinFunction.USB)
        assert side is None

    def test_antenna_side(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        assert tmpl.antenna_side == CardinalSide.NORTH

    def test_version_field(self) -> None:
        tmpl = get_ic_template("ESP32-S3-WROOM-1")
        assert tmpl is not None
        assert tmpl.version == "2.0"


# ---------------------------------------------------------------------------
# IC Template Versioning (user's requirement)
# ---------------------------------------------------------------------------


class TestICTemplateVersioning:
    """Test that customized IC templates maintain version tracking."""

    def test_custom_version_preserved(self) -> None:
        custom = ICTemplate(
            ic_pattern="MyCustomIC*",
            pin_groups=(),
            version="2.0-custom",
        )
        register_ic_template(custom)
        found = get_ic_template("MyCustomIC-123")
        assert found is not None
        assert found.version == "2.0-custom"

    def test_override_with_new_version(self) -> None:
        v1 = ICTemplate(ic_pattern="VersionTestIC*", pin_groups=(), version="1.0")
        v2 = ICTemplate(ic_pattern="VersionTestIC*", pin_groups=(), version="2.0")
        register_ic_template(v1)
        register_ic_template(v2)
        found = get_ic_template("VersionTestIC-ABC")
        assert found is not None
        assert found.version == "2.0"


# ---------------------------------------------------------------------------
# Subcircuit Template Lookup
# ---------------------------------------------------------------------------


class TestSubcircuitTemplateLookup:
    """Test subcircuit template registry lookups."""

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

    def test_nonexistent_returns_none(self) -> None:
        assert get_subcircuit_template("nonexistent_xyz") is None

    def test_by_type_enum(self) -> None:
        from kicad_pipeline.optimization.functional_grouper import SubCircuitType
        tmpl = get_subcircuit_template_by_type(SubCircuitType.VOLTAGE_DIVIDER)
        assert tmpl is not None
        assert tmpl.circuit_type_name == "voltage_divider"


# ---------------------------------------------------------------------------
# Subcircuit Template Content
# ---------------------------------------------------------------------------


class TestSubcircuitTemplateContent:
    """Verify template slot structure and properties."""

    def test_voltage_divider_has_two_series_slots(self) -> None:
        tmpl = get_subcircuit_template("voltage_divider")
        assert tmpl is not None
        series = [s for s in tmpl.slots if s.role == ComponentRole.SERIES]
        assert len(series) == 2

    def test_voltage_divider_pads_face_each_other(self) -> None:
        tmpl = get_subcircuit_template("voltage_divider")
        assert tmpl is not None
        r_top = next(s for s in tmpl.slots if s.ref_pattern == "R_top")
        r_bot = next(s for s in tmpl.slots if s.ref_pattern == "R_bot")
        assert r_top.pad_face_toward == "R_bot"
        assert r_bot.pad_face_toward == "R_top"

    def test_buck_converter_has_anchor(self) -> None:
        tmpl = get_subcircuit_template("buck_converter")
        assert tmpl is not None
        anchors = [s for s in tmpl.slots if s.role == ComponentRole.ANCHOR]
        assert len(anchors) == 1
        assert anchors[0].ref_pattern == "IC"

    def test_relay_driver_flow_direction(self) -> None:
        tmpl = get_subcircuit_template("relay_driver")
        assert tmpl is not None
        assert tmpl.flow_direction == "top_to_bottom"

    def test_relay_driver_has_switch(self) -> None:
        tmpl = get_subcircuit_template("relay_driver")
        assert tmpl is not None
        switches = [s for s in tmpl.slots if s.role == ComponentRole.SWITCH]
        assert len(switches) == 1

    def test_crystal_radial_flow(self) -> None:
        tmpl = get_subcircuit_template("crystal_osc")
        assert tmpl is not None
        assert tmpl.flow_direction == "radial"

    def test_subcircuit_version(self) -> None:
        tmpl = get_subcircuit_template("voltage_divider")
        assert tmpl is not None
        assert tmpl.version == "1.0"


# ---------------------------------------------------------------------------
# Custom Subcircuit Registration
# ---------------------------------------------------------------------------


class TestCustomSubcircuitRegistration:
    """Test registering custom subcircuit templates."""

    def test_register_custom(self) -> None:
        custom = SubcircuitTemplate(
            circuit_type_name="custom_filter",
            name="Custom Filter",
            flow_direction="left_to_right",
            slots=(),
            version="1.0-custom",
        )
        register_subcircuit_template(custom)
        found = get_subcircuit_template("custom_filter")
        assert found is not None
        assert found.version == "1.0-custom"


# ---------------------------------------------------------------------------
# Auto-generated IC Template
# ---------------------------------------------------------------------------


class TestAutoGenerateICTemplate:
    """Test auto_generate_ic_template from footprint geometry."""

    def test_dual_row_ic(self) -> None:
        # SOIC-8: pins 1-4 left, 5-8 right
        pads = tuple(
            Pad(
                number=str(i),
                pad_type="smd",
                shape="rect",
                position=Point(x=-3.0, y=-1.5 + (i - 1) * 1.0),
                size_x=1.5,
                size_y=0.6,
                layers=("F.Cu",),
            )
            for i in range(1, 5)
        ) + tuple(
            Pad(
                number=str(i),
                pad_type="smd",
                shape="rect",
                position=Point(x=3.0, y=-1.5 + (8 - i) * 1.0),
                size_x=1.5,
                size_y=0.6,
                layers=("F.Cu",),
            )
            for i in range(5, 9)
        )
        fp = Footprint(
            lib_id="test:SOIC-8",
            ref="U1",
            value="LM358",
            position=Point(x=50, y=50),
            pads=pads,
        )
        tmpl = auto_generate_ic_template(fp)
        assert tmpl.version == "auto-1.0"
        assert len(tmpl.pin_groups) >= 2
        # Should have west and east groups
        sides = {g.side for g in tmpl.pin_groups}
        assert CardinalSide.WEST in sides
        assert CardinalSide.EAST in sides

    def test_custom_pattern(self) -> None:
        pads = (
            Pad(
                number="1", pad_type="smd", shape="rect",
                position=Point(x=-1, y=0), size_x=1, size_y=0.5,
                layers=("F.Cu",),
            ),
            Pad(
                number="2", pad_type="smd", shape="rect",
                position=Point(x=1, y=0), size_x=1, size_y=0.5,
                layers=("F.Cu",),
            ),
        )
        fp = Footprint(
            lib_id="test:test", ref="U1", value="test",
            position=Point(x=0, y=0), pads=pads,
        )
        tmpl = auto_generate_ic_template(fp, value_pattern="MyChip*")
        assert tmpl.ic_pattern == "MyChip*"

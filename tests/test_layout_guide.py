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

"""Tests for the variant system (footprint remapping)."""

from __future__ import annotations

from kicad_pipeline.models.requirements import (
    Component,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.orchestrator.models import PackageStrategy
from kicad_pipeline.orchestrator.variants import (
    detect_footprint_family,
    fork_requirements_for_variant,
    remap_footprint,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_base_requirements() -> ProjectRequirements:
    """Create a minimal base requirements set with passives and an IC."""
    r1 = Component(
        ref="R1",
        value="10k",
        footprint="R_0805",
        lcsc="C17414",
        pins=(
            Pin(number="1", name="1", pin_type=PinType.PASSIVE),
            Pin(number="2", name="2", pin_type=PinType.PASSIVE),
        ),
    )
    r2 = Component(
        ref="R2",
        value="330R",
        footprint="R_0805",
        lcsc="C17516",
        pins=(
            Pin(number="1", name="1", pin_type=PinType.PASSIVE),
            Pin(number="2", name="2", pin_type=PinType.PASSIVE),
        ),
    )
    c1 = Component(
        ref="C1",
        value="100nF",
        footprint="C_0805",
        lcsc="C49678",
        pins=(
            Pin(number="1", name="1", pin_type=PinType.PASSIVE),
            Pin(number="2", name="2", pin_type=PinType.PASSIVE),
        ),
    )
    led1 = Component(
        ref="D1",
        value="green",
        footprint="LED_0805",
        lcsc="C2297",
        pins=(
            Pin(number="1", name="A", pin_type=PinType.PASSIVE),
            Pin(number="2", name="K", pin_type=PinType.PASSIVE),
        ),
    )
    mcu = Component(
        ref="U1",
        value="ESP32-S3-WROOM-1",
        footprint="ESP32-S3-WROOM-1",
        lcsc="C2913202",
        pins=(
            Pin(number="1", name="GND", pin_type=PinType.POWER_IN),
            Pin(number="2", name="3V3", pin_type=PinType.POWER_IN),
        ),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="test-board"),
        features=(),
        components=(r1, r2, c1, led1, mcu),
        nets=(
            Net(name="GND", connections=(
                NetConnection(ref="R1", pin="1"),
                NetConnection(ref="U1", pin="1"),
            )),
        ),
    )


# ---------------------------------------------------------------------------
# detect_footprint_family
# ---------------------------------------------------------------------------


class TestDetectFootprintFamily:
    def test_resistor(self) -> None:
        assert detect_footprint_family("R_0805") == "R"
        assert detect_footprint_family("R_0603") == "R"
        assert detect_footprint_family("R_0402") == "R"

    def test_capacitor(self) -> None:
        assert detect_footprint_family("C_0805") == "C"
        assert detect_footprint_family("C_0603") == "C"

    def test_led(self) -> None:
        assert detect_footprint_family("LED_0805") == "LED"
        assert detect_footprint_family("LED_0603") == "LED"
        assert detect_footprint_family("LED_D3.0mm") == "LED"

    def test_ic_returns_none(self) -> None:
        assert detect_footprint_family("ESP32-S3-WROOM-1") is None
        assert detect_footprint_family("SOT-23-5") is None
        assert detect_footprint_family("QFP-48") is None

    def test_connector_returns_none(self) -> None:
        assert detect_footprint_family("USB_C_Receptacle") is None
        assert detect_footprint_family("RJ45") is None

    def test_colon_format(self) -> None:
        assert detect_footprint_family("R:R_0805_2012Metric") == "R"
        assert detect_footprint_family("C:C_0805_2012Metric") == "C"


# ---------------------------------------------------------------------------
# remap_footprint
# ---------------------------------------------------------------------------


class TestRemapFootprint:
    def test_resistor_0805_to_0603(self) -> None:
        strategy = PackageStrategy(name="0603", resistor_package="0603")
        result = remap_footprint("R_0805", "R", strategy)
        assert result == "R_0603"

    def test_resistor_0805_to_0402(self) -> None:
        strategy = PackageStrategy(name="0402", resistor_package="0402")
        result = remap_footprint("R_0805", "R", strategy)
        assert result == "R_0402"

    def test_capacitor_0805_to_0603(self) -> None:
        strategy = PackageStrategy(name="0603", capacitor_package="0603")
        result = remap_footprint("C_0805", "C", strategy)
        assert result == "C_0603"

    def test_led_0805_to_0603(self) -> None:
        strategy = PackageStrategy(name="0603", led_package="0603")
        result = remap_footprint("LED_0805", "LED", strategy)
        assert result == "LED_0603"

    def test_resistor_to_through_hole(self) -> None:
        strategy = PackageStrategy(
            name="through-hole",
            resistor_package="Axial_DIN0207",
            prefer_smd=False,
        )
        result = remap_footprint("R_0805", "R", strategy)
        assert result == "R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm"

    def test_capacitor_to_through_hole(self) -> None:
        strategy = PackageStrategy(
            name="through-hole",
            capacitor_package="C_Disc_D5.0mm",
            prefer_smd=False,
        )
        result = remap_footprint("C_0805", "C", strategy)
        assert result == "C_Disc_D5.0mm_W2.5mm_P2.50mm"

    def test_led_to_through_hole(self) -> None:
        strategy = PackageStrategy(
            name="through-hole",
            led_package="LED_D3.0mm",
            prefer_smd=False,
        )
        result = remap_footprint("LED_0805", "LED", strategy)
        assert result == "LED_D3.0mm"

    def test_unknown_package_falls_back(self) -> None:
        strategy = PackageStrategy(name="custom", resistor_package="2512")
        result = remap_footprint("R_0805", "R", strategy)
        assert result == "R_2512"


# ---------------------------------------------------------------------------
# fork_requirements_for_variant
# ---------------------------------------------------------------------------


class TestForkRequirementsForVariant:
    def test_same_strategy_no_change(self) -> None:
        base = _make_base_requirements()
        strategy = PackageStrategy(name="0805")
        forked = fork_requirements_for_variant(base, strategy)
        for orig, new in zip(base.components, forked.components, strict=False):
            assert orig.footprint == new.footprint

    def test_remap_to_0603(self) -> None:
        base = _make_base_requirements()
        strategy = PackageStrategy(
            name="0603",
            resistor_package="0603",
            capacitor_package="0603",
            led_package="0603",
        )
        forked = fork_requirements_for_variant(base, strategy)

        # R1, R2 should be R_0603
        assert forked.components[0].footprint == "R_0603"
        assert forked.components[1].footprint == "R_0603"
        # C1 should be C_0603
        assert forked.components[2].footprint == "C_0603"
        # LED should be LED_0603
        assert forked.components[3].footprint == "LED_0603"

    def test_ic_untouched(self) -> None:
        base = _make_base_requirements()
        strategy = PackageStrategy(
            name="0603",
            resistor_package="0603",
            capacitor_package="0603",
            led_package="0603",
        )
        forked = fork_requirements_for_variant(base, strategy)
        # MCU (U1) should be unchanged
        assert forked.components[4].footprint == "ESP32-S3-WROOM-1"
        assert forked.components[4].lcsc == "C2913202"

    def test_remap_to_through_hole(self) -> None:
        base = _make_base_requirements()
        strategy = PackageStrategy(
            name="through-hole",
            resistor_package="Axial_DIN0207",
            capacitor_package="C_Disc_D5.0mm",
            led_package="LED_D3.0mm",
            prefer_smd=False,
        )
        forked = fork_requirements_for_variant(base, strategy)
        assert "Axial" in forked.components[0].footprint
        assert "Disc" in forked.components[2].footprint
        assert "D3.0mm" in forked.components[3].footprint

    def test_preserves_project_info(self) -> None:
        base = _make_base_requirements()
        strategy = PackageStrategy(name="0603", resistor_package="0603")
        forked = fork_requirements_for_variant(base, strategy)
        assert forked.project == base.project

    def test_preserves_nets(self) -> None:
        base = _make_base_requirements()
        strategy = PackageStrategy(name="0603", resistor_package="0603")
        forked = fork_requirements_for_variant(base, strategy)
        assert forked.nets == base.nets

    def test_preserves_component_count(self) -> None:
        base = _make_base_requirements()
        strategy = PackageStrategy(name="0603", resistor_package="0603")
        forked = fork_requirements_for_variant(base, strategy)
        assert len(forked.components) == len(base.components)

    def test_preserves_pins(self) -> None:
        base = _make_base_requirements()
        strategy = PackageStrategy(name="0603", resistor_package="0603")
        forked = fork_requirements_for_variant(base, strategy)
        for orig, new in zip(base.components, forked.components, strict=False):
            assert orig.pins == new.pins

    def test_preserves_ref_and_value(self) -> None:
        base = _make_base_requirements()
        strategy = PackageStrategy(name="0603", resistor_package="0603")
        forked = fork_requirements_for_variant(base, strategy)
        for orig, new in zip(base.components, forked.components, strict=False):
            assert orig.ref == new.ref
            assert orig.value == new.value

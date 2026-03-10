"""Tests for zone strategy recommendation engine."""

from __future__ import annotations

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.models.requirements import (
    Component,
    MechanicalConstraints,
    Net,
    Pin,
    PinFunction,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.zone_optimizer import (
    ZoneStrategy,
    recommend_zone_strategy,
)


def _make_outline(w: float = 80.0, h: float = 40.0) -> BoardOutline:
    """Create a rectangular board outline."""
    return BoardOutline(
        polygon=(
            Point(0, 0),
            Point(w, 0),
            Point(w, h),
            Point(0, h),
            Point(0, 0),
        ),
    )


def _make_pcb(
    nets: tuple[NetEntry, ...] = (),
    footprints: tuple[Footprint, ...] = (),
    outline: BoardOutline | None = None,
) -> PCBDesign:
    """Create a minimal PCBDesign for testing."""
    return PCBDesign(
        outline=outline or _make_outline(),
        design_rules=DesignRules(),
        nets=nets,
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


def _make_requirements(
    components: tuple[Component, ...] = (),
    nets: tuple[Net, ...] = (),
    power_budget: PowerBudget | None = None,
) -> ProjectRequirements:
    """Create minimal ProjectRequirements for testing."""
    return ProjectRequirements(
        project=ProjectInfo(name="test"),
        features=(),
        components=components,
        nets=nets,
        power_budget=power_budget,
        mechanical=MechanicalConstraints(board_width_mm=80.0, board_height_mm=40.0),
    )


class TestZoneStrategyFrozen:
    """ZoneStrategy is a frozen dataclass."""

    def test_zone_strategy_frozen(self) -> None:
        strategy = ZoneStrategy(
            gnd_strategy="both",
            power_zones=(),
            copper_fill_ratio=0.7,
            thermal_relief_style="relief",
            rationale=("test",),
        )
        assert strategy.gnd_strategy == "both"
        # Frozen — assignment should raise
        try:
            strategy.gnd_strategy = "back_only"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


class TestDefaultStrategy:
    """Default board with no special characteristics."""

    def test_default_strategy_both_gnd(self) -> None:
        pcb = _make_pcb()
        reqs = _make_requirements()
        strategy = recommend_zone_strategy(pcb, reqs)
        assert strategy.gnd_strategy == "both"
        assert strategy.thermal_relief_style == "relief"
        assert strategy.power_zones == ()


class TestRFModule:
    """RF module detection selects back_only GND."""

    def test_rf_module_selects_back_only(self) -> None:
        pcb = _make_pcb()
        reqs = _make_requirements(
            components=(
                Component(ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32-S3-WROOM-1"),
            ),
        )
        strategy = recommend_zone_strategy(pcb, reqs)
        assert strategy.gnd_strategy == "back_only"

    def test_rf_overrides_analog_split(self) -> None:
        """RF takes priority over analog split."""
        pcb = _make_pcb()
        reqs = _make_requirements(
            components=(
                Component(
                    ref="U1",
                    value="nRF52840",
                    footprint="QFN-48",
                    pins=(
                        Pin(
                            number="1", name="AIN0",
                            pin_type=PinType.INPUT, function=PinFunction.ADC,
                        ),
                    ),
                ),
            ),
        )
        strategy = recommend_zone_strategy(pcb, reqs)
        assert strategy.gnd_strategy == "back_only"


class TestAnalogDigital:
    """Analog + digital nets select split GND."""

    def test_analog_digital_selects_split(self) -> None:
        pcb = _make_pcb()
        reqs = _make_requirements(
            components=(
                Component(
                    ref="U1",
                    value="STM32F103",
                    footprint="LQFP-48",
                    pins=(
                        Pin(
                            number="1", name="PA0",
                            pin_type=PinType.INPUT, function=PinFunction.ADC,
                        ),
                    ),
                ),
            ),
        )
        strategy = recommend_zone_strategy(pcb, reqs)
        assert strategy.gnd_strategy == "split"

    def test_analog_net_name_triggers_split(self) -> None:
        pcb = _make_pcb()
        reqs = _make_requirements(
            nets=(Net(name="ANALOG_IN", connections=()),),
        )
        strategy = recommend_zone_strategy(pcb, reqs)
        assert strategy.gnd_strategy == "split"


class TestHighCurrent:
    """High-current rails select solid thermal relief."""

    def test_high_current_selects_solid_relief(self) -> None:
        pcb = _make_pcb()
        reqs = _make_requirements(
            power_budget=PowerBudget(
                rails=(
                    PowerRail(name="+5V", voltage=5.0, current_ma=1500.0, source_ref="U1"),
                ),
                total_current_ma=1500.0,
                notes=(),
            ),
        )
        strategy = recommend_zone_strategy(pcb, reqs)
        assert strategy.thermal_relief_style == "solid"


class TestPowerZones:
    """Multiple power nets create dedicated zones."""

    def test_multiple_power_nets_creates_zones(self) -> None:
        nets = tuple(
            NetEntry(number=i, name=name)
            for i, name in enumerate([
                "GND", "+3V3", "+5V", "+12V", "VCC", "VBUS",
            ])
        )
        pcb = _make_pcb(nets=nets)
        reqs = _make_requirements()
        strategy = recommend_zone_strategy(pcb, reqs)
        # 5 power nets (GND is not matched by _POWER_NET_RE; +3V3, +5V, +12V, VCC, VBUS = 5)
        assert len(strategy.power_zones) >= 5


class TestFillRatio:
    """Copper fill ratio estimation."""

    def test_fill_ratio_estimation(self) -> None:
        # Board 80x40 = 3200 mm^2, one small footprint
        pad = Pad(
            number="1", pad_type="smd", shape="rect",
            position=Point(0, 0), size_x=1.0, size_y=1.0,
            layers=("F.Cu",),
        )
        fp = Footprint(
            lib_id="R_0805:R_0805_2012Metric",
            ref="R1",
            value="10k",
            position=Point(40, 20),
            pads=(pad,),
        )
        pcb = _make_pcb(footprints=(fp,))
        reqs = _make_requirements()
        strategy = recommend_zone_strategy(pcb, reqs)
        # Small component on large board -> high fill ratio
        assert 0.9 <= strategy.copper_fill_ratio <= 1.0

    def test_empty_board_defaults(self) -> None:
        pcb = _make_pcb()
        reqs = _make_requirements()
        strategy = recommend_zone_strategy(pcb, reqs)
        # No footprints -> fill ratio = 1.0
        assert strategy.copper_fill_ratio == 1.0


class TestRationale:
    """Rationale is always populated."""

    def test_rationale_populated(self) -> None:
        pcb = _make_pcb()
        reqs = _make_requirements()
        strategy = recommend_zone_strategy(pcb, reqs)
        assert len(strategy.rationale) >= 4
        assert all(isinstance(r, str) for r in strategy.rationale)

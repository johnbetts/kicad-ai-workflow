"""Tests for the DOA-board netlist lint (2026-06-12 defect classes)."""

from __future__ import annotations

from kicad_pipeline.evals.netlist_lint import (
    check_diode_polarity,
    check_pin_pad_coverage,
)
from kicad_pipeline.models.pcb import Footprint, Pad, Point
from kicad_pipeline.models.requirements import (
    Component,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.placement_v2.ir import Severity


def _pad(number: str) -> Pad:
    return Pad(
        number=number, pad_type="smd", shape="rect",
        position=Point(0.0, 0.0), size_x=1.0, size_y=1.0,
        layers=("F.Cu",),
    )


def _fp(ref: str, pad_numbers: tuple[str, ...]) -> Footprint:
    return Footprint(
        lib_id="test:fp", ref=ref, value="v",
        position=Point(0.0, 0.0), rotation=0.0, layer="F.Cu",
        pads=tuple(_pad(n) for n in pad_numbers),
    )


def _req(components, nets=()):  # type: ignore[no-untyped-def]
    return ProjectRequirements(
        project=ProjectInfo(name="t"), features=(),
        components=tuple(components), nets=tuple(nets),
    )


class TestPinPadCoverage:
    def test_full_coverage_passes(self) -> None:
        comp = Component(ref="U1", value="ic", footprint="fp", pins=tuple(
            Pin(number=str(i), name=f"P{i}", pin_type=PinType.PASSIVE)
            for i in range(1, 9)
        ))
        fps = {"U1": _fp("U1", tuple(str(i) for i in range(1, 9)))}
        assert check_pin_pad_coverage(_req([comp]), fps) == ()

    def test_stub_pin_map_is_critical(self) -> None:
        # The W5500 class: 18 declared pins on a 48-pad footprint.
        comp = Component(ref="U6", value="W5500", footprint="LQFP-48", pins=tuple(
            Pin(number=str(i), name=f"P{i}", pin_type=PinType.PASSIVE)
            for i in range(1, 19)
        ))
        fps = {"U6": _fp("U6", tuple(str(i) for i in range(1, 49)))}
        violations = check_pin_pad_coverage(_req([comp]), fps)
        assert len(violations) == 1
        assert violations[0].severity is Severity.CRITICAL
        assert "stub pin map" in violations[0].message

    def test_ghost_pin_is_critical(self) -> None:
        comp = Component(ref="U1", value="ic", footprint="fp", pins=(
            Pin(number="1", name="A", pin_type=PinType.PASSIVE),
            Pin(number="9", name="B", pin_type=PinType.PASSIVE),  # no pad 9
        ))
        fps = {"U1": _fp("U1", ("1", "2"))}
        violations = check_pin_pad_coverage(_req([comp]), fps)
        assert any("no pad" in v.message and "9" in v.message for v in violations)

    def test_module_with_unused_ios_passes(self) -> None:
        # ESP32 class: ~73% of pads modeled is a normal partial map.
        comp = Component(ref="U3", value="esp32", footprint="module", pins=tuple(
            Pin(number=str(i), name=f"IO{i}", pin_type=PinType.BIDIRECTIONAL)
            for i in range(1, 31)
        ))
        fps = {"U3": _fp("U3", tuple(str(i) for i in range(1, 42)))}
        assert check_pin_pad_coverage(_req([comp]), fps) == ()

    def test_shield_pads_do_not_count(self) -> None:
        comp = Component(ref="J1", value="rj45", footprint="jack", pins=(
            Pin(number="1", name="P1", pin_type=PinType.PASSIVE),
            Pin(number="2", name="P2", pin_type=PinType.PASSIVE),
        ))
        fps = {"J1": _fp("J1", ("1", "2", "SH1", "SH2"))}
        assert check_pin_pad_coverage(_req([comp]), fps) == ()


class TestDiodePolarity:
    @staticmethod
    def _tvs(anode_net: str, cathode_net: str, value: str = "30V_TVS"):  # type: ignore[no-untyped-def]
        comp = Component(ref="D5", value=value, footprint="SOD-123", pins=(
            Pin(number="1", name="A", pin_type=PinType.PASSIVE),
            Pin(number="2", name="K", pin_type=PinType.PASSIVE),
        ))
        nets = (
            Net(name=anode_net, connections=(NetConnection("D5", "1"),)),
            Net(name=cathode_net, connections=(NetConnection("D5", "2"),)),
        )
        return _req([comp], nets)

    def test_reversed_tvs_is_critical(self) -> None:
        violations = check_diode_polarity(self._tvs("VIN", "GND"))
        assert len(violations) == 1
        assert violations[0].severity is Severity.CRITICAL
        assert "forward-biased" in violations[0].message

    def test_correct_tvs_passes(self) -> None:
        assert check_diode_polarity(self._tvs("GND", "VIN")) == ()

    def test_led_anode_on_rail_is_fine(self) -> None:
        assert check_diode_polarity(self._tvs("+5V", "GND", value="GREEN LED")) == ()

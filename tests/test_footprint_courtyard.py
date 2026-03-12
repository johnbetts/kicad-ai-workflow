"""Tests for estimate_courtyard_mm() — accurate courtyard size estimation."""

from __future__ import annotations

import pytest

from kicad_pipeline.models.pcb import Footprint, Pad, Point
from kicad_pipeline.pcb.footprints import estimate_courtyard_mm


def _make_fp(
    lib_id: str,
    pads: tuple[Pad, ...] = (),
    ref: str = "U1",
) -> Footprint:
    """Helper to build a minimal Footprint for testing."""
    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value="test",
        position=Point(x=0.0, y=0.0),
        pads=pads,
    )


def _make_pad(x: float, y: float, sx: float = 0.6, sy: float = 0.6) -> Pad:
    return Pad(
        number="1",
        pad_type="smd",
        shape="rect",
        position=Point(x=x, y=y),
        size_x=sx,
        size_y=sy,
        layers=("F.Cu",),
    )


class TestModuleCourtyards:
    """ESP32 and similar modules should have large courtyards."""

    def test_esp32_wroom_courtyard_size(self) -> None:
        """ESP32-S3-WROOM-1 courtyard should be ~18.5x26mm, not ~16x19mm."""
        # ESP32 pads: roughly 16mm wide, 18mm tall pad field
        pads = tuple(
            _make_pad(x, y, 0.5, 0.5)
            for x in (-8.0, 8.0)
            for y in range(-9, 10)
        )
        fp = _make_fp("ESP32-S3-WROOM-1", pads=pads)
        w, h = estimate_courtyard_mm(fp)
        # Body extension: 0.5mm/side width + 4.0mm/side height + 0.25mm courtyard
        # pad_w=16.5, pad_h=18.5 → w≈18.0, h≈27.0
        assert w >= 17.0, f"ESP32 width {w} too small (expected ≥17mm)"
        assert h >= 25.0, f"ESP32 height {h} too small (expected ≥25mm)"

    def test_w5500_module_courtyard(self) -> None:
        """W5500 module should get module-class body extension."""
        pads = tuple(
            _make_pad(x, y, 0.4, 0.4)
            for x in (-6.0, 6.0)
            for y in (-6.0, 6.0)
        )
        fp = _make_fp("W5500_QFN-48", pads=pads)
        w, h = estimate_courtyard_mm(fp)
        # QFN, not module — should be smaller extension
        assert w < 20.0, "W5500 QFN should not get module extension"


class TestPassiveCourtyards:
    """SMD passives should have compact courtyards close to pad extent."""

    def test_0603_resistor(self) -> None:
        """0603 resistor courtyard should be ~2.4x1.8mm."""
        pads = (
            _make_pad(-0.75, 0.0, 0.6, 0.6),
            _make_pad(0.75, 0.0, 0.6, 0.6),
        )
        fp = _make_fp("R_0603_1608Metric", pads=pads, ref="R1")
        w, h = estimate_courtyard_mm(fp)
        # pad_w = 2.1, pad_h = 0.6, passive ext 0.25/side + 0.25 clearance
        assert 1.5 < w < 4.0, f"0603 width {w} out of range"
        assert 1.0 < h < 3.0, f"0603 height {h} out of range"

    def test_0805_capacitor(self) -> None:
        """0805 cap courtyard should be slightly larger than 0603."""
        pads = (
            _make_pad(-0.95, 0.0, 0.7, 0.8),
            _make_pad(0.95, 0.0, 0.7, 0.8),
        )
        fp = _make_fp("C_0805_2012Metric", pads=pads, ref="C1")
        w, h = estimate_courtyard_mm(fp)
        assert 2.0 < w < 5.0, f"0805 width {w} out of range"
        assert 1.0 < h < 3.0, f"0805 height {h} out of range"


class TestSOTCourtyards:
    """SOT packages have body wider than pads."""

    def test_sot23_5(self) -> None:
        """SOT-23-5 courtyard should be ~3.5x3.5mm."""
        pads = (
            _make_pad(-1.0, -0.95, 0.6, 0.5),
            _make_pad(-1.0, 0.0, 0.6, 0.5),
            _make_pad(-1.0, 0.95, 0.6, 0.5),
            _make_pad(1.0, -0.95, 0.6, 0.5),
            _make_pad(1.0, 0.95, 0.6, 0.5),
        )
        fp = _make_fp("SOT-23-5", pads=pads, ref="U2")
        w, h = estimate_courtyard_mm(fp)
        # SOT extension: 0.75mm/side + 0.25 clearance
        assert w >= 3.0, f"SOT-23-5 width {w} too small"
        assert h >= 3.0, f"SOT-23-5 height {h} too small"


class TestConnectorCourtyards:
    """Connectors get moderate body extension."""

    def test_6pin_terminal_block(self) -> None:
        """6-pin screw terminal courtyard should be ~30x10mm."""
        pads = tuple(
            _make_pad(i * 5.08, 0.0, 1.7, 1.7)
            for i in range(6)
        )
        fp = _make_fp("TerminalBlock_01x06_P5.08mm", pads=pads, ref="J1")
        w, h = estimate_courtyard_mm(fp)
        assert w >= 27.0, f"6-pin TB width {w} too small"
        assert h >= 2.0, f"6-pin TB height {h} too small"


class TestNoPadsFallback:
    """Footprints without pads should fall back to estimate_footprint_size."""

    def test_no_pads_uses_lib_id_estimate(self) -> None:
        fp = _make_fp("R_0603_1608Metric", pads=())
        w, h = estimate_courtyard_mm(fp)
        assert w > 0.0 and h > 0.0


class TestClassification:
    """Package classification picks the right body extension."""

    def test_qfn_not_module(self) -> None:
        """QFN packages should not get module-class extension."""
        pads = tuple(
            _make_pad(x, y, 0.3, 0.3)
            for x in (-3.0, 3.0)
            for y in (-3.0, 3.0)
        )
        fp_qfn = _make_fp("QFN-48_7x7mm", pads=pads)
        fp_mod = _make_fp("ESP32-S3-WROOM-1", pads=pads)
        w_qfn, h_qfn = estimate_courtyard_mm(fp_qfn)
        w_mod, h_mod = estimate_courtyard_mm(fp_mod)
        # Module should have larger height due to antenna extension
        assert h_mod > h_qfn, "Module should be taller than QFN"

    def test_relay_gets_relay_extension(self) -> None:
        pads = (
            _make_pad(-5.0, 0.0, 1.5, 1.5),
            _make_pad(5.0, 0.0, 1.5, 1.5),
        )
        fp = _make_fp("Relay_SPDT_Omron_G6K", pads=pads, ref="K1")
        w, h = estimate_courtyard_mm(fp)
        # Relay extension: 1.0mm/side + 0.25 clearance
        assert w >= 13.0, f"Relay width {w} too small"

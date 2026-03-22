"""Tests for footprint_loader — .kicad_mod → Footprint parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from kicad_pipeline.pcb.footprint_loader import load_kicad_mod


@pytest.fixture()
def simple_mod(tmp_path: Path) -> Path:
    """Create a minimal .kicad_mod file for testing."""
    mod = tmp_path / "test.kicad_mod"
    mod.write_text(textwrap.dedent("""\
        (module test:R_0805 (layer F.Cu)
            (attr smd)
            (fp_text reference REF** (at 0 -1.5) (layer F.SilkS)
                (effects (font (size 1 1) (thickness 0.15)))
            )
            (fp_text value R_0805 (at 0 1.5) (layer F.Fab)
                (effects (font (size 1 1) (thickness 0.15)))
            )
            (pad 1 smd rect (at -1.0 0.0 0.00) (size 1.0 1.2) (layers F.Cu F.Paste F.Mask))
            (pad 2 smd rect (at 1.0 0.0 0.00) (size 1.0 1.2) (layers F.Cu F.Paste F.Mask))
            (fp_line (start -1.5 -0.8) (end 1.5 -0.8) (layer F.SilkS) (width 0.12))
            (fp_line (start -1.5 0.8) (end 1.5 0.8) (layer F.SilkS) (width 0.12))
        )
    """))
    return mod


@pytest.fixture()
def esp32_mod(tmp_path: Path) -> Path:
    """Create a realistic ESP32-style .kicad_mod with multi-pad GND."""
    mod = tmp_path / "ESP32.kicad_mod"
    mod.write_text(textwrap.dedent("""\
        (module easyeda2kicad:ESP32-S3-WROOM-1 (layer F.Cu) (tedit 5DC5F6A4)
            (attr smd)
            (fp_text reference REF** (at 0 -12.89) (layer F.SilkS)
                (effects (font (size 1 1) (thickness 0.15)))
            )
            (fp_text value ESP32-S3-WROOM-1 (at 0 12.89) (layer F.Fab)
                (effects (font (size 1 1) (thickness 0.15)))
            )
            (pad 1 smd rect (at -8.75 -8.89 0.00) (size 1.50 0.90) (layers F.Cu F.Paste F.Mask))
            (pad 2 smd rect (at -8.75 -7.62 0.00) (size 1.50 0.90) (layers F.Cu F.Paste F.Mask))
            (pad 40 smd rect (at 8.75 -8.89 180.00) (size 1.50 0.90) (layers F.Cu F.Paste F.Mask))
            (pad 41 smd rect (at -0.10 0.23 90.00) (size 0.90 0.90) (layers F.Cu F.Paste F.Mask))
            (pad 41 smd rect (at -1.50 0.23 90.00) (size 0.90 0.90) (layers F.Cu F.Paste F.Mask))
            (pad 41 smd rect (at -2.90 0.23 90.00) (size 0.90 0.90) (layers F.Cu F.Paste F.Mask))
            (fp_line (start -9.00 -10.35) (end 9.02 -10.35) (layer F.SilkS) (width 0.25))
            (fp_circle (center -9.00 -16.39) (end -8.97 -16.39) (layer F.Fab) (width 0.06))
        )
    """))
    return mod


@pytest.fixture()
def thru_hole_mod(tmp_path: Path) -> Path:
    """Create a through-hole .kicad_mod."""
    mod = tmp_path / "conn.kicad_mod"
    mod.write_text(textwrap.dedent("""\
        (module test:Conn_1x02 (layer F.Cu)
            (attr through_hole)
            (fp_text reference REF** (at 0 -3) (layer F.SilkS)
                (effects (font (size 1 1) (thickness 0.15)))
            )
            (pad 1 thru_hole rect (at 0 0) (size 1.7 1.7) (drill 1.0) (layers *.Cu *.Mask))
            (pad 2 thru_hole oval (at 0 2.54) (size 1.7 1.7) (drill 1.0) (layers *.Cu *.Mask))
        )
    """))
    return mod


class TestLoadKicadMod:
    """Test basic .kicad_mod parsing."""

    def test_simple_two_pad(self, simple_mod: Path) -> None:
        fp = load_kicad_mod(simple_mod, ref="R1", value="10k")
        assert fp.ref == "R1"
        assert fp.value == "10k"
        assert len(fp.pads) == 2
        assert fp.pads[0].number == "1"
        assert fp.pads[1].number == "2"
        assert fp.attr == "smd"

    def test_pad_positions(self, simple_mod: Path) -> None:
        fp = load_kicad_mod(simple_mod, ref="R1", value="10k")
        assert fp.pads[0].position.x == pytest.approx(-1.0)
        assert fp.pads[0].position.y == pytest.approx(0.0)
        assert fp.pads[1].position.x == pytest.approx(1.0)

    def test_pad_sizes(self, simple_mod: Path) -> None:
        fp = load_kicad_mod(simple_mod, ref="R1", value="10k")
        assert fp.pads[0].size_x == pytest.approx(1.0)
        assert fp.pads[0].size_y == pytest.approx(1.2)

    def test_pad_layers(self, simple_mod: Path) -> None:
        fp = load_kicad_mod(simple_mod, ref="R1", value="10k")
        assert "F.Cu" in fp.pads[0].layers
        assert "F.Paste" in fp.pads[0].layers

    def test_graphics_parsed(self, simple_mod: Path) -> None:
        fp = load_kicad_mod(simple_mod, ref="R1", value="10k")
        assert len(fp.graphics) == 2  # 2 fp_line entries

    def test_texts_override_ref_value(self, simple_mod: Path) -> None:
        fp = load_kicad_mod(simple_mod, ref="R42", value="100k")
        ref_texts = [t for t in fp.texts if t.text_type == "reference"]
        val_texts = [t for t in fp.texts if t.text_type == "value"]
        assert ref_texts[0].text == "R42"
        assert val_texts[0].text == "100k"

    def test_lib_id_includes_prefix(self, simple_mod: Path) -> None:
        fp = load_kicad_mod(simple_mod, ref="R1", value="10k")
        assert ":" in fp.lib_id

    def test_lcsc_attached(self, simple_mod: Path) -> None:
        fp = load_kicad_mod(simple_mod, ref="R1", value="10k", lcsc="C12345")
        assert fp.lcsc == "C12345"

    def test_position_at_origin(self, simple_mod: Path) -> None:
        fp = load_kicad_mod(simple_mod, ref="R1", value="10k")
        assert fp.position.x == 0.0
        assert fp.position.y == 0.0


class TestMultiPadGND:
    """Test handling of multiple pads with the same number (e.g., ESP32 GND)."""

    def test_multiple_pad_41(self, esp32_mod: Path) -> None:
        fp = load_kicad_mod(esp32_mod, ref="U3", value="ESP32-S3-WROOM-1")
        pad_41s = [p for p in fp.pads if p.number == "41"]
        assert len(pad_41s) == 3  # 3 GND pads in test fixture

    def test_total_pad_count(self, esp32_mod: Path) -> None:
        fp = load_kicad_mod(esp32_mod, ref="U3", value="ESP32")
        # 1, 2, 40, 41x3 = 6 total
        assert len(fp.pads) == 6

    def test_circle_graphic(self, esp32_mod: Path) -> None:
        fp = load_kicad_mod(esp32_mod, ref="U3", value="ESP32")
        from kicad_pipeline.models.pcb import FootprintCircle
        circles = [g for g in fp.graphics if isinstance(g, FootprintCircle)]
        assert len(circles) == 1


class TestThroughHole:
    """Test through-hole pad parsing."""

    def test_attr_through_hole(self, thru_hole_mod: Path) -> None:
        fp = load_kicad_mod(thru_hole_mod, ref="J1", value="Conn")
        assert fp.attr == "through_hole"

    def test_drill_diameter(self, thru_hole_mod: Path) -> None:
        fp = load_kicad_mod(thru_hole_mod, ref="J1", value="Conn")
        assert fp.pads[0].drill_diameter == pytest.approx(1.0)
        assert fp.pads[1].drill_diameter == pytest.approx(1.0)

    def test_pad_shapes(self, thru_hole_mod: Path) -> None:
        fp = load_kicad_mod(thru_hole_mod, ref="J1", value="Conn")
        assert fp.pads[0].shape == "rect"
        assert fp.pads[1].shape == "oval"


class TestRealFile:
    """Test loading a real easyeda2kicad-generated file if available."""

    @pytest.fixture()
    def real_esp32_mod(self) -> Path | None:
        """Find real ESP32 .kicad_mod from cache."""
        cache = Path.home() / ".cache" / "kicad-ai-pipeline" / "footprints"
        pretty = cache / "C2913202.pretty"
        if pretty.exists():
            mods = list(pretty.glob("*.kicad_mod"))
            if mods:
                return mods[0]
        # Also try /tmp from easyeda2kicad test
        tmp_pretty = Path("/tmp/test_easyeda.pretty")
        if tmp_pretty.exists():
            mods = list(tmp_pretty.glob("*.kicad_mod"))
            if mods:
                return mods[0]
        return None

    def test_real_esp32_pad_count(self, real_esp32_mod: Path | None) -> None:
        if real_esp32_mod is None:
            pytest.skip("Real ESP32 .kicad_mod not available")
        fp = load_kicad_mod(real_esp32_mod, ref="U3", value="ESP32-S3-WROOM-1")
        # ESP32-S3-WROOM-1: 40 signal pads + 9 GND pads = 49
        assert len(fp.pads) == 49

    def test_real_esp32_pad_positions(self, real_esp32_mod: Path | None) -> None:
        if real_esp32_mod is None:
            pytest.skip("Real ESP32 .kicad_mod not available")
        fp = load_kicad_mod(real_esp32_mod, ref="U3", value="ESP32-S3-WROOM-1")
        pad1 = next(p for p in fp.pads if p.number == "1")
        # Pin 1 should be at left side, x ~ -8.75
        assert pad1.position.x == pytest.approx(-8.75)


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_kicad_mod(tmp_path / "nonexistent.kicad_mod", ref="R1", value="10k")

    def test_invalid_content_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.kicad_mod"
        bad.write_text("(not_a_footprint foo)")
        with pytest.raises(ValueError, match="Expected 'module' or 'footprint'"):
            load_kicad_mod(bad, ref="R1", value="10k")

    def test_empty_pads_ok(self, tmp_path: Path) -> None:
        mod = tmp_path / "nopads.kicad_mod"
        mod.write_text("(module test:empty (layer F.Cu) (attr smd))")
        fp = load_kicad_mod(mod, ref="R1", value="10k")
        assert len(fp.pads) == 0

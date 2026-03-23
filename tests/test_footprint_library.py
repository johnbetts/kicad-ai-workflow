"""Tests for kicad_pipeline.pcb.footprint_library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.models.requirements import (
    Component,
    Net,
    NetConnection,
    Pin,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.pcb.footprint_library import (
    build_footprint_library,
    footprint_name_from_lib_id,
    footprint_to_kicad_mod,
    remap_footprint_lib_ids,
    write_fp_lib_table,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_requirements(*components: Component) -> ProjectRequirements:
    """Build minimal requirements from a list of components."""
    return ProjectRequirements(
        project=ProjectInfo(name="test-project", description="test"),
        components=tuple(components),
        nets=(
            Net(
                name="GND",
                connections=tuple(NetConnection(ref=c.ref, pin="1") for c in components),
            ),
        ),
        features=(),
    )


def _resistor(ref: str = "R1") -> Component:
    return Component(
        ref=ref,
        value="10k",
        footprint="R_0805_2012Metric",
        pins=(
            Pin(number="1", name="1", pin_type="passive"),
            Pin(number="2", name="2", pin_type="passive"),
        ),
    )


def _capacitor(ref: str = "C1") -> Component:
    return Component(
        ref=ref,
        value="100nF",
        footprint="C_0402_1005Metric",
        pins=(
            Pin(number="1", name="1", pin_type="passive"),
            Pin(number="2", name="2", pin_type="passive"),
        ),
    )


# ---------------------------------------------------------------------------
# footprint_name_from_lib_id
# ---------------------------------------------------------------------------


class TestFootprintNameFromLibId:
    def test_with_colon(self) -> None:
        assert footprint_name_from_lib_id("Resistor_SMD:R_0805_2012Metric") == "R_0805_2012Metric"

    def test_kicad_ai_prefix(self) -> None:
        assert footprint_name_from_lib_id("kicad-ai:R_0805") == "R_0805"

    def test_easyeda_prefix(self) -> None:
        assert footprint_name_from_lib_id("easyeda2kicad:C0402") == "C0402"

    def test_bare_name(self) -> None:
        assert footprint_name_from_lib_id("PinHeader_1x14_P2.54mm") == "PinHeader_1x14_P2.54mm"


# ---------------------------------------------------------------------------
# remap_footprint_lib_ids
# ---------------------------------------------------------------------------


class TestRemapFootprintLibIds:
    def test_all_prefixes(self) -> None:
        comps = (
            Component(ref="R1", value="10k", footprint="Resistor_SMD:R_0805", pins=()),
            Component(ref="C1", value="100n", footprint="kicad-ai:C_0402", pins=()),
            Component(ref="U1", value="ESP32", footprint="ESP32-S3-WROOM-1", pins=()),
            Component(ref="J1", value="USB", footprint="easyeda2kicad:USB-C-16P", pins=()),
        )
        req = _minimal_requirements(*comps)
        mapping = remap_footprint_lib_ids(req, "my-board")

        assert mapping["Resistor_SMD:R_0805"] == "my-board:R_0805"
        assert mapping["kicad-ai:C_0402"] == "my-board:C_0402"
        assert mapping["ESP32-S3-WROOM-1"] == "my-board:ESP32-S3-WROOM-1"
        assert mapping["easyeda2kicad:USB-C-16P"] == "my-board:USB-C-16P"


# ---------------------------------------------------------------------------
# footprint_to_kicad_mod
# ---------------------------------------------------------------------------


class TestFootprintToKicadMod:
    @pytest.fixture
    def sample_fp_sexp(self) -> list:
        """A minimal footprint S-expression as produced by _footprint_sexp."""
        return [
            "footprint",
            "Resistor_SMD:R_0805_2012Metric",
            ["layer", "F.Cu"],
            ["at", 50.0, 30.0, 0],
            ["attr", "smd"],
            ["uuid", "abc-123"],
            [
                "property",
                "Reference",
                "R1",
                ["at", 0, -2.0, 0],
                ["layer", "F.SilkS"],
                ["effects", ["font", ["size", 1.0, 1.0]]],
            ],
            [
                "property",
                "Value",
                "10k",
                ["at", 0, 2.0, 0],
                ["layer", "F.Fab"],
                ["effects", ["font", ["size", 1.0, 1.0]]],
            ],
            [
                "property",
                "Footprint",
                "Resistor_SMD:R_0805_2012Metric",
                ["at", 0, 0, 0],
                ["layer", "F.Fab"],
                ["effects", ["font", ["size", 1.0, 1.0]], ["hide", "yes"]],
            ],
            [
                "pad",
                "1",
                "smd",
                "roundrect",
                ["at", -0.51, 0],
                ["size", 0.54, 0.64],
                ["layers", "F.Cu", "F.Mask", "F.Paste"],
                ["net", 1, "GND"],
                ["uuid", "pad-uuid-1"],
            ],
            [
                "pad",
                "2",
                "smd",
                "roundrect",
                ["at", 0.51, 0],
                ["size", 0.54, 0.64],
                ["layers", "F.Cu", "F.Mask", "F.Paste"],
                ["net", 2, "VCC"],
                ["uuid", "pad-uuid-2"],
            ],
        ]

    def test_strips_nets(self, sample_fp_sexp: list) -> None:
        result = footprint_to_kicad_mod(sample_fp_sexp, "R_0805_2012Metric")
        assert "(net " not in result

    def test_strips_board_position(self, sample_fp_sexp: list) -> None:
        result = footprint_to_kicad_mod(sample_fp_sexp, "R_0805_2012Metric")
        # Should not contain the board-relative (at 50 30 0)
        assert "50" not in result or "(at 50" not in result

    def test_template_refs(self, sample_fp_sexp: list) -> None:
        result = footprint_to_kicad_mod(sample_fp_sexp, "R_0805_2012Metric")
        assert "REF**" in result
        assert '"R1"' not in result

    def test_template_value(self, sample_fp_sexp: list) -> None:
        result = footprint_to_kicad_mod(sample_fp_sexp, "R_0805_2012Metric")
        assert '"R_0805_2012Metric"' in result
        assert '"10k"' not in result

    def test_strips_uuid(self, sample_fp_sexp: list) -> None:
        result = footprint_to_kicad_mod(sample_fp_sexp, "R_0805_2012Metric")
        assert "abc-123" not in result
        assert "pad-uuid" not in result

    def test_has_version_and_generator(self, sample_fp_sexp: list) -> None:
        result = footprint_to_kicad_mod(sample_fp_sexp, "R_0805_2012Metric")
        assert "(version " in result
        assert "(generator " in result
        assert "(generator_version " in result

    def test_footprint_name_is_bare(self, sample_fp_sexp: list) -> None:
        result = footprint_to_kicad_mod(sample_fp_sexp, "R_0805_2012Metric")
        # First element after "(footprint " should be quoted footprint name
        assert '(footprint "R_0805_2012Metric"' in result

    def test_has_embedded_fonts(self, sample_fp_sexp: list) -> None:
        result = footprint_to_kicad_mod(sample_fp_sexp, "R_0805_2012Metric")
        assert "(embedded_fonts no)" in result


# ---------------------------------------------------------------------------
# build_footprint_library
# ---------------------------------------------------------------------------


class TestBuildFootprintLibrary:
    def test_creates_pretty_dir(self, tmp_path: Path) -> None:
        req = _minimal_requirements(_resistor())
        pretty_dir = build_footprint_library(req, tmp_path, "test-project")
        assert pretty_dir.exists()
        assert pretty_dir.name == "test-project.pretty"

    def test_generates_kicad_mod_files(self, tmp_path: Path) -> None:
        req = _minimal_requirements(_resistor(), _capacitor())
        pretty_dir = build_footprint_library(req, tmp_path, "test-project")
        mod_files = list(pretty_dir.glob("*.kicad_mod"))
        assert len(mod_files) == 2

    def test_deduplicates_shared_footprints(self, tmp_path: Path) -> None:
        """Two resistors with the same footprint produce only one .kicad_mod."""
        r1 = _resistor("R1")
        r2 = Component(
            ref="R2",
            value="4.7k",
            footprint="R_0805_2012Metric",
            pins=(
                Pin(number="1", name="1", pin_type="passive"),
                Pin(number="2", name="2", pin_type="passive"),
            ),
        )
        req = _minimal_requirements(r1, r2)
        pretty_dir = build_footprint_library(req, tmp_path, "test-project")
        mod_files = list(pretty_dir.glob("*.kicad_mod"))
        assert len(mod_files) == 1
        assert mod_files[0].name == "R_0805_2012Metric.kicad_mod"

    def test_kicad_mod_content_valid(self, tmp_path: Path) -> None:
        req = _minimal_requirements(_resistor())
        pretty_dir = build_footprint_library(req, tmp_path, "test-project")
        mod_file = pretty_dir / "R_0805_2012Metric.kicad_mod"
        content = mod_file.read_text()
        assert content.startswith("(footprint")
        assert "(net " not in content
        assert "REF**" in content


# ---------------------------------------------------------------------------
# write_fp_lib_table
# ---------------------------------------------------------------------------


class TestWriteFpLibTable:
    def test_creates_file(self, tmp_path: Path) -> None:
        path = write_fp_lib_table(tmp_path, "test-project")
        assert path.exists()
        assert path.name == "fp-lib-table"

    def test_content_format(self, tmp_path: Path) -> None:
        write_fp_lib_table(tmp_path, "test-project")
        content = (tmp_path / "fp-lib-table").read_text()
        assert "(fp_lib_table" in content
        assert "(version 7)" in content
        assert '"test-project"' in content
        assert "${KIPRJMOD}/test-project.pretty" in content
        assert '"KiCad"' in content


# ---------------------------------------------------------------------------
# footprint_name_from_lib_id — additional edge cases
# ---------------------------------------------------------------------------


class TestFootprintNameEdgeCases:
    def test_multiple_colons(self) -> None:
        """Only splits on the last colon."""
        assert footprint_name_from_lib_id("a:b:c") == "c"

    def test_empty_string(self) -> None:
        assert footprint_name_from_lib_id("") == ""

    def test_colon_only(self) -> None:
        assert footprint_name_from_lib_id(":") == ""

    def test_jlcpcb_prefix(self) -> None:
        assert footprint_name_from_lib_id("jlcpcb:SOT-23-5") == "SOT-23-5"


# ---------------------------------------------------------------------------
# remap_footprint_lib_ids — edge cases
# ---------------------------------------------------------------------------


class TestRemapEdgeCases:
    def test_empty_components(self) -> None:
        req = _minimal_requirements()
        mapping = remap_footprint_lib_ids(req, "proj")
        assert mapping == {}

    def test_deduplicates_same_footprint(self) -> None:
        """Two components with same footprint produce one mapping entry."""
        r1 = Component(ref="R1", value="10k", footprint="R_0805", pins=())
        r2 = Component(ref="R2", value="4.7k", footprint="R_0805", pins=())
        req = _minimal_requirements(r1, r2)
        mapping = remap_footprint_lib_ids(req, "proj")
        assert len(mapping) == 1
        assert mapping["R_0805"] == "proj:R_0805"


# ---------------------------------------------------------------------------
# footprint_to_kicad_mod — additional edge/error cases
# ---------------------------------------------------------------------------


class TestFootprintToKicadModEdge:
    def test_minimal_sexp(self) -> None:
        """Minimal S-expression with just a name and layer."""
        sexp = ["footprint", "test:R_0805", ["layer", "F.Cu"]]
        result = footprint_to_kicad_mod(sexp, "R_0805")
        assert '(footprint "R_0805"' in result
        assert "(version " in result

    def test_no_properties(self) -> None:
        """S-expression without property nodes should not crash."""
        sexp = [
            "footprint", "test:C_0402",
            ["layer", "F.Cu"],
            ["pad", "1", "smd", "rect", ["at", 0, 0], ["size", 0.5, 0.5]],
        ]
        result = footprint_to_kicad_mod(sexp, "C_0402")
        assert '(footprint "C_0402"' in result

    def test_does_not_modify_original(self) -> None:
        """footprint_to_kicad_mod should not mutate the input list."""
        import copy
        sexp = [
            "footprint", "test:R_0805",
            ["layer", "F.Cu"],
            ["at", 50.0, 30.0, 0],
        ]
        original = copy.deepcopy(sexp)
        footprint_to_kicad_mod(sexp, "R_0805")
        assert sexp == original


# ---------------------------------------------------------------------------
# build_footprint_library — additional edge cases
# ---------------------------------------------------------------------------


class TestBuildFootprintLibraryEdge:
    def test_empty_requirements(self, tmp_path: Path) -> None:
        """No components -> empty .pretty directory."""
        req = _minimal_requirements()
        pretty_dir = build_footprint_library(req, tmp_path, "empty-proj")
        assert pretty_dir.exists()
        mod_files = list(pretty_dir.glob("*.kicad_mod"))
        assert len(mod_files) == 0

    def test_returns_path(self, tmp_path: Path) -> None:
        req = _minimal_requirements(_resistor())
        result = build_footprint_library(req, tmp_path, "proj")
        assert str(result).endswith(".pretty")


# ---------------------------------------------------------------------------
# write_fp_lib_table — edge cases
# ---------------------------------------------------------------------------


class TestWriteFpLibTableEdge:
    def test_overwrites_existing(self, tmp_path: Path) -> None:
        """Writing twice overwrites the file."""
        write_fp_lib_table(tmp_path, "first")
        write_fp_lib_table(tmp_path, "second")
        content = (tmp_path / "fp-lib-table").read_text()
        assert '"second"' in content
        assert '"first"' not in content

    def test_returns_path_type(self, tmp_path: Path) -> None:
        from pathlib import Path as P
        result = write_fp_lib_table(tmp_path, "test")
        assert isinstance(result, P)

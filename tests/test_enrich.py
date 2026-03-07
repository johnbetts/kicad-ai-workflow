"""Tests for B.Cu connector support, 3D model references, and PCB enrichment."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from kicad_pipeline.constants import LAYER_B_CU, LAYER_F_CU
from kicad_pipeline.models.pcb import (
    Footprint3DModel,
    PlacementConstraint,
    PlacementConstraintType,
)
from kicad_pipeline.pcb.board_templates import get_template
from kicad_pipeline.pcb.enrich import (
    _extract_lib_id,
    _extract_ref,
    _find_footprint_nodes,
    _flip_footprint_layer,
    _has_model,
    _inject_model,
    _model_path_from_lib_id,
    enrich_pcb_file,
)
from kicad_pipeline.pcb.footprints import (
    _flip_layer,
    _model_for_package,
    footprint_for_component,
    make_pin_header_socket,
    make_smd_resistor_capacitor,
)
from kicad_pipeline.pcb.placement import LayoutResult
from kicad_pipeline.sexp.parser import parse

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# _flip_layer tests
# ---------------------------------------------------------------------------


class TestFlipLayer:
    def test_front_to_back(self) -> None:
        assert _flip_layer("F.Cu") == "B.Cu"
        assert _flip_layer("F.SilkS") == "B.SilkS"
        assert _flip_layer("F.CrtYd") == "B.CrtYd"
        assert _flip_layer("F.Fab") == "B.Fab"

    def test_back_to_front(self) -> None:
        assert _flip_layer("B.Cu") == "F.Cu"
        assert _flip_layer("B.SilkS") == "F.SilkS"

    def test_no_flip(self) -> None:
        assert _flip_layer("Edge.Cuts") == "Edge.Cuts"
        assert _flip_layer("Dwgs.User") == "Dwgs.User"


# ---------------------------------------------------------------------------
# 3D model mapping tests
# ---------------------------------------------------------------------------


class TestModelForPackage:
    def test_resistor_0805(self) -> None:
        model = _model_for_package("Resistor_SMD:R_0805_2012Metric")
        assert model is not None
        assert "Resistor_SMD.3dshapes" in model.path
        assert "R_0805_2012Metric.step" in model.path

    def test_capacitor_0603(self) -> None:
        model = _model_for_package("Capacitor_SMD:C_0603_1608Metric")
        assert model is not None
        assert "Capacitor_SMD.3dshapes" in model.path

    def test_led_0805(self) -> None:
        model = _model_for_package("LED_SMD:LED_0805_2012Metric")
        assert model is not None
        assert "LED_SMD.3dshapes" in model.path

    def test_sot23(self) -> None:
        model = _model_for_package("Package_TO_SOT_SMD:SOT-23")
        assert model is not None
        assert "SOT-23.step" in model.path

    def test_soic(self) -> None:
        model = _model_for_package("Package_SO:SOIC-8_P1.27mm")
        assert model is not None
        assert "Package_SO.3dshapes" in model.path

    def test_pin_header_fcu(self) -> None:
        model = _model_for_package(
            "Connector_PinHeader_2.54mm:PinHeader_2x20_P2.54mm_Vertical",
        )
        assert model is not None
        assert "PinHeader" in model.path
        assert "Connector_PinHeader" in model.path

    def test_pin_header_bcu_becomes_socket(self) -> None:
        model = _model_for_package(
            "Connector_PinHeader_2.54mm:PinHeader_2x20_P2.54mm_Vertical",
            layer=LAYER_B_CU,
        )
        assert model is not None
        assert "PinSocket" in model.path
        assert "Connector_PinSocket" in model.path

    def test_unknown_returns_none(self) -> None:
        assert _model_for_package("Unknown:SomeWeirdPart") is None


# ---------------------------------------------------------------------------
# Footprint generator B.Cu tests
# ---------------------------------------------------------------------------


class TestPinHeaderSocketBCu:
    def test_bcu_layer(self) -> None:
        fp = make_pin_header_socket("J1", "GPIO", 40, rows=2, layer=LAYER_B_CU)
        assert fp.layer == LAYER_B_CU

    def test_bcu_pinsocket_lib_id(self) -> None:
        fp = make_pin_header_socket("J1", "GPIO", 40, rows=2, layer=LAYER_B_CU)
        assert "PinSocket" in fp.lib_id

    def test_bcu_courtyard_layer(self) -> None:
        fp = make_pin_header_socket("J1", "GPIO", 40, rows=2, layer=LAYER_B_CU)
        crtyd_layers = {g.layer for g in fp.graphics}
        assert "B.CrtYd" in crtyd_layers
        assert "F.CrtYd" not in crtyd_layers

    def test_bcu_silk_layer(self) -> None:
        fp = make_pin_header_socket("J1", "GPIO", 40, rows=2, layer=LAYER_B_CU)
        ref_text = next(t for t in fp.texts if t.text_type == "reference")
        assert ref_text.layer == "B.SilkS"

    def test_bcu_has_3d_model(self) -> None:
        fp = make_pin_header_socket("J1", "GPIO", 40, rows=2, layer=LAYER_B_CU)
        assert len(fp.models) > 0
        assert "PinSocket" in fp.models[0].path

    def test_fcu_default(self) -> None:
        fp = make_pin_header_socket("J1", "GPIO", 40, rows=2)
        assert fp.layer == LAYER_F_CU
        assert "PinHeader" in fp.lib_id


class TestFootprintForComponentLayer:
    def test_connector_with_bcu_layer(self) -> None:
        fp = footprint_for_component("J1", "GPIO", "Conn_02x20_Odd_Even", layer=LAYER_B_CU)
        assert fp.layer == LAYER_B_CU
        # Courtyard/silk should be on B-side layers
        crtyd_layers = {g.layer for g in fp.graphics}
        assert "B.CrtYd" in crtyd_layers

    def test_pinheader_with_bcu_layer(self) -> None:
        fp = footprint_for_component("J1", "GPIO", "PinHeader_2x20_P2.54mm", layer=LAYER_B_CU)
        assert fp.layer == LAYER_B_CU
        assert "PinSocket" in fp.lib_id

    def test_default_fcu(self) -> None:
        fp = footprint_for_component("R1", "10k", "R_0805")
        assert fp.layer == LAYER_F_CU


# ---------------------------------------------------------------------------
# 3D models on SMD generators
# ---------------------------------------------------------------------------


class TestSmdGenerators3DModels:
    def test_resistor_has_model(self) -> None:
        fp = make_smd_resistor_capacitor("R1", "10k", package="0805")
        assert len(fp.models) > 0
        assert "R_0805" in fp.models[0].path

    def test_footprint_for_component_preserves_models(self) -> None:
        fp = footprint_for_component("R1", "10k", "R_0805", lcsc="C12345")
        assert len(fp.models) > 0
        assert fp.lcsc == "C12345"


# ---------------------------------------------------------------------------
# Board template B.Cu tests
# ---------------------------------------------------------------------------


class TestBoardTemplateBCu:
    def test_rpi_hat_gpio_on_bcu(self) -> None:
        tmpl = get_template("RPI_HAT")
        gpio = tmpl.fixed_components[0]
        assert gpio.layer == "B.Cu"

    def test_arduino_default_fcu(self) -> None:
        tmpl = get_template("ARDUINO_UNO")
        # Arduino has no fixed_components, but verify template loads
        assert tmpl.board_width_mm == 68.6


# ---------------------------------------------------------------------------
# LayoutResult layers field
# ---------------------------------------------------------------------------


class TestLayoutResultLayers:
    def test_layers_field(self) -> None:
        from kicad_pipeline.models.pcb import Point

        lr = LayoutResult(
            positions={"J1": Point(32.5, 3.5)},
            rotations={"J1": 0.0},
            layers={"J1": "B.Cu"},
        )
        assert lr.layers is not None
        assert lr.layers["J1"] == "B.Cu"

    def test_layers_default_none(self) -> None:
        from kicad_pipeline.models.pcb import Point

        lr = LayoutResult(
            positions={"R1": Point(10.0, 10.0)},
            rotations={"R1": 0.0},
        )
        assert lr.layers is None


# ---------------------------------------------------------------------------
# Footprint3DModel dataclass
# ---------------------------------------------------------------------------


class TestFootprint3DModel:
    def test_defaults(self) -> None:
        m = Footprint3DModel(path="test.step")
        assert m.offset == (0.0, 0.0, 0.0)
        assert m.scale == (1.0, 1.0, 1.0)
        assert m.rotate == (0.0, 0.0, 0.0)

    def test_custom(self) -> None:
        m = Footprint3DModel(path="a.step", offset=(1.0, 2.0, 3.0))
        assert m.offset == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Enrich post-processor tests
# ---------------------------------------------------------------------------

_MINIMAL_PCB = textwrap.dedent("""\
    (kicad_pcb
      (version 20241229)
      (generator "kicad-ai-pipeline")
      (footprint "Resistor_SMD:R_0805_2012Metric"
        (layer "F.Cu")
        (at 10 10)
        (property "Reference" "R1" (at 0 -2.5) (layer "F.SilkS"))
        (property "Value" "10k" (at 0 2.5) (layer "F.Fab"))
        (pad "1" smd rect (at -1 0) (size 1 1.2) (layers "F.Cu" "F.Paste" "F.Mask"))
      )
      (footprint "Connector_PinHeader_2.54mm:PinHeader_2x20_P2.54mm_Vertical"
        (layer "F.Cu")
        (at 32.5 3.5)
        (property "Reference" "J1" (at 0 -2.5) (layer "F.SilkS"))
        (property "Value" "GPIO" (at 0 2.5) (layer "F.Fab"))
        (pad "1" thru_hole circle (at -12.065 -1.27) (size 1.7 1.7)
          (drill 1) (layers "F.Cu" "B.Cu" "F.Mask" "B.Mask"))
      )
    )
""")


class TestEnrichHelpers:
    def test_find_footprint_nodes(self) -> None:
        tree = parse(_MINIMAL_PCB)
        fps = _find_footprint_nodes(tree)
        assert len(fps) == 2

    def test_extract_ref(self) -> None:
        tree = parse(_MINIMAL_PCB)
        fps = _find_footprint_nodes(tree)
        refs = {_extract_ref(fp) for fp in fps}
        assert refs == {"R1", "J1"}

    def test_extract_lib_id(self) -> None:
        tree = parse(_MINIMAL_PCB)
        fps = _find_footprint_nodes(tree)
        lib_ids = {_extract_lib_id(fp) for fp in fps}
        assert "Resistor_SMD:R_0805_2012Metric" in lib_ids

    def test_has_model_false(self) -> None:
        tree = parse(_MINIMAL_PCB)
        fps = _find_footprint_nodes(tree)
        assert not _has_model(fps[0])

    def test_inject_model(self) -> None:
        tree = parse(_MINIMAL_PCB)
        fps = _find_footprint_nodes(tree)
        _inject_model(fps[0], "test.step")
        assert _has_model(fps[0])

    def test_flip_footprint_layer(self) -> None:
        tree = parse(_MINIMAL_PCB)
        fps = _find_footprint_nodes(tree)
        j1 = next(fp for fp in fps if _extract_ref(fp) == "J1")
        _flip_footprint_layer(j1)
        # Check lib_id changed to PinSocket
        assert "PinSocket" in _extract_lib_id(j1)
        # Check layer node changed
        layer_node = next(c for c in j1 if isinstance(c, list) and c[0] == "layer")
        assert layer_node[1] == "B.Cu"

    def test_model_path_from_lib_id(self) -> None:
        path = _model_path_from_lib_id("Resistor_SMD:R_0805_2012Metric")
        assert path is not None
        assert "R_0805" in path


class TestEnrichPcbFile:
    def test_add_3d_models(self, tmp_path: Path) -> None:
        pcb_file = tmp_path / "test.kicad_pcb"
        pcb_file.write_text(_MINIMAL_PCB)
        out_file = tmp_path / "enriched.kicad_pcb"

        enrich_pcb_file(str(pcb_file), str(out_file), add_3d_models=True)

        content = out_file.read_text()
        assert "(model" in content
        assert "R_0805" in content

    def test_flip_to_bcu(self, tmp_path: Path) -> None:
        pcb_file = tmp_path / "test.kicad_pcb"
        pcb_file.write_text(_MINIMAL_PCB)
        out_file = tmp_path / "enriched.kicad_pcb"

        enrich_pcb_file(str(pcb_file), str(out_file), flip_refs=("J1",))

        content = out_file.read_text()
        # J1 should now be on B.Cu
        tree = parse(content)
        fps = _find_footprint_nodes(tree)
        j1 = next(fp for fp in fps if _extract_ref(fp) == "J1")
        layer_node = next(c for c in j1 if isinstance(c, list) and c[0] == "layer")
        assert layer_node[1] == "B.Cu"
        # lib_id should be PinSocket
        assert "PinSocket" in _extract_lib_id(j1)

    def test_overwrite_in_place(self, tmp_path: Path) -> None:
        pcb_file = tmp_path / "test.kicad_pcb"
        pcb_file.write_text(_MINIMAL_PCB)

        enrich_pcb_file(str(pcb_file), output_path=None, add_3d_models=True)

        content = pcb_file.read_text()
        assert "(model" in content

    def test_no_3d_models_flag(self, tmp_path: Path) -> None:
        pcb_file = tmp_path / "test.kicad_pcb"
        pcb_file.write_text(_MINIMAL_PCB)
        out_file = tmp_path / "enriched.kicad_pcb"

        enrich_pcb_file(str(pcb_file), str(out_file), add_3d_models=False)

        content = out_file.read_text()
        assert "(model" not in content

    def test_preserves_existing_content(self, tmp_path: Path) -> None:
        pcb_file = tmp_path / "test.kicad_pcb"
        pcb_file.write_text(_MINIMAL_PCB)
        out_file = tmp_path / "enriched.kicad_pcb"

        enrich_pcb_file(str(pcb_file), str(out_file), flip_refs=("J1",))

        content = out_file.read_text()
        # R1 should still be on F.Cu
        tree = parse(content)
        fps = _find_footprint_nodes(tree)
        r1 = next(fp for fp in fps if _extract_ref(fp) == "R1")
        layer_node = next(c for c in r1 if isinstance(c, list) and c[0] == "layer")
        assert layer_node[1] == "F.Cu"

    def test_custom_model_var(self, tmp_path: Path) -> None:
        pcb_file = tmp_path / "test.kicad_pcb"
        pcb_file.write_text(_MINIMAL_PCB)
        out_file = tmp_path / "enriched.kicad_pcb"

        enrich_pcb_file(
            str(pcb_file), str(out_file),
            add_3d_models=True,
            model_var="${KICAD8_3DMODEL_DIR}",
        )

        content = out_file.read_text()
        assert "${KICAD8_3DMODEL_DIR}" in content
        assert "${KICAD9_3DMODEL_DIR}" not in content


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestEnrichCli:
    def test_cli_parser(self) -> None:
        from kicad_pipeline.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "enrich", "-p", "test.kicad_pcb",
            "--flip-to-bcu", "J1",
            "--no-3d-models",
        ])
        assert args.command == "enrich"
        assert args.pcb == "test.kicad_pcb"
        assert args.flip_to_bcu == ["J1"]
        assert args.no_3d_models is True

    def test_cli_enrich_command(self, tmp_path: Path) -> None:
        from kicad_pipeline.cli.main import main

        pcb_file = tmp_path / "test.kicad_pcb"
        pcb_file.write_text(_MINIMAL_PCB)
        out_file = tmp_path / "enriched.kicad_pcb"

        result = main([
            "enrich", "-p", str(pcb_file),
            "-o", str(out_file),
            "--flip-to-bcu", "J1",
        ])
        assert result == 0
        content = out_file.read_text()
        assert "(model" in content

    def test_cli_multiple_flip_refs(self) -> None:
        from kicad_pipeline.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "enrich", "-p", "test.kicad_pcb",
            "--flip-to-bcu", "J1",
            "--flip-to-bcu", "J2",
        ])
        assert args.flip_to_bcu == ["J1", "J2"]


# ---------------------------------------------------------------------------
# PlacementConstraint layer field
# ---------------------------------------------------------------------------


class TestPlacementConstraintLayer:
    def test_layer_field(self) -> None:
        c = PlacementConstraint(
            ref="J1",
            constraint_type=PlacementConstraintType.FIXED,
            x=32.5, y=3.5,
            layer="B.Cu",
        )
        assert c.layer == "B.Cu"

    def test_layer_default_none(self) -> None:
        c = PlacementConstraint(
            ref="R1",
            constraint_type=PlacementConstraintType.NEAR,
            target_ref="U1",
        )
        assert c.layer is None

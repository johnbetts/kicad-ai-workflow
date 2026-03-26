"""Tests for the unified footprint agent facade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.exceptions import FootprintError
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    Footprint3DModel,
    Keepout,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.pcb.footprint_agent import (
    FootprintAuditReport,
    ModelStatus,
    PropertyStatus,
    _check_model,
    _check_properties,
    _check_rf_keepout,
    audit_board,
    create_footprint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fp(
    ref: str = "R1",
    lib_id: str = "R_0805:R_0805_2012Metric",
    value: str = "10k",
    lcsc: str | None = "C17414",
    datasheet: str | None = "https://example.com/ds.pdf",
    description: str | None = "Resistor 10k 0805",
    mpn: str | None = "RC0805FR-0710KL",
    manufacturer: str | None = "Yageo",
    models: tuple[Footprint3DModel, ...] | None = None,
    pads: tuple[Pad, ...] | None = None,
) -> Footprint:
    if pads is None:
        pads = (
            Pad(number="1", pad_type="smd", shape="rect",
                position=Point(-1.0, 0.0), size_x=1.0, size_y=1.0,
                layers=("F.Cu",)),
            Pad(number="2", pad_type="smd", shape="rect",
                position=Point(1.0, 0.0), size_x=1.0, size_y=1.0,
                layers=("F.Cu",)),
        )
    if models is None:
        models = (Footprint3DModel(
            path="${KICAD9_3DMODEL_DIR}/Resistor_SMD.3dshapes/R_0805_2012Metric.wrl",
        ),)
    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value=value,
        position=Point(10.0, 20.0),
        pads=pads,
        lcsc=lcsc,
        datasheet=datasheet,
        description=description,
        mpn=mpn,
        manufacturer=manufacturer,
        models=models,
    )


def _make_pcb(
    footprints: tuple[Footprint, ...] = (),
    keepouts: tuple[Keepout, ...] = (),
) -> PCBDesign:
    return PCBDesign(
        outline=BoardOutline(polygon=(
            Point(0, 0), Point(100, 0), Point(100, 80), Point(0, 80),
        )),
        design_rules=DesignRules(),
        nets=(NetEntry(0, ""), NetEntry(1, "GND")),
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=keepouts,
    )


# ---------------------------------------------------------------------------
# TestCreateFootprint
# ---------------------------------------------------------------------------


class TestCreateFootprint:
    """Test create_footprint() delegation and property enrichment."""

    def test_returns_footprint_with_properties(self) -> None:
        fp = create_footprint(
            ref="R1",
            value="10k",
            fp_id="R_0805",
            lcsc="C17414",
            datasheet="https://example.com/ds.pdf",
            description="10k 0805 resistor",
            mpn="RC0805FR-0710KL",
            manufacturer="Yageo",
        )
        assert fp.ref == "R1"
        assert fp.lcsc == "C17414"
        assert fp.datasheet == "https://example.com/ds.pdf"
        assert fp.description == "10k 0805 resistor"
        assert fp.mpn == "RC0805FR-0710KL"
        assert fp.manufacturer == "Yageo"

    def test_returns_footprint_without_optional_properties(self) -> None:
        fp = create_footprint(ref="C1", value="100nF", fp_id="C_0402")
        assert fp.ref == "C1"
        assert fp.mpn is None
        assert fp.manufacturer is None

    def test_raises_footprint_error_on_failure(self) -> None:
        with (
            patch(
                "kicad_pipeline.pcb.footprint_agent.footprint_for_component",
                side_effect=ValueError("bad footprint"),
            ),
            pytest.raises(FootprintError, match="Failed to create footprint"),
        ):
            create_footprint(ref="X1", value="bad", fp_id="INVALID_PACKAGE")

    def test_has_3d_model(self) -> None:
        fp = create_footprint(ref="R1", value="10k", fp_id="R_0805")
        assert len(fp.models) > 0


# ---------------------------------------------------------------------------
# TestCheckModel
# ---------------------------------------------------------------------------


class TestCheckModel:
    """Test _check_model() for model status detection."""

    def test_present(self) -> None:
        fp = _make_fp()
        report = _check_model(fp)
        assert report.status == ModelStatus.PRESENT
        assert report.model_path != ""
        assert report.rotation_warnings == ()

    def test_missing(self) -> None:
        fp = _make_fp(models=(), pads=(
            Pad(number="1", pad_type="smd", shape="rect",
                position=Point(0, 0), size_x=1.0, size_y=1.0,
                layers=("F.Cu",)),
        ))
        report = _check_model(fp)
        assert report.status == ModelStatus.MISSING
        assert report.model_path == ""

    def test_rotation_suspect(self) -> None:
        bad_model = Footprint3DModel(
            path="${KICAD9_3DMODEL_DIR}/test.wrl",
            rotate=(0.0, 0.0, 45.0),  # Not a multiple of 90
        )
        fp = _make_fp(models=(bad_model,))
        report = _check_model(fp)
        assert report.status == ModelStatus.ROTATION_SUSPECT
        assert len(report.rotation_warnings) > 0


# ---------------------------------------------------------------------------
# TestCheckProperties
# ---------------------------------------------------------------------------


class TestCheckProperties:
    """Test _check_properties() for property completeness."""

    def test_all_present(self) -> None:
        fp = _make_fp()
        report = _check_properties(fp)
        assert report.lcsc == PropertyStatus.PRESENT
        assert report.datasheet == PropertyStatus.PRESENT
        assert report.description == PropertyStatus.PRESENT
        assert report.mpn == PropertyStatus.PRESENT
        assert report.manufacturer == PropertyStatus.PRESENT

    def test_missing_lcsc(self) -> None:
        fp = _make_fp(lcsc=None)
        report = _check_properties(fp)
        assert report.lcsc == PropertyStatus.MISSING

    def test_empty_datasheet(self) -> None:
        fp = _make_fp(datasheet="")
        report = _check_properties(fp)
        assert report.datasheet == PropertyStatus.EMPTY

    def test_missing_mpn_and_manufacturer(self) -> None:
        fp = _make_fp(mpn=None, manufacturer=None)
        report = _check_properties(fp)
        assert report.mpn == PropertyStatus.MISSING
        assert report.manufacturer == PropertyStatus.MISSING


# ---------------------------------------------------------------------------
# TestCheckRfKeepout
# ---------------------------------------------------------------------------


class TestCheckRfKeepout:
    """Test _check_rf_keepout() for RF keepout zone auditing."""

    def test_non_rf_component(self) -> None:
        fp = _make_fp(lib_id="R_0805:R_0805_2012Metric")
        report = _check_rf_keepout(fp, ())
        assert not report.requires_keepout
        assert not report.keepout_present

    def test_esp32_without_keepout(self) -> None:
        fp = _make_fp(
            ref="U1",
            lib_id="ESP32-S3-WROOM-1:ESP32-S3-WROOM-1",
        )
        report = _check_rf_keepout(fp, ())
        assert report.requires_keepout
        assert not report.keepout_present

    def test_esp32_with_keepout(self) -> None:
        fp = _make_fp(
            ref="U1",
            lib_id="ESP32-S3-WROOM-1:ESP32-S3-WROOM-1",
        )
        # Keepout polygon surrounding the footprint at (10, 20)
        ko = Keepout(
            polygon=(
                Point(5, 15), Point(15, 15),
                Point(15, 25), Point(5, 25),
            ),
            layers=("F.Cu",),
        )
        report = _check_rf_keepout(fp, (ko,))
        assert report.requires_keepout
        assert report.keepout_present
        assert report.keepout_area_mm2 > 0


# ---------------------------------------------------------------------------
# TestAuditBoard
# ---------------------------------------------------------------------------


class TestAuditBoard:
    """Test audit_board() end-to-end."""

    def test_catches_missing_model(self) -> None:
        fp = _make_fp(ref="R1", models=())
        pcb = _make_pcb(footprints=(fp,))
        report = audit_board(pcb)
        assert report.models_missing == 1
        assert any("missing 3D model" in i for i in report.issues)

    def test_catches_missing_lcsc(self) -> None:
        fp = _make_fp(ref="R1", lcsc=None)
        pcb = _make_pcb(footprints=(fp,))
        report = audit_board(pcb)
        assert report.properties_incomplete == 1
        assert any("LCSC" in i for i in report.issues)

    def test_catches_missing_keepout(self) -> None:
        fp = _make_fp(
            ref="U1",
            lib_id="ESP32-S3-WROOM-1:ESP32-S3-WROOM-1",
        )
        pcb = _make_pcb(footprints=(fp,))
        report = audit_board(pcb)
        assert report.keepouts_required == 1
        assert report.keepouts_present == 0
        assert any("keepout" in i for i in report.issues)

    def test_clean_board_no_issues(self) -> None:
        fp = _make_fp()
        pcb = _make_pcb(footprints=(fp,))
        report = audit_board(pcb)
        assert report.models_missing == 0
        assert report.properties_complete == 1
        assert len(report.issues) == 0

    def test_empty_board(self) -> None:
        pcb = _make_pcb()
        report = audit_board(pcb)
        assert report.total_components == 0
        assert len(report.issues) == 0


# ---------------------------------------------------------------------------
# TestAuditReport
# ---------------------------------------------------------------------------


class TestAuditReport:
    """Test FootprintAuditReport.to_report() text output."""

    def test_clean_report(self) -> None:
        report = FootprintAuditReport(
            entries=(),
            total_components=5,
            models_present=5,
            models_missing=0,
            properties_complete=5,
            properties_incomplete=0,
            keepouts_required=1,
            keepouts_present=1,
            issues=(),
        )
        text = report.to_report()
        assert "Total components: 5" in text
        assert "No issues found" in text

    def test_report_with_issues(self) -> None:
        report = FootprintAuditReport(
            entries=(),
            total_components=3,
            models_present=2,
            models_missing=1,
            properties_complete=2,
            properties_incomplete=1,
            keepouts_required=1,
            keepouts_present=0,
            issues=("R1: missing 3D model", "U1: RF component missing keepout zone"),
        )
        text = report.to_report()
        assert "Issues (2):" in text
        assert "R1: missing 3D model" in text


# ---------------------------------------------------------------------------
# TestLcscPropertyEmission
# ---------------------------------------------------------------------------


class TestLcscPropertyEmission:
    """Test LCSC/MPN/Manufacturer properties appear in _footprint_sexp()."""

    def test_lcsc_property_in_sexp(self) -> None:
        from kicad_pipeline.pcb.builder import _footprint_sexp

        fp = _make_fp(lcsc="C17414", mpn="RC0805FR-0710KL", manufacturer="Yageo")
        sexp = _footprint_sexp(fp)
        # Flatten to string for assertion
        flat = str(sexp)
        assert "LCSC" in flat
        assert "C17414" in flat
        assert "MPN" in flat
        assert "RC0805FR-0710KL" in flat
        assert "Manufacturer" in flat
        assert "Yageo" in flat

    def test_no_lcsc_property_when_none(self) -> None:
        from kicad_pipeline.pcb.builder import _footprint_sexp

        fp = _make_fp(lcsc=None, mpn=None, manufacturer=None)
        sexp = _footprint_sexp(fp)
        flat = str(sexp)
        assert "LCSC" not in flat
        assert "MPN" not in flat
        assert "Manufacturer" not in flat


# ---------------------------------------------------------------------------
# TestBuildLibrary
# ---------------------------------------------------------------------------


class TestBuildLibrary:
    """Test build_library() delegates correctly."""

    def test_delegates_to_underlying_functions(self, tmp_path: object) -> None:
        from pathlib import Path

        with patch(
            "kicad_pipeline.pcb.footprint_agent.build_footprint_library"
        ) as mock_build, patch(
            "kicad_pipeline.pcb.footprint_agent.write_fp_lib_table"
        ) as mock_table:
            from kicad_pipeline.pcb.footprint_agent import build_library

            mock_reqs = MagicMock()
            proj_dir = Path(str(tmp_path))
            mock_build.return_value = proj_dir / "test.pretty"

            result = build_library(mock_reqs, proj_dir, "test")

            mock_build.assert_called_once_with(mock_reqs, proj_dir, "test")
            mock_table.assert_called_once_with(proj_dir, "test")
            assert result == proj_dir / "test.pretty"

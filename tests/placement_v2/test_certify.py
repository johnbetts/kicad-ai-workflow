"""Tests for placement_v2 Stage 0 certification (certify.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.models.pcb import (
    Footprint,
    Footprint3DModel,
    FootprintLine,
    Pad,
    Point,
)
from kicad_pipeline.placement_v2.certify import (
    CertificateMismatchError,
    CertificateStore,
    PadGeom,
    PartCertificate,
    UncertifiedPartError,
    build_certificate,
    certify_board_footprints,
    compute_footprint_sha256,
)
from kicad_pipeline.placement_v2.ir import Severity

CERTIFIED_AT = "2026-06-10"


def _pad(
    number: str,
    x: float,
    y: float,
    size_x: float = 0.6,
    size_y: float = 0.7,
    pad_type: str = "smd",
) -> Pad:
    return Pad(
        number=number,
        pad_type=pad_type,
        shape="roundrect",
        position=Point(x, y),
        size_x=size_x,
        size_y=size_y,
        layers=("F.Cu", "F.Paste", "F.Mask"),
    )


def _fp(
    ref: str = "R1",
    lib_id: str = "Resistor_SMD:R_0402_1005Metric",
    pads: tuple[Pad, ...] | None = None,
    graphics: tuple[FootprintLine, ...] = (),
    models: tuple[Footprint3DModel, ...] = (),
) -> Footprint:
    if pads is None:
        pads = (_pad("1", -0.5, 0.0), _pad("2", 0.5, 0.0))
    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value="10k",
        position=Point(50.0, 50.0),
        pads=pads,
        graphics=graphics,
        models=models,
    )


def _cert(fp: Footprint, key: str) -> PartCertificate:
    return build_certificate(fp, key=key, certified_at=CERTIFIED_AT)


class TestFootprintHash:
    def test_same_footprint_same_hash(self) -> None:
        assert compute_footprint_sha256(_fp()) == compute_footprint_sha256(_fp())

    def test_pad_order_shuffle_same_hash(self) -> None:
        a = _fp(pads=(_pad("1", -0.5, 0.0), _pad("2", 0.5, 0.0)))
        b = _fp(pads=(_pad("2", 0.5, 0.0), _pad("1", -0.5, 0.0)))
        assert compute_footprint_sha256(a) == compute_footprint_sha256(b)

    def test_changed_pad_size_different_hash(self) -> None:
        a = _fp(pads=(_pad("1", -0.5, 0.0, size_x=0.6),))
        b = _fp(pads=(_pad("1", -0.5, 0.0, size_x=0.8),))
        assert compute_footprint_sha256(a) != compute_footprint_sha256(b)

    def test_changed_lib_id_different_hash(self) -> None:
        a = _fp(lib_id="Resistor_SMD:R_0402_1005Metric")
        b = _fp(lib_id="Resistor_SMD:R_0603_1608Metric")
        assert compute_footprint_sha256(a) != compute_footprint_sha256(b)


class TestBuildCertificate:
    def test_pad_map_derived_sorted(self) -> None:
        fp = _fp(
            pads=(
                _pad("2", 0.5, 0.1),
                _pad("1", -0.5, -0.1, pad_type="thru_hole"),
            )
        )
        cert = _cert(fp, "C25804")
        assert cert.pad_map == (
            PadGeom(pin="1", x=-0.5, y=-0.1, width=0.6, height=0.7, through_hole=True),
            PadGeom(pin="2", x=0.5, y=0.1, width=0.6, height=0.7, through_hole=False),
        )
        assert cert.key == "C25804"
        assert cert.footprint_id == fp.lib_id
        assert cert.footprint_sha256 == compute_footprint_sha256(fp)
        assert cert.certified_at == CERTIFIED_AT
        assert cert.kicad_lib_version == "kicad10"

    def test_no_courtyard_fallback_inflates_pad_bbox(self) -> None:
        fp = _fp(pads=(_pad("1", -0.5, 0.0), _pad("2", 0.5, 0.0)))
        cert = _cert(fp, "C1")
        # Pad bbox: x in [-0.8, 0.8], y in [-0.35, 0.35]; inflated 0.25mm.
        assert cert.courtyard == (
            (-1.05, -0.6),
            (1.05, -0.6),
            (1.05, 0.6),
            (-1.05, 0.6),
        )

    def test_explicit_courtyard_used(self) -> None:
        lines = (
            FootprintLine(Point(-2.0, -1.0), Point(2.0, -1.0), "F.CrtYd"),
            FootprintLine(Point(2.0, -1.0), Point(2.0, 1.0), "F.CrtYd"),
            FootprintLine(Point(2.0, 1.0), Point(-2.0, 1.0), "F.CrtYd"),
            FootprintLine(Point(-2.0, 1.0), Point(-2.0, -1.0), "F.CrtYd"),
        )
        cert = _cert(_fp(graphics=lines), "C1")
        assert cert.courtyard == ((-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0))

    def test_body_defaults_to_courtyard(self) -> None:
        cert = _cert(_fp(), "C1")
        assert cert.body == cert.courtyard

    def test_model_env_path_unresolved(self) -> None:
        model = Footprint3DModel(
            path="${KICAD10_3DMODEL_DIR}/Resistor_SMD.3dshapes/R_0402.step",
            offset=(0.1, 0.2, 0.3),
            rotate=(0.0, 0.0, 90.0),
        )
        cert = _cert(_fp(models=(model,)), "C1")
        assert cert.model_path == model.path
        assert cert.model_sha256 is None
        assert cert.resolved is False
        assert cert.model_offset == (0.1, 0.2, 0.3)
        assert cert.model_rotation == (0.0, 0.0, 90.0)

    def test_model_real_file_hashed(self, tmp_path: Path) -> None:
        model_file = tmp_path / "body.step"
        model_file.write_bytes(b"STEP DATA")
        cert = _cert(_fp(models=(Footprint3DModel(path=str(model_file)),)), "C1")
        assert cert.resolved is True
        assert cert.model_sha256 is not None
        assert len(cert.model_sha256) == 64

    def test_no_model_defaults(self) -> None:
        cert = _cert(_fp(), "C1")
        assert cert.model_path is None
        assert cert.model_sha256 is None
        assert cert.resolved is False
        assert cert.model_offset == (0.0, 0.0, 0.0)


class TestCertificateStore:
    def test_load_missing_file_empty_store(self, tmp_path: Path) -> None:
        store = CertificateStore.load(tmp_path / "missing.json")
        assert len(store) == 0

    def test_round_trip_through_json(self, tmp_path: Path) -> None:
        model = Footprint3DModel(path="${KICAD10_3DMODEL_DIR}/r.step", rotate=(0, 0, 90))
        store = (
            CertificateStore()
            .add(_cert(_fp(models=(model,)), "C25804"))
            .add(_cert(_fp(lib_id="Capacitor_SMD:C_0402_1005Metric"), "C1525"))
        )
        path = tmp_path / "certs.json"
        store.save(path)
        loaded = CertificateStore.load(path)
        assert len(loaded) == 2
        assert loaded.lookup("C25804") == store.lookup("C25804")
        assert loaded.lookup("C1525") == store.lookup("C1525")

    def test_add_returns_new_store(self) -> None:
        empty = CertificateStore()
        one = empty.add(_cert(_fp(), "C1"))
        assert len(empty) == 0
        assert len(one) == 1
        assert "C1" in one
        assert "C1" not in empty

    def test_add_replaces_same_key(self) -> None:
        store = CertificateStore().add(_cert(_fp(), "C1"))
        newer = _cert(_fp(lib_id="Other:Lib"), "C1")
        store = store.add(newer)
        assert len(store) == 1
        assert store.lookup("C1").footprint_id == "Other:Lib"

    def test_lookup_exact_match(self) -> None:
        cert = _cert(_fp(), "C25804")
        store = CertificateStore().add(cert)
        assert store.lookup("C25804") is cert

    def test_lookup_unknown_raises_with_remedy(self) -> None:
        store = CertificateStore()
        with pytest.raises(UncertifiedPartError, match="C99999") as excinfo:
            store.lookup("C99999")
        assert "--certify C99999" in str(excinfo.value)

    def test_lookup_no_substring_matching(self) -> None:
        store = CertificateStore().add(_cert(_fp(), "C25804"))
        with pytest.raises(UncertifiedPartError):
            store.lookup("C258")
        with pytest.raises(UncertifiedPartError):
            store.lookup("c25804")  # case-sensitive exact match only

    def test_lookup_verified_passes_for_unchanged_footprint(self) -> None:
        fp = _fp()
        store = CertificateStore().add(_cert(fp, "C1"))
        assert store.lookup_verified("C1", fp).key == "C1"

    def test_lookup_verified_detects_drift(self) -> None:
        store = CertificateStore().add(_cert(_fp(), "C1"))
        drifted = _fp(pads=(_pad("1", -0.5, 0.0, size_x=0.9), _pad("2", 0.5, 0.0)))
        with pytest.raises(CertificateMismatchError, match="no longer matches"):
            store.lookup_verified("C1", drifted)


class TestCertifyBoardFootprints:
    def test_all_certified_returns_empty(self) -> None:
        r1 = _fp(ref="R1")
        c1 = _fp(ref="C1", lib_id="Capacitor_SMD:C_0402_1005Metric")
        store = CertificateStore().add(_cert(r1, "C25804")).add(_cert(c1, "C1525"))
        violations = certify_board_footprints(
            {"R1": r1, "C1": c1}, {"R1": "C25804", "C1": "C1525"}, store
        )
        assert violations == ()

    def test_uncertified_refs_one_critical_violation_each(self) -> None:
        r1 = _fp(ref="R1")
        r2 = _fp(ref="R2")
        violations = certify_board_footprints(
            {"R1": r1, "R2": r2}, {"R1": "C1", "R2": "C2"}, CertificateStore()
        )
        assert len(violations) == 2
        assert [v.refs for v in violations] == [("R1",), ("R2",)]
        for v in violations:
            assert v.severity is Severity.CRITICAL
            assert v.constraint == "certificate"
            assert v.measured == 0.0
            assert v.limit == 0.0

    def test_missing_key_is_violation(self) -> None:
        r1 = _fp(ref="R1")
        violations = certify_board_footprints({"R1": r1}, {}, CertificateStore())
        assert len(violations) == 1
        assert "no part key" in violations[0].message

    def test_drifted_footprint_is_violation(self) -> None:
        certified = _fp(ref="R1")
        store = CertificateStore().add(_cert(certified, "C1"))
        drifted = _fp(ref="R1", pads=(_pad("1", -0.6, 0.0), _pad("2", 0.6, 0.0)))
        violations = certify_board_footprints({"R1": drifted}, {"R1": "C1"}, store)
        assert len(violations) == 1
        assert violations[0].severity is Severity.CRITICAL
        assert "no longer matches" in violations[0].message

    def test_mixed_results_reported_per_ref(self) -> None:
        r1 = _fp(ref="R1")
        r2 = _fp(ref="R2", lib_id="Other:Lib")
        store = CertificateStore().add(_cert(r1, "C1"))
        violations = certify_board_footprints(
            {"R1": r1, "R2": r2}, {"R1": "C1", "R2": "C2"}, store
        )
        assert len(violations) == 1
        assert violations[0].refs == ("R2",)

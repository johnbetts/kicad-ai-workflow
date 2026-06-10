"""End-to-end tests for the v2 placement pipeline."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

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
from kicad_pipeline.pcb.pin_map import origin_to_centroid
from kicad_pipeline.placement_v2.ledger import BuildLedger
from kicad_pipeline.placement_v2.pipeline import (
    run_placement_v2,
    to_kicad_rotation,
)

if TYPE_CHECKING:
    from pathlib import Path


def _pad(number: str, x: float, y: float, w: float = 0.6, h: float = 0.6) -> Pad:
    return Pad(
        number=number, pad_type="smd", shape="rect",
        position=Point(x, y), size_x=w, size_y=h, layers=("F.Cu",),
    )


def _passive_fp(ref: str, value: str = "100nF") -> Footprint:
    return Footprint(
        lib_id="C_0402", ref=ref, value=value, position=Point(0, 0),
        pads=(_pad("1", -0.5, 0.0), _pad("2", 0.5, 0.0)),
    )


def _ic_fp(ref: str) -> Footprint:
    west = [_pad(str(i + 1), -2.5, -1.5 + i * 1.0) for i in range(4)]
    east = [_pad(str(8 - i), 2.5, -1.5 + i * 1.0) for i in range(4)]
    return Footprint(
        lib_id="SOIC-8", ref=ref, value="MCU", position=Point(0, 0),
        pads=(*west, *east),
    )


def _board() -> tuple[ProjectRequirements, dict[str, Footprint]]:
    """IC + two decoupling caps + pull-up resistor."""
    u1 = Component(
        ref="U1", value="MCU", footprint="SOIC-8",
        pins=(
            Pin("8", "VDD", PinType.POWER_IN, net="+3V3"),
            Pin("4", "GND", PinType.POWER_IN, net="GND"),
        ),
    )
    c1 = Component(ref="C1", value="100nF", footprint="C_0402")
    c2 = Component(ref="C2", value="1uF", footprint="C_0402")
    r1 = Component(ref="R1", value="10k", footprint="R_0402")
    reqs = ProjectRequirements(
        project=ProjectInfo(name="t"),
        features=(),
        components=(u1, c1, c2, r1),
        nets=(
            Net("+3V3", (
                NetConnection("U1", "8"), NetConnection("C1", "1"),
                NetConnection("C2", "1"), NetConnection("R1", "1"),
            )),
            Net("GND", (
                NetConnection("U1", "4"), NetConnection("C1", "2"),
                NetConnection("C2", "2"),
            )),
            Net("SDA", (NetConnection("U1", "1"), NetConnection("R1", "2"))),
        ),
    )
    fps = {
        "U1": _ic_fp("U1"),
        "C1": _passive_fp("C1"),
        "C2": _passive_fp("C2", "1uF"),
        "R1": _passive_fp("R1", "10k"),
    }
    return reqs, fps


class TestRunPlacementV2:
    def test_pipeline_completes_and_places_everything(self, tmp_path: Path) -> None:
        reqs, fps = _board()
        result = run_placement_v2(
            reqs, fps,
            ledger_path=tmp_path / "build.jsonl",
            certificate_store_path=tmp_path / "certs.json",
            timestamp="2026-06-10T00:00:00",
        )
        assert result.ok, result.violations
        placed_refs = {ref for ref, _, _ in result.positions}
        assert placed_refs == {"U1", "C1", "C2", "R1"}
        assert result.board_width > 0 and result.board_height > 0

    def test_ledger_records_all_stages_green(self, tmp_path: Path) -> None:
        reqs, fps = _board()
        ledger_path = tmp_path / "build.jsonl"
        result = run_placement_v2(
            reqs, fps,
            ledger_path=ledger_path,
            certificate_store_path=tmp_path / "certs.json",
            timestamp="2026-06-10T00:00:00",
        )
        assert result.ok
        ledger = BuildLedger(ledger_path)
        assert ledger.all_green(("certify", "cells", "floorplan"))

    def test_decoupling_caps_near_ic_in_final_layout(self, tmp_path: Path) -> None:
        reqs, fps = _board()
        result = run_placement_v2(reqs, fps, timestamp="2026-06-10")
        assert result.ok
        pos = {ref: (x, y) for ref, x, y in result.positions}
        rot = result.rotations_dict()
        # Compare centroid positions (origin == centroid for these fps).
        cu = origin_to_centroid(fps["U1"], *pos["U1"], rot["U1"])
        for cap in ("C1", "C2"):
            cc = origin_to_centroid(fps[cap], *pos[cap], rot[cap])
            dist = math.hypot(cc[0] - cu[0], cc[1] - cu[1])
            assert dist <= 8.0, f"{cap} is {dist:.1f}mm from U1 centroid"

    def test_deterministic(self, tmp_path: Path) -> None:
        reqs, fps = _board()
        r1 = run_placement_v2(reqs, fps, timestamp="t")
        r2 = run_placement_v2(reqs, fps, timestamp="t")
        assert r1.positions == r2.positions
        assert r1.rotations == r2.rotations

    def test_certificate_drift_halts_build(self, tmp_path: Path) -> None:
        reqs, fps = _board()
        store_path = tmp_path / "certs.json"
        # First build bootstraps certificates.
        first = run_placement_v2(
            reqs, fps, certificate_store_path=store_path, timestamp="t",
        )
        assert first.ok
        # Drift: same key, different pad geometry.
        drifted = dict(fps)
        drifted["C1"] = Footprint(
            lib_id="C_0402", ref="C1", value="100nF", position=Point(0, 0),
            pads=(_pad("1", -0.8, 0.0), _pad("2", 0.8, 0.0)),  # wrong pitch
        )
        second = run_placement_v2(
            reqs, drifted,
            certificate_store_path=store_path,
            bootstrap_certificates=False,
            timestamp="t",
        )
        assert not second.ok
        assert second.halted_stage == "certify"
        assert any("C1" in v.refs for v in second.violations)

    def test_explicit_board_size_respected(self, tmp_path: Path) -> None:
        reqs, fps = _board()
        result = run_placement_v2(
            reqs, fps, board_width_mm=50.0, board_height_mm=40.0, timestamp="t",
        )
        assert result.ok
        assert (result.board_width, result.board_height) == (50.0, 40.0)
        for _, x, y in result.positions:
            assert 0 <= x <= 50 and 0 <= y <= 40


class TestRotationConvention:
    def test_v2_to_kicad_rotation_mapping(self) -> None:
        assert to_kicad_rotation(0.0) == 0.0
        assert to_kicad_rotation(90.0) == 270.0
        assert to_kicad_rotation(180.0) == 180.0
        assert to_kicad_rotation(270.0) == 90.0

    def test_round_trip_pad_position(self) -> None:
        """A pad placed by v2 math lands at the same board point when
        re-derived through the KiCad convention used by pin_map."""
        from kicad_pipeline.placement_v2.footprint_geom import (
            pad_position_in_frame,
        )

        fp = _passive_fp("C1")
        v2_rot = 90.0
        cx, cy = 30.0, 20.0
        # v2 frame: where the cell math believes pad 1 is.
        v2_pad = pad_position_in_frame(fp, "1", cx, cy, v2_rot)
        # KiCad frame: rotate pad offset by -kicad_rot (pin_map convention).
        kicad_rot = to_kicad_rotation(v2_rot)
        rad = math.radians(-kicad_rot)
        px, py = -0.5, 0.0  # pad 1 offset from centroid
        kicad_pad = (
            cx + px * math.cos(rad) - py * math.sin(rad),
            cy + px * math.sin(rad) + py * math.cos(rad),
        )
        assert v2_pad == pytest.approx(kicad_pad)

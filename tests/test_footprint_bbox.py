"""Tests for FootprintBBox, OriginType, and related placement functions."""

from __future__ import annotations

import math

import pytest

from kicad_pipeline.models.pcb import (
    FootprintBBox,
    Footprint,
    Footprint3DModel,
    FootprintLine,
    OriginType,
    Pad,
    Point,
)
from kicad_pipeline.pcb.constraints import check_courtyard_collisions
from kicad_pipeline.pcb.footprints import (
    compute_footprint_bbox,
    detect_origin_type,
    estimate_footprint_size,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_pad(number: str, x: float, y: float, sx: float = 1.0, sy: float = 0.6) -> Pad:
    """Create a minimal Pad for testing."""
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x=x, y=y),
        size_x=sx,
        size_y=sy,
        layers=("F.Cu", "F.Paste", "F.Mask"),
    )


def _make_fp(
    lib_id: str,
    ref: str,
    pads: tuple[Pad, ...] = (),
    models: tuple[Footprint3DModel, ...] = (),
) -> Footprint:
    """Create a minimal Footprint for testing."""
    return Footprint(
        lib_id=lib_id,
        ref=ref,
        value="test",
        position=Point(x=50.0, y=50.0),
        pads=pads,
        models=models,
    )


# ---------------------------------------------------------------------------
# Phase 1: FootprintBBox unit tests
# ---------------------------------------------------------------------------


class TestFootprintBBox:
    """Tests for the FootprintBBox dataclass."""

    def test_width_height(self) -> None:
        bbox = FootprintBBox(min_x=-1.0, min_y=-0.5, max_x=1.0, max_y=0.5)
        assert bbox.width == pytest.approx(2.0)
        assert bbox.height == pytest.approx(1.0)

    def test_center_offset_symmetric(self) -> None:
        bbox = FootprintBBox(min_x=-1.0, min_y=-0.5, max_x=1.0, max_y=0.5)
        dx, dy = bbox.center_offset
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(0.0)

    def test_center_offset_pin1_origin(self) -> None:
        # Pin-1 origin: body extends from 0 to 20mm
        bbox = FootprintBBox(min_x=-1.0, min_y=0.0, max_x=1.0, max_y=20.0)
        dx, dy = bbox.center_offset
        assert dx == pytest.approx(0.0)
        assert dy == pytest.approx(10.0)

    def test_rotated_0(self) -> None:
        bbox = FootprintBBox(min_x=-2.0, min_y=-1.0, max_x=2.0, max_y=1.0)
        r = bbox.rotated(0.0)
        assert r.width == pytest.approx(4.0)
        assert r.height == pytest.approx(2.0)

    def test_rotated_90_swaps_dims(self) -> None:
        bbox = FootprintBBox(min_x=-2.0, min_y=-1.0, max_x=2.0, max_y=1.0)
        r = bbox.rotated(90.0)
        assert r.width == pytest.approx(2.0, abs=0.01)
        assert r.height == pytest.approx(4.0, abs=0.01)

    def test_rotated_180_same_dims(self) -> None:
        bbox = FootprintBBox(min_x=-2.0, min_y=-1.0, max_x=2.0, max_y=1.0)
        r = bbox.rotated(180.0)
        assert r.width == pytest.approx(4.0, abs=0.01)
        assert r.height == pytest.approx(2.0, abs=0.01)

    def test_rotated_45_expands_aabb(self) -> None:
        bbox = FootprintBBox(min_x=-2.0, min_y=-1.0, max_x=2.0, max_y=1.0)
        r = bbox.rotated(45.0)
        # Diagonal expansion: wider and taller than either original dimension
        assert r.width > 2.0
        assert r.height > 2.0
        # Both dimensions should be about sqrt(2) * max(w,h) / 2 * 2
        diag = math.hypot(4.0, 2.0)
        assert r.width <= diag + 0.01
        assert r.height <= diag + 0.01


# ---------------------------------------------------------------------------
# Phase 1: compute_footprint_bbox tests
# ---------------------------------------------------------------------------


class TestComputeFootprintBBox:
    """Tests for compute_footprint_bbox()."""

    def test_bbox_from_0805_pads(self) -> None:
        """0805 resistor: two pads at ±1.0mm, size 1.0x0.6mm → known geometry."""
        pads = (
            _make_pad("1", x=-1.0, y=0.0, sx=1.0, sy=0.6),
            _make_pad("2", x=1.0, y=0.0, sx=1.0, sy=0.6),
        )
        fp = _make_fp("R_0805:R_0805_2012Metric", "R1", pads=pads)
        bbox = compute_footprint_bbox(fp)
        # Pads span from -1.5 to 1.5 in X, -0.3 to 0.3 in Y
        # Plus courtyard clearance (0.25mm)
        assert bbox.min_x == pytest.approx(-1.75)
        assert bbox.max_x == pytest.approx(1.75)
        assert bbox.min_y == pytest.approx(-0.55)
        assert bbox.max_y == pytest.approx(0.55)

    def test_bbox_from_sot23_pads(self) -> None:
        """SOT-23: asymmetric 3-pin layout."""
        pads = (
            _make_pad("1", x=-0.95, y=1.1, sx=0.7, sy=0.7),
            _make_pad("2", x=0.95, y=1.1, sx=0.7, sy=0.7),
            _make_pad("3", x=0.0, y=-1.1, sx=0.7, sy=0.7),
        )
        fp = _make_fp("Package_TO_SOT_SMD:SOT-23", "Q1", pads=pads)
        bbox = compute_footprint_bbox(fp)
        # X: -(0.95+0.35+0.25) to (0.95+0.35+0.25) = ±1.55
        assert bbox.min_x == pytest.approx(-1.55)
        assert bbox.max_x == pytest.approx(1.55)
        # Y: -(1.1+0.35+0.25) to (1.1+0.35+0.25)
        assert bbox.min_y == pytest.approx(-1.7)
        assert bbox.max_y == pytest.approx(1.7)

    def test_bbox_fallback_no_pads(self) -> None:
        """Empty pads falls back to estimate_footprint_size()."""
        fp = _make_fp("R_0805:R_0805_2012Metric", "R1", pads=())
        bbox = compute_footprint_bbox(fp)
        w, h = estimate_footprint_size("R_0805:R_0805_2012Metric")
        assert bbox.width == pytest.approx(w)
        assert bbox.height == pytest.approx(h)
        # Should be centered
        assert bbox.center_offset[0] == pytest.approx(0.0)
        assert bbox.center_offset[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Phase 1: detect_origin_type tests
# ---------------------------------------------------------------------------


class TestDetectOriginType:
    """Tests for detect_origin_type()."""

    def test_detect_origin_center_smd(self) -> None:
        """0805 resistor with center-origin pads → CENTER."""
        pads = (
            _make_pad("1", x=-1.0, y=0.0),
            _make_pad("2", x=1.0, y=0.0),
        )
        fp = _make_fp("R_0805:R_0805_2012Metric", "R1", pads=pads)
        assert detect_origin_type(fp) == OriginType.CENTER

    def test_detect_origin_pin1_connector(self) -> None:
        """Pin header with pad 1 at origin, body extending downward → PIN1."""
        pads = tuple(
            Pad(
                number=str(i + 1),
                pad_type="thru_hole",
                shape="oval",
                position=Point(x=0.0, y=i * 2.54),
                size_x=1.7,
                size_y=1.7,
                layers=("*.Cu", "*.Mask"),
            )
            for i in range(8)
        )
        fp = _make_fp("Connector:PinHeader_1x08_P2.54mm_Vertical", "J1", pads=pads)
        origin = detect_origin_type(fp)
        assert origin == OriginType.PIN1

    def test_detect_origin_no_pads(self) -> None:
        """No pads → defaults to CENTER."""
        fp = _make_fp("Unknown:Unknown", "U1", pads=())
        assert detect_origin_type(fp) == OriginType.CENTER

    def test_detect_origin_no_pad1(self) -> None:
        """No pad numbered '1' → CENTER."""
        pads = (
            _make_pad("A", x=-1.0, y=0.0),
            _make_pad("B", x=1.0, y=0.0),
        )
        fp = _make_fp("Custom:Custom", "U1", pads=pads)
        assert detect_origin_type(fp) == OriginType.CENTER


# ---------------------------------------------------------------------------
# Phase 2: Courtyard collision with bbox + rotation awareness
# ---------------------------------------------------------------------------


def _make_pin_header_pads(n_pins: int) -> tuple[Pad, ...]:
    """Create THT pin header pads (pin-1 origin, Y-extending)."""
    return tuple(
        Pad(
            number=str(i + 1),
            pad_type="thru_hole",
            shape="oval",
            position=Point(x=0.0, y=i * 2.54),
            size_x=1.7,
            size_y=1.7,
            layers=("*.Cu", "*.Mask"),
        )
        for i in range(n_pins)
    )


class TestCourtyardCollisionBBox:
    """Tests for check_courtyard_collisions() with footprint_bboxes."""

    def test_courtyard_collision_pin1_origin(self) -> None:
        """Two THT connectors at same X but offset Y: bbox detects overlap."""
        # J1 at (10, 10), body extends from ~(10, 10) to (10, 28) (8 pins)
        # J2 at (10, 25), body extends from ~(10, 25) to (10, 43)
        # With center-based logic they might appear non-overlapping;
        # with pin-1 bbox they correctly overlap.
        pads_j1 = _make_pin_header_pads(8)
        pads_j2 = _make_pin_header_pads(8)
        fp_j1 = _make_fp("Connector:PinHeader_1x08", "J1", pads=pads_j1)
        fp_j2 = _make_fp("Connector:PinHeader_1x08", "J2", pads=pads_j2)
        bbox_j1 = compute_footprint_bbox(fp_j1)
        bbox_j2 = compute_footprint_bbox(fp_j2)

        positions = {
            "J1": Point(x=10.0, y=10.0),
            "J2": Point(x=10.0, y=25.0),  # overlaps with J1 body
        }
        fp_sizes = {"J1": (2.0, 20.0), "J2": (2.0, 20.0)}
        bboxes = {"J1": bbox_j1, "J2": bbox_j2}

        violations = check_courtyard_collisions(
            positions, fp_sizes,
            footprint_bboxes=bboxes,
        )
        assert any("J1" in v and "J2" in v for v in violations)

    def test_placement_solver_no_overlap_with_bbox(self) -> None:
        """Two connectors placed far apart should have no collision."""
        pads = _make_pin_header_pads(4)
        fp = _make_fp("Connector:PinHeader_1x04", "J1", pads=pads)
        bbox = compute_footprint_bbox(fp)

        positions = {
            "J1": Point(x=10.0, y=10.0),
            "J2": Point(x=30.0, y=10.0),
        }
        fp_sizes = {"J1": (2.0, 10.0), "J2": (2.0, 10.0)}
        bboxes = {"J1": bbox, "J2": bbox}

        violations = check_courtyard_collisions(
            positions, fp_sizes,
            footprint_bboxes=bboxes,
        )
        assert len(violations) == 0

    def test_rotation_aware_collision(self) -> None:
        """90-degree rotated part uses correct dimensions for collision."""
        pads = (
            _make_pad("1", x=-3.0, y=0.0, sx=1.0, sy=0.6),
            _make_pad("2", x=3.0, y=0.0, sx=1.0, sy=0.6),
        )
        fp = _make_fp("Custom:WideComponent", "U1", pads=pads)
        bbox = compute_footprint_bbox(fp)  # ~7mm wide, ~1.1mm tall

        # At 0° rotation: wide in X → no collision at Y=5
        positions_0 = {
            "U1": Point(x=10.0, y=10.0),
            "U2": Point(x=10.0, y=15.0),
        }
        violations_0 = check_courtyard_collisions(
            positions_0,
            {"U1": (7.0, 1.1), "U2": (7.0, 1.1)},
            footprint_bboxes={"U1": bbox, "U2": bbox},
            rotations={"U1": 0.0, "U2": 0.0},
        )
        assert len(violations_0) == 0

        # At 90° rotation: tall in Y (~7mm) → collision at Y=15
        violations_90 = check_courtyard_collisions(
            positions_0,
            {"U1": (7.0, 1.1), "U2": (7.0, 1.1)},
            footprint_bboxes={"U1": bbox, "U2": bbox},
            rotations={"U1": 90.0, "U2": 90.0},
        )
        assert any("U1" in v and "U2" in v for v in violations_90)

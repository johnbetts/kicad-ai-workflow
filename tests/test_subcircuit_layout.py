"""Tests for subcircuit layout engine (Stage 1).

Verifies that each layout template places components at expected positions,
all components fall within the convex hull polygon, anchors are at origin,
and layout_all_subcircuits returns the correct number of layouts.
"""

from __future__ import annotations

import pytest

from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.optimization.functional_grouper import (
    DetectedSubCircuit,
    SubCircuitType,
    VoltageDomain,
)
from kicad_pipeline.optimization.subcircuit_layout import (
    SubCircuitLayout,
    layout_all_subcircuits,
    layout_subcircuit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FP_SIZES: dict[str, tuple[float, float]] = {
    "K1": (17.7, 15.6),
    "D6": (3.0, 1.5),
    "Q1": (3.0, 1.5),
    "R10": (2.0, 1.0),
    "U1": (8.0, 5.0),
    "C1": (2.0, 1.0),
    "C2": (2.0, 1.0),
    "L1": (4.0, 3.0),
    "Y1": (5.0, 2.0),
    "C3": (2.0, 1.0),
    "C4": (2.0, 1.0),
    "R1": (2.0, 1.0),
    "R2": (2.0, 1.0),
    "C5": (2.0, 1.0),
}


def _make_requirements(
    components: tuple[Component, ...],
    nets: tuple[Net, ...] = (),
    features: tuple[FeatureBlock, ...] = (),
) -> ProjectRequirements:
    """Build minimal ProjectRequirements for layout tests."""
    if not features:
        refs = tuple(c.ref for c in components)
        features = (
            FeatureBlock(
                name="Test",
                description="test block",
                components=refs,
                nets=(),
                subcircuits=(),
            ),
        )
    return ProjectRequirements(
        project=ProjectInfo(name="test", revision="1.0", description="test"),
        features=features,
        components=components,
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=160.0, board_height_mm=80.0),
    )


def _point_in_convex_polygon(
    px: float,
    py: float,
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    """Check if (px, py) is inside a convex polygon using cross-product test.

    Returns True if the point is inside or on the boundary.
    Uses a small epsilon for floating-point tolerance.
    """
    n = len(polygon)
    if n < 3:
        return True  # degenerate polygon, skip check

    eps = 1e-6
    sign: float | None = None
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < eps:
            continue  # on the edge
        if sign is None:
            sign = cross
        elif (cross > 0) != (sign > 0):
            return False
    return True


def _all_corners_in_polygon(
    layout: SubCircuitLayout,
    fp_sizes: dict[str, tuple[float, float]],
) -> bool:
    """Verify every component's 4 bounding-box corners are inside the hull."""
    polygon = layout.polygon
    for ref, (dx, dy, rot) in layout.positions.items():
        w, h = fp_sizes.get(ref, (2.0, 1.0))
        # Swap dimensions for ~90deg rotations
        if abs(rot % 180 - 90) < 10:
            w, h = h, w
        hw, hh = w / 2.0, h / 2.0
        corners = [
            (dx - hw, dy - hh),
            (dx + hw, dy - hh),
            (dx + hw, dy + hh),
            (dx - hw, dy + hh),
        ]
        for cx, cy in corners:
            if not _point_in_convex_polygon(cx, cy, polygon):
                return False
    return True


# ---------------------------------------------------------------------------
# Test: relay driver layout
# ---------------------------------------------------------------------------


def test_relay_driver_layout() -> None:
    """Relay driver: K1 anchor at (0,0,90), support components below."""
    sc = DetectedSubCircuit(
        circuit_type=SubCircuitType.RELAY_DRIVER,
        refs=("K1", "D6", "Q1", "R10"),
        anchor_ref="K1",
        net_connections=("RELAY_COIL_1", "RELAY_COM_1"),
        domain=VoltageDomain.POWER_5V,
        layout_hint="row",
    )
    reqs = _make_requirements(
        components=(
            Component(ref="K1", value="SRD-05VDC", footprint="Relay_SPDT"),
            Component(ref="D6", value="1N4148", footprint="D_SOD-123"),
            Component(ref="Q1", value="2N7002", footprint="SOT-23"),
            Component(ref="R10", value="10k", footprint="R_0805"),
        ),
        nets=(
            Net(
                name="RELAY_COIL_1",
                connections=(
                    NetConnection(ref="K1", pin="5"),
                    NetConnection(ref="Q1", pin="2"),
                ),
            ),
        ),
    )

    layout = layout_subcircuit(sc, FP_SIZES, reqs)

    assert isinstance(layout, SubCircuitLayout)
    assert layout.anchor_ref == "K1"

    # Anchor at origin, rotated 90deg
    kx, ky, krot = layout.positions["K1"]
    assert kx == pytest.approx(0.0)
    assert ky == pytest.approx(0.0)
    assert krot == pytest.approx(90.0)

    # All support components should be below the relay (positive y)
    for ref in ("D6", "Q1", "R10"):
        _, sy, _ = layout.positions[ref]
        assert sy > 0.0, f"{ref} should be below relay (y > 0)"

    # All 4 refs present
    assert set(layout.positions.keys()) == {"K1", "D6", "Q1", "R10"}


# ---------------------------------------------------------------------------
# Test: buck converter layout
# ---------------------------------------------------------------------------


def test_buck_converter_layout() -> None:
    """Buck converter: linear flow Cin -> IC -> L -> Cout."""
    sc = DetectedSubCircuit(
        circuit_type=SubCircuitType.BUCK_CONVERTER,
        refs=("U1", "C1", "L1", "C2"),
        anchor_ref="U1",
        net_connections=("VIN", "SW", "VOUT"),
        domain=VoltageDomain.POWER_5V,
        layout_hint="linear",
    )
    reqs = _make_requirements(
        components=(
            Component(ref="U1", value="TPS5430", footprint="SOIC-8"),
            Component(ref="C1", value="10uF", footprint="C_0805"),
            Component(ref="L1", value="10uH", footprint="IND_5x5"),
            Component(ref="C2", value="22uF", footprint="C_0805"),
        ),
        nets=(
            Net(
                name="SW",
                connections=(
                    NetConnection(ref="U1", pin="3"),
                    NetConnection(ref="L1", pin="1"),
                ),
            ),
            Net(
                name="VOUT",
                connections=(
                    NetConnection(ref="L1", pin="2"),
                    NetConnection(ref="C2", pin="1"),
                ),
            ),
            Net(
                name="VIN",
                connections=(
                    NetConnection(ref="U1", pin="1"),
                    NetConnection(ref="C1", pin="1"),
                ),
            ),
        ),
    )

    layout = layout_subcircuit(sc, FP_SIZES, reqs)

    assert layout.anchor_ref == "U1"

    # Anchor IC at origin
    ux, uy, _ = layout.positions["U1"]
    assert ux == pytest.approx(0.0)
    assert uy == pytest.approx(0.0)

    # Inductor should be to the right of IC (positive x)
    lx, _, _ = layout.positions["L1"]
    assert lx > 0.0, "L1 should be right of U1"

    # All 4 refs present
    assert set(layout.positions.keys()) == {"U1", "C1", "L1", "C2"}


# ---------------------------------------------------------------------------
# Test: decoupling layout
# ---------------------------------------------------------------------------


def test_decoupling_layout() -> None:
    """Decoupling: cap placed below the IC anchor."""
    sc = DetectedSubCircuit(
        circuit_type=SubCircuitType.DECOUPLING,
        refs=("U1", "C5"),
        anchor_ref="U1",
        net_connections=("+3V3", "GND"),
        domain=VoltageDomain.DIGITAL_3V3,
        layout_hint="cluster",
    )
    reqs = _make_requirements(
        components=(
            Component(ref="U1", value="ESP32", footprint="QFN-48"),
            Component(ref="C5", value="100nF", footprint="C_0402"),
        ),
    )

    layout = layout_subcircuit(sc, FP_SIZES, reqs)

    assert layout.anchor_ref == "U1"

    # IC at origin
    ux, uy, _ = layout.positions["U1"]
    assert ux == pytest.approx(0.0)
    assert uy == pytest.approx(0.0)

    # Cap should be adjacent to IC (left column for single cap)
    cx, cy, _ = layout.positions["C5"]
    assert cx != 0.0 or cy != 0.0, "C5 should not be at origin with U1"

    assert set(layout.positions.keys()) == {"U1", "C5"}


# ---------------------------------------------------------------------------
# Test: crystal oscillator layout
# ---------------------------------------------------------------------------


def test_crystal_osc_layout() -> None:
    """Crystal oscillator: Y1 anchor, C3/C4 load caps symmetric."""
    sc = DetectedSubCircuit(
        circuit_type=SubCircuitType.CRYSTAL_OSC,
        refs=("Y1", "C3", "C4"),
        anchor_ref="Y1",
        net_connections=("XIN", "XOUT"),
        domain=VoltageDomain.DIGITAL_3V3,
        layout_hint="cluster",
    )
    reqs = _make_requirements(
        components=(
            Component(ref="Y1", value="32.768kHz", footprint="Crystal_SMD_2012"),
            Component(ref="C3", value="22pF", footprint="C_0402"),
            Component(ref="C4", value="22pF", footprint="C_0402"),
        ),
    )

    layout = layout_subcircuit(sc, FP_SIZES, reqs)

    assert layout.anchor_ref == "Y1"

    # Crystal at origin
    yx, yy, _ = layout.positions["Y1"]
    assert yx == pytest.approx(0.0)
    assert yy == pytest.approx(0.0)

    # Load caps should be symmetric: one left (negative x), one right (positive x)
    c3x, _, _ = layout.positions["C3"]
    c4x, _, _ = layout.positions["C4"]
    assert c3x < 0.0, "C3 should be left of Y1"
    assert c4x > 0.0, "C4 should be right of Y1"

    # Symmetric distance from center
    assert abs(c3x) == pytest.approx(abs(c4x), abs=0.5)

    assert set(layout.positions.keys()) == {"Y1", "C3", "C4"}


# ---------------------------------------------------------------------------
# Test: voltage divider layout
# ---------------------------------------------------------------------------


def test_voltage_divider_layout() -> None:
    """Voltage divider: R1 anchor, R2 below in vertical chain."""
    sc = DetectedSubCircuit(
        circuit_type=SubCircuitType.VOLTAGE_DIVIDER,
        refs=("R1", "R2"),
        anchor_ref="R1",
        net_connections=("VIN_DIV", "VMID", "GND"),
        domain=VoltageDomain.ANALOG,
        layout_hint="linear",
    )
    reqs = _make_requirements(
        components=(
            Component(ref="R1", value="10k", footprint="R_0805"),
            Component(ref="R2", value="4.7k", footprint="R_0805"),
        ),
    )

    layout = layout_subcircuit(sc, FP_SIZES, reqs)

    assert layout.anchor_ref == "R1"

    # R1 at origin
    r1x, r1y, _ = layout.positions["R1"]
    assert r1x == pytest.approx(0.0)
    assert r1y == pytest.approx(0.0)

    # R2 below R1 (positive y), same x
    r2x, r2y, _ = layout.positions["R2"]
    assert r2x == pytest.approx(0.0)
    assert r2y > 0.0, "R2 should be below R1 in the divider chain"

    assert set(layout.positions.keys()) == {"R1", "R2"}


# ---------------------------------------------------------------------------
# Test: layout_all_subcircuits returns correct count
# ---------------------------------------------------------------------------


def test_layout_all_subcircuits() -> None:
    """layout_all_subcircuits returns one layout per subcircuit."""
    subcircuits = [
        DetectedSubCircuit(
            circuit_type=SubCircuitType.RELAY_DRIVER,
            refs=("K1", "D6", "Q1", "R10"),
            anchor_ref="K1",
            net_connections=("RELAY_COIL_1",),
            domain=VoltageDomain.POWER_5V,
        ),
        DetectedSubCircuit(
            circuit_type=SubCircuitType.DECOUPLING,
            refs=("U1", "C5"),
            anchor_ref="U1",
            net_connections=("+3V3",),
            domain=VoltageDomain.DIGITAL_3V3,
        ),
        DetectedSubCircuit(
            circuit_type=SubCircuitType.VOLTAGE_DIVIDER,
            refs=("R1", "R2"),
            anchor_ref="R1",
            net_connections=("VIN_DIV", "GND"),
            domain=VoltageDomain.ANALOG,
        ),
    ]
    reqs = _make_requirements(
        components=(
            Component(ref="K1", value="SRD-05VDC", footprint="Relay_SPDT"),
            Component(ref="D6", value="1N4148", footprint="D_SOD-123"),
            Component(ref="Q1", value="2N7002", footprint="SOT-23"),
            Component(ref="R10", value="10k", footprint="R_0805"),
            Component(ref="U1", value="ESP32", footprint="QFN-48"),
            Component(ref="C5", value="100nF", footprint="C_0402"),
            Component(ref="R1", value="10k", footprint="R_0805"),
            Component(ref="R2", value="4.7k", footprint="R_0805"),
        ),
    )

    layouts = layout_all_subcircuits(subcircuits, FP_SIZES, reqs)

    assert len(layouts) == 3
    assert all(isinstance(lay, SubCircuitLayout) for lay in layouts)

    # Verify each layout matches its subcircuit type
    types = [lay.subcircuit.circuit_type for lay in layouts]
    assert SubCircuitType.RELAY_DRIVER in types
    assert SubCircuitType.DECOUPLING in types
    assert SubCircuitType.VOLTAGE_DIVIDER in types


# ---------------------------------------------------------------------------
# Test: polygon contains all components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc_type, refs, anchor, nets_tuple",
    [
        (
            SubCircuitType.RELAY_DRIVER,
            ("K1", "D6", "Q1", "R10"),
            "K1",
            ("RELAY_COIL_1",),
        ),
        (
            SubCircuitType.BUCK_CONVERTER,
            ("U1", "C1", "L1", "C2"),
            "U1",
            ("VIN", "SW", "VOUT"),
        ),
        (
            SubCircuitType.DECOUPLING,
            ("U1", "C5"),
            "U1",
            ("+3V3",),
        ),
        (
            SubCircuitType.CRYSTAL_OSC,
            ("Y1", "C3", "C4"),
            "Y1",
            ("XIN", "XOUT"),
        ),
        (
            SubCircuitType.VOLTAGE_DIVIDER,
            ("R1", "R2"),
            "R1",
            ("VIN_DIV", "GND"),
        ),
    ],
    ids=["relay", "buck", "decoupling", "crystal", "divider"],
)
def test_polygon_contains_all_components(
    sc_type: SubCircuitType,
    refs: tuple[str, ...],
    anchor: str,
    nets_tuple: tuple[str, ...],
) -> None:
    """All component bounding-box corners must be inside the convex hull."""
    sc = DetectedSubCircuit(
        circuit_type=sc_type,
        refs=refs,
        anchor_ref=anchor,
        net_connections=nets_tuple,
        domain=VoltageDomain.POWER_5V,
    )
    components = tuple(
        Component(ref=r, value="X", footprint="FP") for r in refs
    )
    reqs = _make_requirements(components=components)

    layout = layout_subcircuit(sc, FP_SIZES, reqs)

    assert len(layout.polygon) >= 3, "Polygon must have at least 3 vertices"
    assert _all_corners_in_polygon(layout, FP_SIZES), (
        f"Not all component corners inside polygon for {sc_type.value}"
    )


# ---------------------------------------------------------------------------
# Test: anchor at origin for all template types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sc_type, refs, anchor",
    [
        (SubCircuitType.RELAY_DRIVER, ("K1", "D6", "Q1", "R10"), "K1"),
        (SubCircuitType.BUCK_CONVERTER, ("U1", "C1", "L1", "C2"), "U1"),
        (SubCircuitType.DECOUPLING, ("U1", "C5"), "U1"),
        (SubCircuitType.CRYSTAL_OSC, ("Y1", "C3", "C4"), "Y1"),
        (SubCircuitType.VOLTAGE_DIVIDER, ("R1", "R2"), "R1"),
        (SubCircuitType.LDO_REGULATOR, ("U1", "C1", "C2"), "U1"),
        (SubCircuitType.RC_FILTER, ("R1", "C1"), "R1"),
        (SubCircuitType.RF_ANTENNA, ("U1", "C1"), "U1"),
    ],
    ids=["relay", "buck", "decoupling", "crystal", "divider", "ldo", "rc_filter", "rf_antenna"],
)
def test_anchor_at_origin(
    sc_type: SubCircuitType,
    refs: tuple[str, ...],
    anchor: str,
) -> None:
    """Anchor component position must be (0, 0, *) for every template."""
    sc = DetectedSubCircuit(
        circuit_type=sc_type,
        refs=refs,
        anchor_ref=anchor,
        net_connections=("NET1",),
        domain=VoltageDomain.POWER_5V,
    )
    components = tuple(
        Component(ref=r, value="X", footprint="FP") for r in refs
    )
    reqs = _make_requirements(components=components)

    layout = layout_subcircuit(sc, FP_SIZES, reqs)

    ax, ay, _ = layout.positions[anchor]
    assert ax == pytest.approx(0.0), f"Anchor {anchor} x must be 0, got {ax}"
    assert ay == pytest.approx(0.0), f"Anchor {anchor} y must be 0, got {ay}"

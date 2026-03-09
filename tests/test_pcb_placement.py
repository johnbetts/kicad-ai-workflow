"""Tests for kicad_pipeline.pcb.placement."""

from __future__ import annotations

import pytest

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import BoardOutline, Point
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.pcb.placement import (
    PCB_ZONES,
    PCBZone,
    assign_pcb_zones,
    layout_pcb,
    place_pcb_components,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_requirements(
    extra_components: tuple[Component, ...] = (),
) -> ProjectRequirements:
    """Return a minimal ProjectRequirements for placement tests."""
    mcu = Component(
        ref="U1",
        value="ESP32-S3",
        footprint="ESP32-S3-WROOM-1",
        pins=(
            Pin(number="1", name="GND", pin_type=PinType.POWER_IN, net="GND"),
            Pin(number="2", name="VCC", pin_type=PinType.POWER_IN, net="+3V3"),
        ),
    )
    resistor = Component(
        ref="R1",
        value="10k",
        footprint="R_0805",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, net="+3V3"),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )
    led = Component(
        ref="D1",
        value="RED",
        footprint="LED_0805",
        pins=(
            Pin(number="1", name="K", pin_type=PinType.PASSIVE, net="GND"),
            Pin(number="2", name="A", pin_type=PinType.PASSIVE, net="LED_NET"),
        ),
    )
    all_comps = (mcu, resistor, led, *extra_components)
    fb_mcu = FeatureBlock(
        name="MCU",
        description="Microcontroller",
        components=("U1",),
        nets=("+3V3", "GND"),
        subcircuits=(),
    )
    fb_power = FeatureBlock(
        name="Power",
        description="Power supply",
        components=("R1",),
        nets=("+3V3", "GND"),
        subcircuits=(),
    )
    fb_status = FeatureBlock(
        name="Status",
        description="Status LED",
        components=("D1",),
        nets=("LED_NET",),
        subcircuits=(),
    )
    net_3v3 = Net(name="+3V3", connections=(NetConnection(ref="U1", pin="2"),))
    net_gnd = Net(name="GND", connections=(NetConnection(ref="U1", pin="1"),))
    return ProjectRequirements(
        project=ProjectInfo(name="PlacementTest"),
        features=(fb_mcu, fb_power, fb_status),
        components=all_comps,
        nets=(net_3v3, net_gnd),
    )


def _simple_board() -> BoardOutline:
    """Return a simple rectangular board outline."""
    return BoardOutline(
        polygon=(
            Point(x=0.0, y=0.0),
            Point(x=80.0, y=0.0),
            Point(x=80.0, y=40.0),
            Point(x=0.0, y=40.0),
        )
    )


# ---------------------------------------------------------------------------
# PCB_ZONES dictionary
# ---------------------------------------------------------------------------


def test_pcb_zones_dict_has_required_zones() -> None:
    """PCB_ZONES contains all required functional zones."""
    for zone_name in ("MCU", "USB_POWER", "STATUS", "ETHERNET", "RJ45", "ANALOG", "PERIPHERALS"):
        assert zone_name in PCB_ZONES, f"Zone '{zone_name}' missing from PCB_ZONES"


def test_pcb_zone_is_frozen() -> None:
    """PCBZone instances are frozen dataclasses — attribute assignment raises."""
    zone = PCBZone("TEST_FROZEN", 0.0, 0.0, 10.0, 10.0)
    with pytest.raises((AttributeError, TypeError)):
        zone.name = "CHANGED"  # type: ignore[misc]


def test_pcb_zones_positive_dimensions() -> None:
    """All zones have positive width and height."""
    for name, zone in PCB_ZONES.items():
        assert zone.width > 0.0, f"Zone {name} has non-positive width"
        assert zone.height > 0.0, f"Zone {name} has non-positive height"


# ---------------------------------------------------------------------------
# assign_pcb_zones
# ---------------------------------------------------------------------------


def test_assign_pcb_zones_mcu() -> None:
    """MCU feature name → MCU zone."""
    result = assign_pcb_zones([("U1", "MCU")])
    assert result["U1"].name == "MCU"


def test_assign_pcb_zones_usb() -> None:
    """USB feature name → USB_POWER zone."""
    result = assign_pcb_zones([("J1", "USB")])
    assert result["J1"].name == "USB_POWER"


def test_assign_pcb_zones_led() -> None:
    """LED feature name → STATUS zone."""
    result = assign_pcb_zones([("D1", "LED")])
    assert result["D1"].name == "STATUS"


def test_assign_pcb_zones_ethernet() -> None:
    """Ethernet feature name → ETHERNET zone."""
    result = assign_pcb_zones([("U2", "Ethernet")])
    assert result["U2"].name == "ETHERNET"


def test_assign_pcb_zones_rj45() -> None:
    """rj45 feature name (lowercase) → RJ45 zone."""
    result = assign_pcb_zones([("J2", "rj45")])
    assert result["J2"].name == "RJ45"


def test_assign_pcb_zones_analog() -> None:
    """analog feature name (lowercase) → ANALOG zone."""
    result = assign_pcb_zones([("U3", "analog")])
    assert result["U3"].name == "ANALOG"


def test_assign_pcb_zones_unknown_feature() -> None:
    """Unknown feature name → PERIPHERALS zone."""
    result = assign_pcb_zones([("X1", "SomethingElse")])
    assert result["X1"].name == "PERIPHERALS"


def test_assign_pcb_zones_multiple() -> None:
    """Multiple components receive correct zone assignments."""
    result = assign_pcb_zones([("U1", "MCU"), ("J1", "RJ45"), ("R1", "Unknown")])
    assert result["U1"].name == "MCU"
    assert result["J1"].name == "RJ45"
    assert result["R1"].name == "PERIPHERALS"


# ---------------------------------------------------------------------------
# place_pcb_components
# ---------------------------------------------------------------------------


def test_place_pcb_components_single() -> None:
    """Single component in MCU zone is placed at zone origin."""
    zone = PCB_ZONES["MCU"]
    result = place_pcb_components(["U1"], zone, grid_mm=0.5)
    assert "U1" in result
    pt = result["U1"]
    assert pt.x == pytest.approx(zone.x)
    assert pt.y == pytest.approx(zone.y)


def test_place_pcb_components_multiple() -> None:
    """3 components receive 3 different positions."""
    zone = PCBZone("TEST", 0.0, 0.0, 40.0, 40.0)
    refs = ["R1", "R2", "R3"]
    result = place_pcb_components(refs, zone)
    assert set(result.keys()) == set(refs)
    # All positions should be distinct
    positions = list(result.values())
    for i, pt_a in enumerate(positions):
        for pt_b in positions[i + 1:]:
            assert (pt_a.x, pt_a.y) != (pt_b.x, pt_b.y)


def test_place_pcb_components_grid_snapped() -> None:
    """Positions are exact multiples of grid_mm."""
    zone = PCBZone("TEST", 0.0, 0.0, 40.0, 40.0)
    refs = ["R1", "R2", "C1"]
    grid = 1.0
    result = place_pcb_components(refs, zone, grid_mm=grid)
    for ref, pt in result.items():
        rem_x = round(pt.x % grid, 9)
        rem_y = round(pt.y % grid, 9)
        assert rem_x == pytest.approx(0.0) or rem_x == pytest.approx(grid), (
            f"{ref} x={pt.x} not on grid {grid}"
        )
        assert rem_y == pytest.approx(0.0) or rem_y == pytest.approx(grid), (
            f"{ref} y={pt.y} not on grid {grid}"
        )


def test_place_pcb_components_empty() -> None:
    """Empty ref list returns an empty dict."""
    zone = PCBZone("TEST", 0.0, 0.0, 40.0, 40.0)
    result = place_pcb_components([], zone)
    assert result == {}


def test_place_pcb_components_too_many_raises() -> None:
    """PCBError raised when zone is too small to accommodate all components."""
    zone = PCBZone("TINY", 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(PCBError):
        place_pcb_components(["R1", "R2", "R3", "R4", "R5"], zone, component_spacing_mm=3.0)


def test_snap_internal() -> None:
    """Verify grid snapping via place_pcb_components with grid=1.0 mm."""
    zone = PCBZone("GRID_TEST", 0.3, 0.3, 20.0, 20.0)
    refs = ["R1"]
    result = place_pcb_components(refs, zone, grid_mm=1.0)
    pt = result["R1"]
    # 0.3 snapped to nearest 1.0 mm multiple = 0.0
    assert pt.x == pytest.approx(0.0)
    assert pt.y == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# layout_pcb
# ---------------------------------------------------------------------------


def test_layout_pcb_all_refs_placed() -> None:
    """layout_pcb returns a position for every component in requirements."""
    req = _minimal_requirements()
    board = _simple_board()
    result = layout_pcb(req, board)
    all_refs = {c.ref for c in req.components}
    assert all_refs <= set(result.positions.keys())


def test_layout_pcb_returns_points() -> None:
    """All values returned by layout_pcb are Point instances."""
    req = _minimal_requirements()
    board = _simple_board()
    result = layout_pcb(req, board)
    for ref, pt in result.positions.items():
        assert isinstance(pt, Point), f"{ref} value is not a Point: {pt!r}"


def test_layout_pcb_returns_layout_result() -> None:
    """layout_pcb returns a LayoutResult with positions and rotations."""
    from kicad_pipeline.pcb.placement import LayoutResult

    req = _minimal_requirements()
    board = _simple_board()
    result = layout_pcb(req, board)
    assert isinstance(result, LayoutResult)
    assert isinstance(result.positions, dict)
    assert isinstance(result.rotations, dict)


# ---------------------------------------------------------------------------
# _subcircuit_sort
# ---------------------------------------------------------------------------


def test_subcircuit_sort_groups_caps_near_ics() -> None:
    """Decoupling caps should be placed adjacent to their IC."""
    from kicad_pipeline.pcb.placement import _subcircuit_sort

    ic = Component(
        ref="U1", value="MCU", footprint="QFP-48",
        pins=(
            Pin("1", "VCC", PinType.POWER_IN, net="+3V3"),
            Pin("2", "GND", PinType.POWER_IN, net="GND"),
            Pin("3", "IO1", PinType.BIDIRECTIONAL, net="SIG1"),
        ),
    )
    cap = Component(
        ref="C1", value="100nF", footprint="C_0402",
        pins=(
            Pin("1", "~", PinType.PASSIVE, net="+3V3"),
            Pin("2", "~", PinType.PASSIVE, net="GND"),
        ),
    )
    res = Component(
        ref="R1", value="10k", footprint="R_0402",
        pins=(
            Pin("1", "~", PinType.PASSIVE, net="SIG1"),
            Pin("2", "~", PinType.PASSIVE, net="SIG2"),
        ),
    )
    req = ProjectRequirements(
        project=ProjectInfo(name="SortTest"),
        features=(),
        components=(ic, cap, res),
        nets=(
            Net("+3V3", (NetConnection("U1", "1"), NetConnection("C1", "1"))),
            Net("GND", (NetConnection("U1", "2"), NetConnection("C1", "2"))),
            Net("SIG1", (NetConnection("U1", "3"), NetConnection("R1", "1"))),
            Net("SIG2", (NetConnection("R1", "2"),)),
        ),
    )
    sorted_refs = _subcircuit_sort(["R1", "C1", "U1"], req)
    # U1 should appear before C1, and C1 should be right after U1
    u1_idx = sorted_refs.index("U1")
    c1_idx = sorted_refs.index("C1")
    assert c1_idx == u1_idx + 1, f"C1 at {c1_idx} should follow U1 at {u1_idx}"


def test_subcircuit_sort_preserves_all_refs() -> None:
    """All refs must appear in output."""
    from kicad_pipeline.pcb.placement import _subcircuit_sort

    req = _minimal_requirements()
    refs = [c.ref for c in req.components]
    sorted_refs = _subcircuit_sort(refs, req)
    assert set(sorted_refs) == set(refs)


def test_subcircuit_sort_short_list_unchanged() -> None:
    """Lists of 1-2 refs returned unchanged."""
    from kicad_pipeline.pcb.placement import _subcircuit_sort

    req = _minimal_requirements()
    assert _subcircuit_sort(["U1"], req) == ["U1"]
    assert _subcircuit_sort(["U1", "R1"], req) == ["U1", "R1"]


# ---------------------------------------------------------------------------
# _edge_priority_sort
# ---------------------------------------------------------------------------


def test_edge_priority_sort_moves_wifi_to_edge() -> None:
    """WiFi modules should be moved to CONNECTORS zone."""
    from kicad_pipeline.pcb.placement import _edge_priority_sort

    esp = Component(
        ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32-S3-WROOM-1",
        pins=(Pin("1", "GND", PinType.POWER_IN, net="GND"),),
    )
    cap = Component(ref="C1", value="100nF", footprint="C_0402",
                    pins=(Pin("1", "~", PinType.PASSIVE, net="GND"),))
    req = ProjectRequirements(
        project=ProjectInfo(name="EdgeTest"),
        features=(),
        components=(esp, cap),
        nets=(Net("GND", (NetConnection("U1", "1"), NetConnection("C1", "1"))),),
    )
    groups: dict[str, list[str]] = {"MCU": ["U1", "C1"]}
    result = _edge_priority_sort(groups, req)
    assert "U1" in result.get("CONNECTORS", [])
    assert "U1" not in result["MCU"]


def test_edge_priority_sort_no_wifi_noop() -> None:
    """Without WiFi/edge components, groups remain unchanged."""
    from kicad_pipeline.pcb.placement import _edge_priority_sort

    # Use non-WiFi components so nothing triggers edge movement
    stm = Component(ref="U1", value="STM32F103", footprint="LQFP-48",
                    pins=(Pin("1", "GND", PinType.POWER_IN, net="GND"),))
    res = Component(ref="R1", value="10k", footprint="R_0805",
                    pins=(Pin("1", "~", PinType.PASSIVE, net="GND"),))
    req = ProjectRequirements(
        project=ProjectInfo(name="NoWifi"),
        features=(),
        components=(stm, res),
        nets=(Net("GND", (NetConnection("U1", "1"), NetConnection("R1", "1"))),),
    )
    groups: dict[str, list[str]] = {"MCU": ["U1", "R1"]}
    result = _edge_priority_sort(groups, req)
    assert result["MCU"] == ["U1", "R1"]


# ---------------------------------------------------------------------------
# RELAY zone mapping
# ---------------------------------------------------------------------------


def test_assign_pcb_zones_relay() -> None:
    """Relay feature name maps to RELAY zone."""
    result = assign_pcb_zones([("K1", "Relay Outputs")])
    assert result["K1"].name == "RELAY"


# ---------------------------------------------------------------------------
# place_groups_off_board
# ---------------------------------------------------------------------------


def test_place_groups_off_board_positions_below_board() -> None:
    """All footprints placed by place_groups_off_board have y > board_height."""
    from kicad_pipeline.pcb.placement import place_groups_off_board

    req = _minimal_requirements()
    board_h = 40.0
    fp_sizes = {c.ref: (5.0, 5.0) for c in req.components}
    result = place_groups_off_board(
        footprints=(),
        features=req.features,
        requirements=req,
        board_height_mm=board_h,
        footprint_sizes=fp_sizes,
    )
    for ref, pt in result.positions.items():
        assert pt.y > board_h, f"{ref} at y={pt.y} is not below board (h={board_h})"


def test_place_groups_off_board_groups_separated() -> None:
    """Different FeatureBlock refs have significant X or Y separation."""
    from kicad_pipeline.pcb.placement import place_groups_off_board

    req = _minimal_requirements()
    board_h = 40.0
    fp_sizes = {c.ref: (5.0, 5.0) for c in req.components}
    result = place_groups_off_board(
        footprints=(),
        features=req.features,
        requirements=req,
        board_height_mm=board_h,
        footprint_sizes=fp_sizes,
    )
    # U1 is in MCU, R1 in Power, D1 in Status — all different groups
    u1_pos = result.positions["U1"]
    d1_pos = result.positions["D1"]
    dist = ((u1_pos.x - d1_pos.x) ** 2 + (u1_pos.y - d1_pos.y) ** 2) ** 0.5
    assert dist > 5.0, f"Groups not separated: U1={u1_pos}, D1={d1_pos}"


def test_place_groups_off_board_preserves_fixed() -> None:
    """Refs in fixed_positions keep their positions."""
    from kicad_pipeline.pcb.placement import place_groups_off_board

    req = _minimal_requirements()
    board_h = 40.0
    fp_sizes = {c.ref: (5.0, 5.0) for c in req.components}
    fixed = {"U1": (10.0, 20.0, 90.0)}
    result = place_groups_off_board(
        footprints=(),
        features=req.features,
        requirements=req,
        board_height_mm=board_h,
        footprint_sizes=fp_sizes,
        fixed_positions=fixed,
    )
    assert abs(result.positions["U1"].x - 10.0) < 0.01
    assert abs(result.positions["U1"].y - 20.0) < 0.01
    assert abs(result.rotations["U1"] - 90.0) < 0.01


def test_place_groups_off_board_ungrouped_refs_placed() -> None:
    """Refs not in any FeatureBlock still get placed."""
    from kicad_pipeline.pcb.placement import place_groups_off_board

    extra = Component(
        ref="C99",
        value="100nF",
        footprint="C_0402",
        pins=(
            Pin(number="1", name="~", pin_type=PinType.PASSIVE, net="+3V3"),
            Pin(number="2", name="~", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )
    req = _minimal_requirements(extra_components=(extra,))
    board_h = 40.0
    fp_sizes = {c.ref: (5.0, 5.0) for c in req.components}
    result = place_groups_off_board(
        footprints=(),
        features=req.features,
        requirements=req,
        board_height_mm=board_h,
        footprint_sizes=fp_sizes,
    )
    assert "C99" in result.positions, "Ungrouped ref C99 not placed"


def test_place_groups_off_board_all_refs_placed() -> None:
    """Every component ref gets a position."""
    from kicad_pipeline.pcb.placement import place_groups_off_board

    req = _minimal_requirements()
    board_h = 40.0
    fp_sizes = {c.ref: (5.0, 5.0) for c in req.components}
    result = place_groups_off_board(
        footprints=(),
        features=req.features,
        requirements=req,
        board_height_mm=board_h,
        footprint_sizes=fp_sizes,
    )
    all_refs = {c.ref for c in req.components}
    assert all_refs <= set(result.positions.keys())


def test_place_groups_off_board_subcircuit_clustering() -> None:
    """Relay subcircuit components (K1, Q1, R10, D6) cluster tightly."""
    from kicad_pipeline.pcb.placement import place_groups_off_board

    # Build a relay-driver-like subcircuit: K1, Q1, R10, D6 sharing signal nets
    k1 = Component(
        ref="K1", value="G5V-1-5V", footprint="Relay_SPDT",
        pins=(
            Pin("1", "COIL+", PinType.PASSIVE, net="RELAY_COIL1"),
            Pin("2", "COIL-", PinType.PASSIVE, net="GND"),
            Pin("3", "COM", PinType.PASSIVE, net="RELAY_COM1"),
            Pin("4", "NO", PinType.PASSIVE, net="RELAY_NO1"),
        ),
    )
    q1 = Component(
        ref="Q1", value="2N2222", footprint="SOT-23",
        pins=(
            Pin("B", "B", PinType.INPUT, net="RELAY_DRIVE1"),
            Pin("C", "C", PinType.OUTPUT, net="RELAY_COIL1"),
            Pin("E", "E", PinType.PASSIVE, net="GND"),
        ),
    )
    r10 = Component(
        ref="R10", value="1k", footprint="R_0603",
        pins=(
            Pin("1", "~", PinType.PASSIVE, net="GPIO_RELAY1"),
            Pin("2", "~", PinType.PASSIVE, net="RELAY_DRIVE1"),
        ),
    )
    d6 = Component(
        ref="D6", value="1N4148", footprint="SOD-323",
        pins=(
            Pin("1", "K", PinType.PASSIVE, net="RELAY_COIL1"),
            Pin("2", "A", PinType.PASSIVE, net="+5V"),
        ),
    )
    # Unrelated component in the same feature group
    c5 = Component(
        ref="C5", value="100nF", footprint="C_0603",
        pins=(
            Pin("1", "~", PinType.PASSIVE, net="+5V"),
            Pin("2", "~", PinType.PASSIVE, net="GND"),
        ),
    )
    fb = FeatureBlock(
        name="Relay Outputs",
        description="Relay driver",
        components=("K1", "Q1", "R10", "D6", "C5"),
        nets=("RELAY_COIL1", "RELAY_DRIVE1", "GPIO_RELAY1"),
        subcircuits=("relay_driver",),
    )
    nets = (
        Net("RELAY_COIL1", (
            NetConnection("K1", "1"),
            NetConnection("Q1", "C"),
            NetConnection("D6", "1"),
        )),
        Net("RELAY_DRIVE1", (
            NetConnection("Q1", "B"),
            NetConnection("R10", "2"),
        )),
        Net("GPIO_RELAY1", (NetConnection("R10", "1"),)),
        Net("GND", (
            NetConnection("K1", "2"),
            NetConnection("Q1", "E"),
            NetConnection("C5", "2"),
        )),
        Net("+5V", (
            NetConnection("D6", "2"),
            NetConnection("C5", "1"),
        )),
        Net("RELAY_COM1", (NetConnection("K1", "3"),)),
        Net("RELAY_NO1", (NetConnection("K1", "4"),)),
    )
    req = ProjectRequirements(
        project=ProjectInfo(name="RelayTest"),
        features=(fb,),
        components=(k1, q1, r10, d6, c5),
        nets=nets,
    )
    fp_sizes = {
        "K1": (15.0, 10.0),
        "Q1": (3.0, 3.0),
        "R10": (1.6, 0.8),
        "D6": (2.5, 1.2),
        "C5": (1.6, 0.8),
    }
    result = place_groups_off_board(
        footprints=(),
        features=req.features,
        requirements=req,
        board_height_mm=56.0,
        footprint_sizes=fp_sizes,
    )
    # All refs placed
    assert {c.ref for c in req.components} <= set(result.positions.keys())

    # Subcircuit cluster: K1, Q1, R10, D6 should be within 25mm of each other
    cluster_refs = ["K1", "Q1", "R10", "D6"]
    for i, r1 in enumerate(cluster_refs):
        for r2 in cluster_refs[i + 1:]:
            p1 = result.positions[r1]
            p2 = result.positions[r2]
            dist = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
            assert dist < 25.0, (
                f"{r1} and {r2} too far apart: {dist:.1f}mm "
                f"(expected subcircuit clustering)"
            )

    # Pin-adjacent verification: D6 and Q1 both connect to K1 via RELAY_COIL1,
    # so they should all be near K1 (within relay + standoff distance)
    d6_pos = result.positions["D6"]
    k1_pos = result.positions["K1"]
    q1_pos = result.positions["Q1"]
    dist_d6_k1 = ((d6_pos.x - k1_pos.x) ** 2 + (d6_pos.y - k1_pos.y) ** 2) ** 0.5
    assert dist_d6_k1 < 20.0, (
        f"D6 should be near K1 (within 20mm) but is {dist_d6_k1:.1f}mm away"
    )
    dist_q1_k1 = ((q1_pos.x - k1_pos.x) ** 2 + (q1_pos.y - k1_pos.y) ** 2) ** 0.5
    assert dist_q1_k1 < 15.0, (
        f"Q1 should be near K1 (within 15mm) but is {dist_q1_k1:.1f}mm away"
    )

    # R10 connects to Q1 via RELAY_DRIVE1 — should be closer to Q1 than to K1
    r10_pos = result.positions["R10"]
    dist_r10_q1 = ((r10_pos.x - q1_pos.x) ** 2 + (r10_pos.y - q1_pos.y) ** 2) ** 0.5
    dist_r10_k1 = ((r10_pos.x - k1_pos.x) ** 2 + (r10_pos.y - k1_pos.y) ** 2) ** 0.5
    assert dist_r10_q1 < dist_r10_k1, (
        f"R10 should be closer to Q1 ({dist_r10_q1:.1f}mm) than K1 ({dist_r10_k1:.1f}mm) "
        f"because R10 connects to Q1 via RELAY_DRIVE1"
    )

    # Rotation verification: at least some passives should have non-zero
    # rotation (connected pad facing anchor)
    passive_rotations = [result.rotations.get(r, 0.0) for r in ["R10", "D6"]]
    assert any(rot != 0.0 for rot in passive_rotations), (
        f"Expected non-zero rotation for pin-facing passives, got {passive_rotations}"
    )

    # No overlapping components (check centre-to-centre > min size)
    all_refs_list = list(result.positions.keys())
    for i, r1 in enumerate(all_refs_list):
        for r2 in all_refs_list[i + 1:]:
            p1 = result.positions[r1]
            p2 = result.positions[r2]
            w1, h1 = fp_sizes.get(r1, (5.0, 5.0))
            w2, h2 = fp_sizes.get(r2, (5.0, 5.0))
            # Account for rotation
            rot1 = result.rotations.get(r1, 0.0)
            rot2 = result.rotations.get(r2, 0.0)
            if rot1 in (90.0, 270.0):
                w1, h1 = h1, w1
            if rot2 in (90.0, 270.0):
                w2, h2 = h2, w2
            dx = abs(p1.x - p2.x)
            dy = abs(p1.y - p2.y)
            min_dx = (w1 + w2) / 2.0 - 0.1  # small tolerance
            min_dy = (h1 + h2) / 2.0 - 0.1
            overlaps = dx < min_dx and dy < min_dy
            assert not overlaps, (
                f"{r1} and {r2} overlap: dx={dx:.1f} dy={dy:.1f} "
                f"min_dx={min_dx:.1f} min_dy={min_dy:.1f}"
            )

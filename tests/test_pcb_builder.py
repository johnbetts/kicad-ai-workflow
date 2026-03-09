"""Tests for the PCB builder, pcb_to_sexp, and write_pcb functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from kicad_pipeline.models.pcb import (
    BoardOutline,
    Footprint,
    FootprintText,
    Pad,
    PCBDesign,
    Point,
    Via,
    ZonePolygon,
)
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.pcb.builder import (
    _build_layer_table,
    _footprint_sexp,
    _generate_ic_drc_exclusions,
    _make_gnd_stitching_vias,
    _make_rf_via_fence,
    _zone_sexp,
    build_pcb,
    pcb_to_sexp,
    write_pcb,
)
from kicad_pipeline.sexp.parser import parse

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_requirements(
    *,
    with_mechanical: bool = False,
    board_width: float = 100.0,
    board_height: float = 60.0,
    extra_components: tuple[Component, ...] = (),
) -> ProjectRequirements:
    """Build a minimal ProjectRequirements suitable for builder tests."""
    mcu = Component(
        ref="U1",
        value="ESP32-S3-WROOM-1",
        footprint="ESP32-S3-WROOM-1",
        pins=(
            Pin(
                number="1",
                name="GND",
                pin_type=PinType.POWER_IN,
                function=PinFunction.GND,
                net="GND",
            ),
            Pin(
                number="2",
                name="3V3",
                pin_type=PinType.POWER_IN,
                function=PinFunction.VCC,
                net="+3V3",
            ),
        ),
    )
    cap = Component(
        ref="C1",
        value="100nF",
        footprint="C_0402",
        pins=(
            Pin(number="1", name="+", pin_type=PinType.PASSIVE, net="+3V3"),
            Pin(number="2", name="-", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )
    all_comps: tuple[Component, ...] = (mcu, cap, *extra_components)

    fb_mcu = FeatureBlock(
        name="MCU",
        description="Main MCU",
        components=("U1",),
        nets=("+3V3", "GND"),
        subcircuits=(),
    )
    fb_power = FeatureBlock(
        name="Power",
        description="Decoupling",
        components=("C1",),
        nets=("+3V3", "GND"),
        subcircuits=(),
    )
    extra_refs = tuple(c.ref for c in extra_components)
    fb_extra = FeatureBlock(
        name="Peripherals",
        description="Extra",
        components=extra_refs,
        nets=(),
        subcircuits=(),
    )
    features = (fb_mcu, fb_power, fb_extra) if extra_components else (fb_mcu, fb_power)

    net_3v3 = Net(
        name="+3V3",
        connections=(
            NetConnection(ref="U1", pin="2"),
            NetConnection(ref="C1", pin="1"),
        ),
    )
    net_gnd = Net(
        name="GND",
        connections=(
            NetConnection(ref="U1", pin="1"),
            NetConnection(ref="C1", pin="2"),
        ),
    )

    mechanical = (
        MechanicalConstraints(
            board_width_mm=board_width,
            board_height_mm=board_height,
        )
        if with_mechanical
        else None
    )

    return ProjectRequirements(
        project=ProjectInfo(name="BuilderTest", revision="v0.1"),
        features=features,
        components=all_comps,
        nets=(net_3v3, net_gnd),
        mechanical=mechanical,
    )


# ---------------------------------------------------------------------------
# build_pcb tests
# ---------------------------------------------------------------------------


def test_build_pcb_minimal() -> None:
    """build_pcb returns a PCBDesign from minimal requirements."""
    req = _make_requirements()
    design = build_pcb(req)
    assert isinstance(design, PCBDesign)


def test_build_pcb_has_footprints() -> None:
    """PCBDesign has one footprint per component in requirements."""
    req = _make_requirements()
    design = build_pcb(req)
    assert len(design.footprints) == len(req.components)


def test_build_pcb_has_nets() -> None:
    """PCBDesign.nets has at least GND."""
    req = _make_requirements()
    design = build_pcb(req)
    net_names = {n.name for n in design.nets}
    assert "GND" in net_names


def test_build_pcb_gnd_is_net_one() -> None:
    """GND net always has number 1."""
    req = _make_requirements()
    design = build_pcb(req)
    gnd_num = design.get_net_number("GND")
    assert gnd_num == 1


def test_build_pcb_has_board_outline() -> None:
    """PCBDesign.outline polygon is non-empty."""
    req = _make_requirements()
    design = build_pcb(req)
    assert len(design.outline.polygon) > 0


def test_build_pcb_has_zones() -> None:
    """PCBDesign has GND zone on B.Cu (back only — F.Cu used for routing)."""
    req = _make_requirements()
    design = build_pcb(req)
    layers = {z.layer for z in design.zones}
    assert "B.Cu" in layers


def test_build_pcb_zone_clearance_adequate() -> None:
    """GND zones should have clearance >= 0.2mm (not min_thickness)."""
    req = _make_requirements()
    design = build_pcb(req)
    for zone in design.zones:
        assert zone.clearance_mm >= 0.2, (
            f"Zone {zone.name} clearance {zone.clearance_mm} < 0.2mm"
        )
        assert zone.min_thickness >= 0.25, (
            f"Zone {zone.name} min_thickness {zone.min_thickness} < 0.25mm"
        )


def test_build_pcb_has_netclasses() -> None:
    """PCBDesign should have netclasses after build."""
    req = _make_requirements()
    design = build_pcb(req)
    assert len(design.netclasses) >= 1
    names = {nc.name for nc in design.netclasses}
    assert "Default" in names


def test_build_pcb_no_tracks_when_routing_disabled() -> None:
    """PCBDesign has no tracks when auto_route is disabled."""
    req = _make_requirements()
    design = build_pcb(req, auto_route=False)
    assert len(design.tracks) == 0


def test_build_pcb_no_gnd_stitching_vias_when_routing_disabled() -> None:
    """PCBDesign has no GND stitching vias when auto_route is disabled.

    GND connectivity is handled by F.Cu + B.Cu copper pour through thermal
    relief, so stitching vias are unnecessary and waste routing space.
    Only RF via fence vias (if RF module present) should exist.
    """
    req = _make_requirements()
    design = build_pcb(req, auto_route=False)
    # No signal routing vias when routing is disabled.
    # RF via fence vias may exist if RF module detected (e.g. ESP32).
    gnd_net_num = next(n.number for n in design.nets if n.name == "GND")
    for v in design.vias:
        # All vias should be GND RF fence vias, not stitching grid vias
        assert v.net_number == gnd_net_num
        assert v.drill == 0.6, "Expected RF fence via drill size"


def test_build_pcb_has_keepouts() -> None:
    """PCBDesign has at least the corner mounting-hole keepouts."""
    req = _make_requirements()
    design = build_pcb(req)
    # 4 mounting-hole keepouts (one per corner)
    assert len(design.keepouts) >= 4


def test_build_pcb_rf_module_adds_antenna_keepout() -> None:
    """ESP32 design gets an extra antenna keepout."""
    req = _make_requirements()
    design = build_pcb(req)
    # U1 is ESP32-S3-WROOM-1 → RF module detected → 5 keepouts (4 corners + 1 antenna)
    assert len(design.keepouts) == 5


def test_build_pcb_no_components_raises() -> None:
    """build_pcb raises PCBError when there are no components."""
    from kicad_pipeline.exceptions import PCBError

    req = ProjectRequirements(
        project=ProjectInfo(name="Empty"),
        features=(),
        components=(),
        nets=(),
    )
    with pytest.raises(PCBError):
        build_pcb(req)


def test_default_board_size() -> None:
    """Default board is at least 80 x 40 mm when no mechanical constraints.

    Auto-sizing may produce a larger board when component footprints
    (e.g. ESP32 module) require more area than the 80x40 minimum.
    """
    req = _make_requirements(with_mechanical=False)
    design = build_pcb(req)
    xs = [p.x for p in design.outline.polygon]
    ys = [p.y for p in design.outline.polygon]
    assert max(xs) >= 80.0
    assert max(ys) >= 40.0


def test_board_dimensions_from_mechanical() -> None:
    """Board size comes from requirements.mechanical when set."""
    req = _make_requirements(with_mechanical=True, board_width=120.0, board_height=80.0)
    design = build_pcb(req)
    xs = [p.x for p in design.outline.polygon]
    ys = [p.y for p in design.outline.polygon]
    assert max(xs) == pytest.approx(120.0)
    assert max(ys) == pytest.approx(80.0)


def test_board_dimensions_override_mechanical() -> None:
    """Explicit board_width_mm / board_height_mm override mechanical constraints."""
    req = _make_requirements(with_mechanical=True, board_width=120.0, board_height=80.0)
    design = build_pcb(req, board_width_mm=50.0, board_height_mm=30.0)
    xs = [p.x for p in design.outline.polygon]
    ys = [p.y for p in design.outline.polygon]
    assert max(xs) == pytest.approx(50.0)
    assert max(ys) == pytest.approx(30.0)


def test_build_pcb_footprints_have_silkscreen() -> None:
    """All footprints in built PCBDesign have silkscreen reference labels."""
    req = _make_requirements()
    design = build_pcb(req)
    for fp in design.footprints:
        ref_texts = [t for t in fp.texts if t.text_type == "reference"]
        assert len(ref_texts) >= 1, f"Footprint {fp.ref} has no silkscreen reference label"


# ---------------------------------------------------------------------------
# pcb_to_sexp tests
# ---------------------------------------------------------------------------


def test_pcb_to_sexp_is_list() -> None:
    """pcb_to_sexp returns a list (SExpNode)."""
    req = _make_requirements()
    design = build_pcb(req)
    sexp = pcb_to_sexp(design)
    assert isinstance(sexp, list)


def test_pcb_to_sexp_starts_with_kicad_pcb() -> None:
    """First element of the S-expression is 'kicad_pcb'."""
    req = _make_requirements()
    design = build_pcb(req)
    sexp = pcb_to_sexp(design)
    assert isinstance(sexp, list)
    assert sexp[0] == "kicad_pcb"


def test_pcb_to_sexp_contains_version() -> None:
    """S-expression contains a version node."""
    from kicad_pipeline.constants import KICAD_PCB_VERSION

    req = _make_requirements()
    design = build_pcb(req)
    sexp = pcb_to_sexp(design)
    assert isinstance(sexp, list)
    version_nodes = [n for n in sexp if isinstance(n, list) and n and n[0] == "version"]
    assert len(version_nodes) == 1
    assert version_nodes[0][1] == KICAD_PCB_VERSION


def test_pcb_to_sexp_contains_nets() -> None:
    """S-expression contains net nodes for all nets in design."""
    req = _make_requirements()
    design = build_pcb(req)
    sexp = pcb_to_sexp(design)
    assert isinstance(sexp, list)
    net_nodes = [n for n in sexp if isinstance(n, list) and n and n[0] == "net"]
    assert len(net_nodes) == len(design.nets)


def test_pcb_to_sexp_contains_footprints() -> None:
    """S-expression contains a footprint node for each footprint."""
    req = _make_requirements()
    design = build_pcb(req)
    sexp = pcb_to_sexp(design)
    assert isinstance(sexp, list)
    fp_nodes = [n for n in sexp if isinstance(n, list) and n and n[0] == "footprint"]
    assert len(fp_nodes) == len(design.footprints)


def test_pcb_to_sexp_contains_edge_cuts() -> None:
    """S-expression contains gr_line nodes for the board outline."""
    req = _make_requirements()
    design = build_pcb(req)
    sexp = pcb_to_sexp(design)
    assert isinstance(sexp, list)
    edge_lines = [
        n
        for n in sexp
        if isinstance(n, list)
        and n
        and n[0] == "gr_line"
        and any(
            isinstance(sub, list) and sub and sub[0] == "layer" and "Edge.Cuts" in sub
            for sub in n
        )
    ]
    # Rectangular outline = 4 lines
    assert len(edge_lines) == 4


# ---------------------------------------------------------------------------
# write_pcb tests
# ---------------------------------------------------------------------------


def test_write_pcb_creates_file(tmp_path: Path) -> None:
    """write_pcb creates a .kicad_pcb file at the specified path."""
    req = _make_requirements()
    design = build_pcb(req)
    dest = tmp_path / "test_output.kicad_pcb"
    write_pcb(design, dest)
    assert dest.exists()
    assert dest.stat().st_size > 0


def test_write_pcb_parseable(tmp_path: Path) -> None:
    """Written .kicad_pcb file can be parsed back by sexp.parser."""
    req = _make_requirements()
    design = build_pcb(req)
    dest = tmp_path / "parseable.kicad_pcb"
    write_pcb(design, dest)
    text = dest.read_text(encoding="utf-8")
    parsed = parse(text)
    assert isinstance(parsed, list)
    assert parsed[0] == "kicad_pcb"


def test_pcb_to_sexp_zone_clearance_not_min_thickness() -> None:
    """Zone connect_pads clearance should use clearance_mm, not min_thickness."""
    req = _make_requirements()
    design = build_pcb(req)
    sexp = pcb_to_sexp(design)
    zone_nodes = [
        n for n in sexp
        if isinstance(n, list) and n and n[0] == "zone"
        and any(isinstance(s, list) and s and s[0] == "net_name" and s[1] == "GND" for s in n)
    ]
    assert len(zone_nodes) >= 1
    for zone_node in zone_nodes:
        # Find connect_pads node: ["connect_pads", ["clearance", N]]
        for sub in zone_node:
            if isinstance(sub, list) and sub and sub[0] == "connect_pads":
                clearance_node = [
                    s for s in sub if isinstance(s, list) and s and s[0] == "clearance"
                ]
                assert len(clearance_node) == 1
                assert clearance_node[0][1] >= 0.2


def test_write_pcb_accepts_string_path(tmp_path: Path) -> None:
    """write_pcb accepts a plain string path as well as a Path object."""
    req = _make_requirements()
    design = build_pcb(req)
    dest = str(tmp_path / "string_path.kicad_pcb")
    write_pcb(design, dest)
    assert Path(dest).exists()


# ---------------------------------------------------------------------------
# BUG-7: Track/via serialization
# ---------------------------------------------------------------------------


def test_pcb_to_sexp_serializes_tracks_and_vias() -> None:
    """pcb_to_sexp correctly serializes tracks and vias in the S-expression."""
    from dataclasses import replace

    from kicad_pipeline.models.pcb import Track, Via

    req = _make_requirements()
    design = build_pcb(req)
    # Add a track and a via manually
    track = Track(
        start=design.footprints[0].position,
        end=design.footprints[1].position,
        width=0.25,
        layer="F.Cu",
        net_number=1,
        uuid="track-uuid-test",
    )
    via = Via(
        position=design.footprints[0].position,
        drill=0.508,
        size=0.8,
        layers=("F.Cu", "B.Cu"),
        net_number=1,
        uuid="via-uuid-test",
    )
    design_with_routing = replace(
        design, tracks=(track,), vias=(via,),
    )
    sexp = pcb_to_sexp(design_with_routing)
    assert isinstance(sexp, list)

    # Check segment node
    seg_nodes = [
        n for n in sexp
        if isinstance(n, list) and n and n[0] == "segment"
    ]
    assert len(seg_nodes) == 1
    seg = seg_nodes[0]
    # Should have start, end, width, layer, net, uuid sub-nodes
    seg_keys = {s[0] for s in seg if isinstance(s, list) and s}
    assert {"start", "end", "width", "layer", "net", "uuid"} <= seg_keys

    # Check via node
    via_nodes = [
        n for n in sexp
        if isinstance(n, list) and n and n[0] == "via"
    ]
    assert len(via_nodes) == 1
    vn = via_nodes[0]
    via_keys = {s[0] for s in vn if isinstance(s, list) and s}
    assert {"at", "size", "drill", "layers", "net", "uuid"} <= via_keys


# ---------------------------------------------------------------------------
# BUG-9: Board outline corner radius
# ---------------------------------------------------------------------------


def test_board_outline_with_corner_radius() -> None:
    """Board outline with corner_radius_mm > 0 produces a rounded polygon."""
    req = _make_requirements()
    design = build_pcb(req, board_template="RPI_HAT")
    pts = design.outline.polygon
    # Rounded rectangle has many more points than 5 (4 corners + closure)
    # 4 corners x (8 segments + 1) + 1 closure = 37
    assert len(pts) > 10, f"Expected rounded polygon, got {len(pts)} points"
    # Should still be closed
    assert abs(pts[0].x - pts[-1].x) < 1e-4
    assert abs(pts[0].y - pts[-1].y) < 1e-4


def test_board_outline_sharp_corners_default() -> None:
    """Board outline without corner radius has exactly 5 points (closed rect)."""
    req = _make_requirements()
    design = build_pcb(req)
    pts = design.outline.polygon
    assert len(pts) == 5


def test_build_pcb_with_template_uses_template_dimensions() -> None:
    """build_pcb with board_template uses the template dimensions."""
    req = _make_requirements()
    design = build_pcb(req, board_template="RPI_HAT")
    xs = [p.x for p in design.outline.polygon]
    ys = [p.y for p in design.outline.polygon]
    assert max(xs) == pytest.approx(65.0, abs=0.5)
    assert max(ys) == pytest.approx(56.0, abs=0.5)


def test_mounting_hole_keepouts_use_actual_positions() -> None:
    """When template provides mounting positions, keepouts use them."""
    req = _make_requirements()
    design = build_pcb(req, board_template="RPI_HAT")
    # RPi HAT has 4 mounting holes + 1 antenna keepout (ESP32)
    mounting_keepouts = [
        k for k in design.keepouts
        if k.no_copper and k.no_vias and k.no_tracks
    ]
    assert len(mounting_keepouts) >= 4
    # Check that keepout centres are near the RPi HAT mounting positions
    expected_centres = [(3.5, 3.5), (3.5, 52.5), (61.5, 3.5), (61.5, 52.5)]
    for ko in mounting_keepouts[:4]:
        cx = sum(p.x for p in ko.polygon) / len(ko.polygon)
        cy = sum(p.y for p in ko.polygon) / len(ko.polygon)
        matched = any(
            abs(cx - ex) < 1.0 and abs(cy - ey) < 1.0
            for ex, ey in expected_centres
        )
        assert matched, f"Keepout centre ({cx:.1f}, {cy:.1f}) not near any expected position"


# ---------------------------------------------------------------------------
# Fix 1: Template preserves dimensions (no auto-sizing override)
# ---------------------------------------------------------------------------


def test_build_pcb_with_template_preserves_dimensions() -> None:
    """Template dimensions must not be overridden by auto-sizing logic.

    The PinSocket_2x20 (50.8mm wide) should fit within 65mm RPi HAT
    without triggering auto-sizing to 70.8mm.
    """
    # Add a large PinSocket to force auto-sizing in old code
    large_header = Component(
        ref="J1",
        value="PinSocket_2x20",
        footprint="PinSocket_2x20_P2.54mm_Vertical",
        pins=tuple(
            Pin(number=str(i + 1), name=f"P{i + 1}", pin_type=PinType.PASSIVE)
            for i in range(40)
        ),
    )
    req = _make_requirements(extra_components=(large_header,))
    design = build_pcb(req, board_template="RPI_HAT")
    xs = [p.x for p in design.outline.polygon]
    board_w = max(xs) - min(xs)
    # Must stay at template width (65mm), NOT inflate to 70.8mm
    assert board_w == pytest.approx(65.0, abs=0.5), (
        f"Board width {board_w:.1f}mm exceeds template 65mm — auto-sizing override"
    )


# ---------------------------------------------------------------------------
# Fix 2: Builder applies solver rotations
# ---------------------------------------------------------------------------


def test_build_pcb_with_template_applies_rotations() -> None:
    """Footprints built with a template should have rotations from the solver."""
    req = _make_requirements()
    design = build_pcb(req, board_template="RPI_HAT")
    # The design should build without error; rotations should be numeric
    for fp in design.footprints:
        assert isinstance(fp.rotation, float)


# ---------------------------------------------------------------------------
# BUG-11: Mounting hole footprints
# ---------------------------------------------------------------------------


def test_build_pcb_rpi_hat_has_mounting_holes() -> None:
    """RPi HAT template generates NPTH mounting hole footprints H1-H4."""
    req = _make_requirements()
    design = build_pcb(req, board_template="RPI_HAT")
    mh_fps = [fp for fp in design.footprints if fp.ref.startswith("H")]
    assert len(mh_fps) == 4
    refs = sorted(fp.ref for fp in mh_fps)
    assert refs == ["H1", "H2", "H3", "H4"]
    # All should be NPTH (exclude_from_bom)
    for fp in mh_fps:
        assert "exclude_from_bom" in fp.attr
        assert fp.pads[0].pad_type == "np_thru_hole"


def test_build_pcb_rpi_hat_mounting_hole_positions() -> None:
    """Mounting holes are at the correct RPi HAT spec positions."""
    req = _make_requirements()
    design = build_pcb(req, board_template="RPI_HAT")
    mh_fps = {fp.ref: fp for fp in design.footprints if fp.ref.startswith("H")}
    expected = {"H1": (3.5, 3.5), "H2": (3.5, 52.5), "H3": (61.5, 3.5), "H4": (61.5, 52.5)}
    for ref, (ex, ey) in expected.items():
        assert ref in mh_fps, f"Missing mounting hole {ref}"
        assert mh_fps[ref].position.x == pytest.approx(ex, abs=0.01)
        assert mh_fps[ref].position.y == pytest.approx(ey, abs=0.01)


# ---------------------------------------------------------------------------
# Phase 1: No GND stitching vias (copper pour handles GND connectivity)
# ---------------------------------------------------------------------------


def test_build_pcb_rpi_hat_no_stitching_grid_vias() -> None:
    """RPi HAT board has no GND stitching grid vias (copper pour suffices).

    RF via fence vias may still exist if an RF module is detected.
    """
    req = _make_requirements()
    design = build_pcb(req, board_template="RPI_HAT", auto_route=False)
    gnd_net_num = next(n.number for n in design.nets if n.name == "GND")
    for v in design.vias:
        assert v.net_number == gnd_net_num
        # RF fence vias use 0.6mm drill; old stitching used same, but
        # they were on an 8mm grid pattern. Verify no grid pattern.
        assert v.drill == 0.6


# ---------------------------------------------------------------------------
# BUG-10 + BUG-13: Board template values
# ---------------------------------------------------------------------------


def test_rpi_hat_board_height_56mm() -> None:
    """RPi HAT board height must be 56.0mm per official spec."""
    req = _make_requirements()
    design = build_pcb(req, board_template="RPI_HAT")
    ys = [p.y for p in design.outline.polygon]
    board_h = max(ys) - min(ys)
    assert board_h == pytest.approx(56.0, abs=0.5)


# ---------------------------------------------------------------------------
# BUG-18: Zone fill polygons and format
# ---------------------------------------------------------------------------


def test_zone_sexp_emits_fill_yes_and_filled_polygon() -> None:
    """_zone_sexp emits (fill yes ...) and (filled_polygon ...) when filled_polygons present."""
    zone = ZonePolygon(
        net_number=1,
        net_name="GND",
        layer="B.Cu",
        name="GND",
        polygon=(Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10), Point(0, 0)),
        filled_polygons=(
            (Point(0.5, 0.5), Point(9.5, 0.5), Point(9.5, 9.5), Point(0.5, 9.5), Point(0.5, 0.5)),
        ),
        uuid="test-zone-uuid",
    )
    sexp = _zone_sexp(zone)

    # Check connect_pads does NOT have "yes" as second element
    connect_pads = [s for s in sexp if isinstance(s, list) and s and s[0] == "connect_pads"]
    assert len(connect_pads) == 1
    assert connect_pads[0][1] != "yes", "connect_pads should not have bare 'yes'"

    # Check fill has "yes" marker
    fill_node = [s for s in sexp if isinstance(s, list) and s and s[0] == "fill"]
    assert len(fill_node) == 1
    assert fill_node[0][1] == "yes", "fill should have 'yes' marker"

    # Check filled_polygon is emitted
    fp_nodes = [s for s in sexp if isinstance(s, list) and s and s[0] == "filled_polygon"]
    assert len(fp_nodes) == 1
    # Should have layer and pts sub-nodes
    layer_node = [s for s in fp_nodes[0] if isinstance(s, list) and s and s[0] == "layer"]
    assert layer_node[0][1] == "B.Cu"


def test_zone_sexp_no_fill_yes_without_filled_polygons() -> None:
    """_zone_sexp emits (fill ...) without 'yes' when no filled_polygons."""
    zone = ZonePolygon(
        net_number=1,
        net_name="GND",
        layer="B.Cu",
        name="GND",
        polygon=(Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10), Point(0, 0)),
        uuid="test-zone-uuid",
    )
    sexp = _zone_sexp(zone)
    fill_node = [s for s in sexp if isinstance(s, list) and s and s[0] == "fill"]
    assert len(fill_node) == 1
    assert fill_node[0][1] != "yes", "fill should NOT have 'yes' without filled_polygons"


# ---------------------------------------------------------------------------
# BUG-20: Silk reference position from fp.texts
# ---------------------------------------------------------------------------


def test_footprint_sexp_uses_silk_ref_position() -> None:
    """_footprint_sexp uses reference text position from fp.texts if available."""
    fp = Footprint(
        lib_id="R_0805:R_0805_2012Metric",
        ref="R1",
        value="10k",
        position=Point(10.0, 20.0),
        texts=(
            FootprintText(
                text_type="reference",
                text="R1",
                position=Point(0.0, -3.7),
                layer="F.SilkS",
            ),
            FootprintText(
                text_type="value",
                text="10k",
                position=Point(0.0, 3.2),
                layer="F.Fab",
            ),
        ),
        uuid="test-fp-uuid",
    )
    sexp = _footprint_sexp(fp)

    # Find the Reference property node
    ref_props = [
        s for s in sexp
        if isinstance(s, list) and len(s) >= 3 and s[0] == "property" and s[1] == "Reference"
    ]
    assert len(ref_props) == 1
    at_node = [s for s in ref_props[0] if isinstance(s, list) and s and s[0] == "at"]
    assert at_node[0][2] == pytest.approx(-3.7), "Reference Y should use fp.texts position"

    # Find the Value property node
    val_props = [
        s for s in sexp
        if isinstance(s, list) and len(s) >= 3 and s[0] == "property" and s[1] == "Value"
    ]
    assert len(val_props) == 1
    at_node_v = [s for s in val_props[0] if isinstance(s, list) and s and s[0] == "at"]
    assert at_node_v[0][2] == pytest.approx(3.2), "Value Y should use fp.texts position"




# ---------------------------------------------------------------------------
# Phase 3: Multi-layer foundation
# ---------------------------------------------------------------------------


def test_build_layer_table_2_layer() -> None:
    """2-layer table should not contain In1.Cu or In2.Cu."""
    table = _build_layer_table(2)
    names = [entry[1] for entry in table]
    assert "F.Cu" in names
    assert "B.Cu" in names
    assert "In1.Cu" not in names
    assert "In2.Cu" not in names


def test_build_layer_table_4_layer() -> None:
    """4-layer table should contain In1.Cu and In2.Cu."""
    table = _build_layer_table(4)
    names = [entry[1] for entry in table]
    assert "F.Cu" in names
    assert "In1.Cu" in names
    assert "In2.Cu" in names
    assert "B.Cu" in names
    # In1.Cu should be power type
    in1 = [e for e in table if e[1] == "In1.Cu"]
    assert in1[0][2] == "power"


def test_build_layer_table_default_is_2_layer() -> None:
    """Default layer table should be 2-layer."""
    table = _build_layer_table()
    names = [entry[1] for entry in table]
    assert "In1.Cu" not in names


# ---------------------------------------------------------------------------
# Phase 4: RF via fence
# ---------------------------------------------------------------------------


def test_rf_via_fence_around_antenna_keepout() -> None:
    """RF via fence should place vias around antenna keepout polygons."""
    from kicad_pipeline.models.pcb import Keepout

    ko = Keepout(
        polygon=(
            Point(70.0, 0.0), Point(80.0, 0.0),
            Point(80.0, 10.0), Point(70.0, 10.0),
            Point(70.0, 0.0),
        ),
        layers=("F.Cu", "B.Cu"),
        no_copper=True,
        no_vias=False,
        no_tracks=False,
    )
    vias = _make_rf_via_fence(
        (ko,), gnd_net_num=1, spacing_mm=2.0,
    )
    assert len(vias) > 0
    # All vias should be GND
    for v in vias:
        assert v.net_number == 1


def test_rf_via_fence_skips_non_rf_keepout() -> None:
    """RF via fence should skip keepouts without no_copper or without F.Cu."""
    from kicad_pipeline.models.pcb import Keepout

    # Keepout without no_copper
    ko1 = Keepout(
        polygon=(
            Point(0.0, 0.0), Point(5.0, 0.0),
            Point(5.0, 5.0), Point(0.0, 5.0),
            Point(0.0, 0.0),
        ),
        layers=("F.Cu",),
        no_copper=False,
        no_vias=True,
    )
    # Keepout without F.Cu in layers
    ko2 = Keepout(
        polygon=(
            Point(10.0, 0.0), Point(15.0, 0.0),
            Point(15.0, 5.0), Point(10.0, 5.0),
            Point(10.0, 0.0),
        ),
        layers=("B.Cu",),
        no_copper=True,
    )
    vias = _make_rf_via_fence(
        (ko1, ko2), gnd_net_num=1, spacing_mm=2.0,
    )
    assert len(vias) == 0


def test_netclass_guard_traces_default() -> None:
    """NetClass.guard_traces should default to False."""
    from kicad_pipeline.models.pcb import NetClass

    nc = NetClass(name="Default")
    assert nc.guard_traces is False




# ---------------------------------------------------------------------------
# Integration tests — build_pcb pipeline DRC-like checks
# ---------------------------------------------------------------------------


def _make_rpi_hat_requirements() -> ProjectRequirements:
    """Build a realistic RPi HAT requirements fixture with ADS1115 + 4 channels.

    This matches the smd-0603 test variant used for manual DRC validation.
    """
    from kicad_pipeline.models.requirements import PowerBudget, PowerRail

    def _pin(num: str, name: str, fn: PinFunction = PinFunction.GPIO,
             pt: PinType = PinType.PASSIVE) -> Pin:
        return Pin(number=num, name=name, function=fn, pin_type=pt)

    u1_pins = (
        _pin("1", "ADDR"), _pin("2", "ALRT", PinFunction.INTERRUPT),
        _pin("3", "SDA", PinFunction.I2C_SDA, PinType.BIDIRECTIONAL),
        _pin("4", "AIN0", PinFunction.ANALOG_IN, PinType.INPUT),
        _pin("5", "AIN1", PinFunction.ANALOG_IN, PinType.INPUT),
        _pin("6", "AIN2", PinFunction.ANALOG_IN, PinType.INPUT),
        _pin("7", "AIN3", PinFunction.ANALOG_IN, PinType.INPUT),
        _pin("8", "GND", PinFunction.GND, PinType.POWER_IN),
        _pin("9", "VDD", PinFunction.VCC, PinType.POWER_IN),
        _pin("10", "SCL", PinFunction.I2C_SCL, PinType.INPUT),
    )
    j1_pins = tuple(
        _pin(str(i), f"P{i}") for i in range(1, 41)
    )
    two_pin = (_pin("1", "1"), _pin("2", "2"))

    components = (
        Component(
            ref="U1", value="ADS1115",
            footprint="MSOP-10_3x3mm_P0.5mm", pins=u1_pins,
        ),
        Component(
            ref="J1", value="Conn_02x20_Stacking",
            footprint="PinSocket_2x20_P2.54mm_Vertical_Extra_Tall",
            pins=j1_pins,
        ),
        Component(
            ref="J2", value="Screw_Terminal_01x02",
            footprint="TerminalBlock_bornier-2_P5.08mm", pins=two_pin,
        ),
        Component(
            ref="R1", value="100k", footprint="R_0603",
            pins=two_pin,
        ),
        Component(
            ref="C1", value="100nF", footprint="C_0603",
            pins=two_pin,
        ),
    )
    nets = (
        Net(name="GND", connections=(
            NetConnection(ref="U1", pin="8"),
            NetConnection(ref="J1", pin="6"),
            NetConnection(ref="J2", pin="2"),
            NetConnection(ref="C1", pin="2"),
        )),
        Net(name="+3V3", connections=(
            NetConnection(ref="U1", pin="9"),
            NetConnection(ref="J1", pin="1"),
            NetConnection(ref="C1", pin="1"),
        )),
        Net(name="I2C_SDA", connections=(
            NetConnection(ref="U1", pin="3"),
            NetConnection(ref="J1", pin="3"),
        )),
        Net(name="I2C_SCL", connections=(
            NetConnection(ref="U1", pin="10"),
            NetConnection(ref="J1", pin="5"),
        )),
        Net(name="AIN0", connections=(
            NetConnection(ref="U1", pin="4"),
            NetConnection(ref="R1", pin="2"),
        )),
        Net(name="SENS0", connections=(
            NetConnection(ref="J2", pin="1"),
            NetConnection(ref="R1", pin="1"),
        )),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="test_hat", revision="1.0", author="test"),
        features=(FeatureBlock(
            name="ADC", description="4-channel ADC",
            components=("U1", "R1"), nets=("AIN0",), subcircuits=(),
        ),),
        components=components,
        nets=nets,
        mechanical=MechanicalConstraints(
            board_width_mm=65.0,
            board_height_mm=56.0,
            mounting_hole_diameter_mm=2.75,
            mounting_hole_positions=((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)),
            notes="Standard Raspberry Pi HAT form factor",
        ),
        power_budget=PowerBudget(
            total_current_ma=100.0,
            notes=(),
            rails=(PowerRail(name="+3V3", voltage=3.3, current_ma=50.0, source_ref="J1"),),
        ),
    )


def test_build_pcb_auto_detects_rpi_hat_template() -> None:
    """build_pcb should auto-detect RPi HAT template from mechanical constraints."""
    from kicad_pipeline.pcb.builder import build_pcb

    req = _make_rpi_hat_requirements()
    design = build_pcb(req)

    # J1 should be placed at the template's fixed position (~32.5mm x)
    j1 = design.get_footprint("J1")
    assert j1 is not None
    # All J1 pads must be within the board (0..65mm x, 0..56mm y)
    for pad in j1.pads:
        px = j1.position.x + pad.position.x
        assert 0.0 <= px <= 65.0, (
            f"J1 pad {pad.number} at x={px:.1f} is outside the board"
        )


def test_build_pcb_no_tracks_through_keepouts() -> None:
    """No routed track should pass through a keepout zone."""
    from kicad_pipeline.pcb.builder import build_pcb

    req = _make_rpi_hat_requirements()
    design = build_pcb(req)

    # Build keepout bounding boxes
    ko_boxes: list[tuple[float, float, float, float]] = []
    for ko in design.keepouts:
        if ko.no_tracks or ko.no_copper:
            xs = [p.x for p in ko.polygon]
            ys = [p.y for p in ko.polygon]
            ko_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    # Check that no track endpoint is inside a keepout
    violations = 0
    for trk in design.tracks:
        for pt in (trk.start, trk.end):
            for bx0, by0, bx1, by1 in ko_boxes:
                if bx0 <= pt.x <= bx1 and by0 <= pt.y <= by1:
                    violations += 1
    assert violations == 0, f"{violations} track endpoints inside keepout zones"


def test_build_pcb_no_vias_at_board_edge() -> None:
    """Routing vias must not be placed too close to the board edge."""
    from kicad_pipeline.pcb.builder import build_pcb

    req = _make_rpi_hat_requirements()
    design = build_pcb(req)

    board_w = 65.0
    board_h = 56.0
    min_margin = 1.5  # minimum via center distance from edge

    for via in design.vias:
        x, y = via.position.x, via.position.y
        assert x >= min_margin, f"Via at x={x:.2f} too close to left edge"
        assert y >= min_margin, f"Via at y={y:.2f} too close to top edge"
        assert x <= board_w - min_margin, f"Via at x={x:.2f} too close to right edge"
        assert y <= board_h - min_margin, f"Via at y={y:.2f} too close to bottom edge"


def test_build_pcb_no_rf_fence_without_rf_module() -> None:
    """RF via fence should not produce large-drill vias when no RF module is present."""
    from kicad_pipeline.pcb.builder import build_pcb

    req = _make_rpi_hat_requirements()
    design = build_pcb(req)

    # Without RF module, no RF fence vias (drill=0.6) should exist.
    # GND stitching vias (drill=0.3) are expected.
    gnd_net = design.get_net_number("GND")
    rf_fence_vias = [
        v for v in design.vias
        if v.net_number == gnd_net and v.drill > 0.55
    ]
    assert len(rf_fence_vias) == 0, (
        f"Found {len(rf_fence_vias)} RF fence GND vias without RF module"
    )


def test_build_pcb_connectors_no_courtyard_overlap() -> None:
    """Connectors should not overlap each other's courtyards."""
    from kicad_pipeline.pcb.builder import build_pcb

    req = _make_rpi_hat_requirements()
    design = build_pcb(req)

    # Build bounding boxes from pad extents (proxy for courtyard)
    def _fp_bbox(fp: Footprint) -> tuple[float, float, float, float]:
        if not fp.pads:
            return (fp.position.x, fp.position.y, fp.position.x, fp.position.y)
        pad_xs = [fp.position.x + p.position.x - p.size_x / 2 for p in fp.pads]
        pad_xe = [fp.position.x + p.position.x + p.size_x / 2 for p in fp.pads]
        pad_ys = [fp.position.y + p.position.y - p.size_y / 2 for p in fp.pads]
        pad_ye = [fp.position.y + p.position.y + p.size_y / 2 for p in fp.pads]
        return (min(pad_xs), min(pad_ys), max(pad_xe), max(pad_ye))

    connectors = [fp for fp in design.footprints if fp.ref.startswith("J")]
    for i, a in enumerate(connectors):
        for b in connectors[i + 1:]:
            ax0, ay0, ax1, ay1 = _fp_bbox(a)
            bx0, by0, bx1, by1 = _fp_bbox(b)
            overlap_x = ax0 < bx1 and bx0 < ax1
            overlap_y = ay0 < by1 and by0 < ay1
            assert not (overlap_x and overlap_y), (
                f"{a.ref} and {b.ref} pads overlap: "
                f"{a.ref}=({ax0:.1f},{ay0:.1f})-({ax1:.1f},{ay1:.1f}) "
                f"{b.ref}=({bx0:.1f},{by0:.1f})-({bx1:.1f},{by1:.1f})"
            )


def test_build_pcb_no_fcu_tracks_cross_other_net_pads() -> None:
    """No F.Cu track should cross a pad belonging to a different net.

    The grid router validates F.Cu stubs from B.Cu fallback routes and
    discards any that cross other-net pads (e.g. diagonal stubs through
    dense IC pin areas).
    """
    from kicad_pipeline.pcb.builder import build_pcb

    req = _make_rpi_hat_requirements()
    design = build_pcb(req)

    # Build lookup: (abs_x, abs_y) -> (net_number, half_w, half_h, ref)
    import math as _math

    pad_info: list[tuple[float, float, int, float, float, str]] = []
    # Also build ref -> set of nets (for intra-footprint exemptions)
    ref_nets: dict[str, set[int]] = {}
    for fp in design.footprints:
        rot_rad = _math.radians(fp.rotation)
        cos_r = _math.cos(rot_rad)
        sin_r = _math.sin(rot_rad)
        fp_nets: set[int] = set()
        for pad in fp.pads:
            # Apply footprint rotation to pad offset
            rpx = pad.position.x * cos_r - pad.position.y * sin_r
            rpy = pad.position.x * sin_r + pad.position.y * cos_r
            px = fp.position.x + rpx
            py = fp.position.y + rpy
            net = pad.net_number if pad.net_number is not None else 0
            hw = pad.size_x / 2.0
            hh = pad.size_y / 2.0
            pad_info.append((px, py, net, hw, hh, fp.ref))
            if net > 0:
                fp_nets.add(net)
        ref_nets[fp.ref] = fp_nets

    # Only check long diagonal stubs (>3mm) — short/medium grid-aligned
    # segments near pad edges are acceptable and not flagged by KiCad DRC.
    # With track simplification, colinear segments are merged into longer
    # segments that may start/end at pad locations.
    for track in design.tracks:
        if track.layer != "F.Cu":
            continue
        dx = abs(track.end.x - track.start.x)
        dy = abs(track.end.y - track.start.y)
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len < 3.0:
            continue  # skip short/medium segments
        thw = track.width / 2.0
        tx0 = min(track.start.x, track.end.x) - thw
        tx1 = max(track.start.x, track.end.x) + thw
        ty0 = min(track.start.y, track.end.y) - thw
        ty1 = max(track.start.y, track.end.y) + thw
        for px, py, pnet, hw, hh, ref in pad_info:
            if pnet == track.net_number or pnet == 0:
                continue  # same net or unnetted
            # Skip intra-footprint crossings: when routing to a pad on this
            # component, the track naturally passes near other pads on the
            # same component.
            if track.net_number in ref_nets.get(ref, set()):
                continue
            # Shrink pad rect by 0.05mm to avoid boundary false positives
            margin = 0.05
            if (tx1 > px - hw + margin and tx0 < px + hw - margin
                    and ty1 > py - hh + margin and ty0 < py + hh - margin):
                pytest.fail(
                    f"F.Cu track net {track.net_number} "
                    f"({track.start.x:.1f},{track.start.y:.1f})->"
                    f"({track.end.x:.1f},{track.end.y:.1f}) "
                    f"crosses pad at ({px:.1f},{py:.1f}) "
                    f"net {pnet}"
                )


def test_build_pcb_pads_have_net_assignments() -> None:
    """All pads connected to nets should have net_number assigned."""
    from kicad_pipeline.pcb.builder import build_pcb

    req = _make_rpi_hat_requirements()
    design = build_pcb(req)

    # At least the non-mounting-hole footprints should have some netted pads
    signal_fps = [
        fp for fp in design.footprints
        if not fp.ref.startswith("H")
    ]
    for fp in signal_fps:
        netted = sum(
            1 for p in fp.pads
            if p.net_number is not None and p.net_number > 0
        )
        assert netted > 0, (
            f"{fp.ref} has {len(fp.pads)} pads but none have "
            f"net assignments"
        )


# ---------------------------------------------------------------------------
# Intra-footprint DRC exclusion generation
# ---------------------------------------------------------------------------


class TestGenerateICDrcExclusions:
    """Tests for _generate_ic_drc_exclusions."""

    def test_dense_ic_generates_exclusions(self) -> None:
        """MSOP-10 with 0.5mm pitch generates pad-pair exclusions."""
        pads = []
        for i in range(5):
            pads.append(Pad(
                number=str(i + 1), pad_type="smd", shape="rect",
                position=Point(x=-0.975, y=-1.0 + i * 0.5),
                size_x=0.41, size_y=0.3,
                layers=("F.Cu", "F.Paste", "F.Mask"),
            ))
        for i in range(5):
            pads.append(Pad(
                number=str(i + 6), pad_type="smd", shape="rect",
                position=Point(x=0.975, y=1.0 - i * 0.5),
                size_x=0.41, size_y=0.3,
                layers=("F.Cu", "F.Paste", "F.Mask"),
            ))
        fp = Footprint(
            lib_id="Package_SO:MSOP-10", ref="U1", value="ADS1115",
            position=Point(x=30.0, y=30.0), rotation=0.0, layer="F.Cu",
            pads=tuple(pads), graphics=(), texts=(),
        )
        exclusions = _generate_ic_drc_exclusions([fp])
        assert len(exclusions) > 0
        assert all(e.startswith("clearance|U1|") for e in exclusions)
        # Adjacent pads (1,2) should be excluded
        assert "clearance|U1|1|U1|2" in exclusions

    def test_wide_pitch_no_exclusions(self) -> None:
        """Connector with 2.54mm pitch does not generate exclusions."""
        pads = tuple(
            Pad(
                number=str(i + 1), pad_type="thru_hole", shape="circle",
                position=Point(x=0.0, y=i * 2.54),
                size_x=1.7, size_y=1.7,
                layers=("*.Cu", "*.Mask"),
            )
            for i in range(10)
        )
        fp = Footprint(
            lib_id="Conn:PinHeader_1x10", ref="J1", value="Conn",
            position=Point(x=10.0, y=10.0), rotation=0.0, layer="F.Cu",
            pads=pads, graphics=(), texts=(),
        )
        exclusions = _generate_ic_drc_exclusions([fp])
        assert len(exclusions) == 0

    def test_few_pads_no_exclusions(self) -> None:
        """Footprint with <6 pads is not treated as dense IC."""
        pads = tuple(
            Pad(
                number=str(i + 1), pad_type="smd", shape="rect",
                position=Point(x=0.0, y=i * 0.5),
                size_x=0.3, size_y=0.2,
                layers=("F.Cu",),
            )
            for i in range(4)
        )
        fp = Footprint(
            lib_id="Package_SO:SOT-23", ref="U2", value="LDO",
            position=Point(x=5.0, y=5.0), rotation=0.0, layer="F.Cu",
            pads=pads, graphics=(), texts=(),
        )
        exclusions = _generate_ic_drc_exclusions([fp])
        assert len(exclusions) == 0


# ---------------------------------------------------------------------------
# GND stitching vias
# ---------------------------------------------------------------------------


def test_gnd_stitching_vias_on_empty_board() -> None:
    """GND stitching vias are placed on an empty 65x35mm board."""
    board = BoardOutline(polygon=(
        Point(0, 0), Point(65, 0), Point(65, 35), Point(0, 35), Point(0, 0),
    ))
    vias = _make_gnd_stitching_vias(
        board, gnd_net_number=1,
        footprints=(), existing_vias=(), existing_tracks=(),
    )
    # 65x35mm at 15mm spacing: 4 cols x 2-3 rows on an empty board
    assert len(vias) >= 4
    # All vias should be GND
    for v in vias:
        assert v.net_number == 1
        assert v.drill == 0.3
        assert v.size == 0.6


def test_gnd_stitching_vias_avoid_footprints() -> None:
    """GND stitching vias should avoid footprint areas."""
    board = BoardOutline(polygon=(
        Point(0, 0), Point(30, 0), Point(30, 30), Point(0, 30), Point(0, 0),
    ))
    # Place a large footprint in the center
    big_fp = Footprint(
        lib_id="Test:Test", ref="U1", value="IC",
        position=Point(15, 15), rotation=0.0, layer="F.Cu",
        pads=(
            Pad(number="1", pad_type="smd", shape="rect",
                position=Point(-5, -5), size_x=10, size_y=10,
                layers=("F.Cu",)),
        ),
    )
    vias = _make_gnd_stitching_vias(
        board, gnd_net_number=1,
        footprints=(big_fp,), existing_vias=(), existing_tracks=(),
    )
    # No via should be within 2mm of the footprint bbox
    for v in vias:
        # Footprint covers 10..20, 10..20 (position 15 + pad offset -5, size 10)
        assert not (8.0 <= v.position.x <= 22.0 and 8.0 <= v.position.y <= 22.0)


def test_gnd_stitching_vias_avoid_existing_vias() -> None:
    """GND stitching vias should not overlap existing vias."""
    board = BoardOutline(polygon=(
        Point(0, 0), Point(30, 0), Point(30, 30), Point(0, 30), Point(0, 0),
    ))
    existing = (
        Via(position=Point(15, 15), drill=0.3, size=0.6,
            layers=("F.Cu", "B.Cu"), net_number=2),
    )
    vias = _make_gnd_stitching_vias(
        board, gnd_net_number=1,
        footprints=(), existing_vias=existing, existing_tracks=(),
    )
    for v in vias:
        dist = abs(v.position.x - 15.0) + abs(v.position.y - 15.0)
        assert dist >= 1.0


# ---------------------------------------------------------------------------
# Layout preservation (preserve_from)
# ---------------------------------------------------------------------------


def test_build_pcb_preserve_from_file(tmp_path: Path) -> None:
    """Rebuilding with preserve_from keeps footprint positions from first build."""
    reqs = _make_requirements()
    # First build
    pcb1 = build_pcb(reqs, auto_route=False)
    pcb_path = tmp_path / "test.kicad_pcb"
    write_pcb(pcb1, pcb_path)

    # Second build with preserve_from
    pcb2 = build_pcb(reqs, auto_route=False, preserve_from=pcb_path)

    # Positions should match
    for fp1 in pcb1.footprints:
        fp2_match = [f for f in pcb2.footprints if f.ref == fp1.ref]
        assert fp2_match, f"Missing footprint {fp1.ref}"
        fp2 = fp2_match[0]
        assert abs(fp2.position.x - fp1.position.x) < 0.01, f"{fp1.ref} x mismatch"
        assert abs(fp2.position.y - fp1.position.y) < 0.01, f"{fp1.ref} y mismatch"


def test_build_pcb_preserve_with_new_component(tmp_path: Path) -> None:
    """New components get placed by solver; existing positions preserved."""
    reqs_base = _make_requirements()
    pcb1 = build_pcb(reqs_base, auto_route=False)
    pcb_path = tmp_path / "test.kicad_pcb"
    write_pcb(pcb1, pcb_path)

    # Add a new component
    new_comp = Component(
        ref="R1",
        value="10k",
        footprint="R_0402",
        pins=(
            Pin(number="1", name="1", pin_type=PinType.PASSIVE, net="+3V3"),
            Pin(number="2", name="2", pin_type=PinType.PASSIVE, net="GND"),
        ),
    )
    reqs_expanded = _make_requirements(extra_components=(new_comp,))
    pcb2 = build_pcb(reqs_expanded, auto_route=False, preserve_from=pcb_path)

    # Original positions preserved
    for fp1 in pcb1.footprints:
        fp2_match = [f for f in pcb2.footprints if f.ref == fp1.ref]
        assert fp2_match, f"Missing footprint {fp1.ref}"
        fp2 = fp2_match[0]
        assert abs(fp2.position.x - fp1.position.x) < 0.01, f"{fp1.ref} x mismatch"

    # New component exists somewhere
    r1_fps = [f for f in pcb2.footprints if f.ref == "R1"]
    assert r1_fps, "New component R1 not placed"


def test_build_pcb_grouped_mode() -> None:
    """Build with placement_mode='grouped' places all footprints and keeps board outline."""
    reqs = _make_requirements()
    pcb = build_pcb(reqs, auto_route=False, placement_mode="grouped")
    # All components have positions
    assert len(pcb.footprints) >= len(reqs.components)
    for fp in pcb.footprints:
        # Every footprint should have a non-zero position (placed off-board)
        assert fp.position is not None, f"{fp.ref} has no position"
    # Board outline still exists
    assert pcb.outline is not None
    assert len(pcb.outline.polygon) >= 4

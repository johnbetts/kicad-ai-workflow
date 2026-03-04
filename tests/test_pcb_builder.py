"""Tests for the PCB builder, pcb_to_sexp, and write_pcb functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from kicad_pipeline.models.pcb import PCBDesign
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
from kicad_pipeline.pcb.builder import build_pcb, pcb_to_sexp, write_pcb
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
    """PCBDesign has GND zones on both F.Cu and B.Cu."""
    req = _make_requirements()
    design = build_pcb(req)
    layers = {z.layer for z in design.zones}
    assert "F.Cu" in layers
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


def test_build_pcb_no_tracks_yet() -> None:
    """Fresh PCBDesign has no tracks (autorouter adds them)."""
    req = _make_requirements()
    design = build_pcb(req)
    assert len(design.tracks) == 0


def test_build_pcb_no_vias_yet() -> None:
    """Fresh PCBDesign has no vias (autorouter adds them)."""
    req = _make_requirements()
    design = build_pcb(req)
    assert len(design.vias) == 0


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
    """Default board is 80 x 40 mm when no mechanical constraints are given."""
    req = _make_requirements(with_mechanical=False)
    design = build_pcb(req)
    xs = [p.x for p in design.outline.polygon]
    ys = [p.y for p in design.outline.polygon]
    assert max(xs) == pytest.approx(80.0)
    assert max(ys) == pytest.approx(40.0)


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
        # Find connect_pads node
        for sub in zone_node:
            if isinstance(sub, list) and sub and sub[0] == "connect_pads":
                clearance_node = sub[1]
                assert isinstance(clearance_node, list)
                assert clearance_node[0] == "clearance"
                assert clearance_node[1] >= 0.2


def test_write_pcb_accepts_string_path(tmp_path: Path) -> None:
    """write_pcb accepts a plain string path as well as a Path object."""
    req = _make_requirements()
    design = build_pcb(req)
    dest = str(tmp_path / "string_path.kicad_pcb")
    write_pcb(design, dest)
    assert Path(dest).exists()

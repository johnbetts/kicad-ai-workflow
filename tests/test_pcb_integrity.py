"""Tests for PCB integrity validation — especially per-component 3D alignment (BUG-PIPE-004)."""
from __future__ import annotations

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    Footprint3DModel,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.validation.pcb_integrity import (
    _check_3d_alignment,
    _check_board_utilization,
    _check_connector_edge_placement,
    _check_duplicate_pads,
    _check_mounting_hole_clearance,
    _check_orphan_passives,
    _check_power_chain_gap,
    _classify_component,
    _pad_centroid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fp(
    ref: str = "R1",
    lib_id: str = "Resistor_SMD:R_0805_2012Metric",
    pads: tuple[Pad, ...] | None = None,
    models: tuple[Footprint3DModel, ...] = (),
    attr: str = "smd",
) -> Footprint:
    if pads is None:
        pads = (
            Pad(number="1", pad_type="smd", shape="rect",
                position=Point(-1.0, 0.0), size_x=1.0, size_y=1.2, layers=("F.Cu",)),
            Pad(number="2", pad_type="smd", shape="rect",
                position=Point(1.0, 0.0), size_x=1.0, size_y=1.2, layers=("F.Cu",)),
        )
    return Footprint(
        lib_id=lib_id, ref=ref, value="10k",
        position=Point(50.0, 50.0), pads=pads, models=models, attr=attr,
    )


def _make_model(
    path: str = "${KICAD10_3DMODEL_DIR}/Resistor_SMD.3dshapes/R_0805_2012Metric.step",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Footprint3DModel:
    return Footprint3DModel(path=path, offset=offset)


def _make_pcb(footprints: list[Footprint]) -> PCBDesign:
    outline = BoardOutline(polygon=(
        Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100),
    ))
    return PCBDesign(
        outline=outline, design_rules=DesignRules(),
        nets=(), footprints=tuple(footprints),
        tracks=(), vias=(), zones=(), keepouts=(),
    )


# ---------------------------------------------------------------------------
# _classify_component
# ---------------------------------------------------------------------------

def test_classify_passive() -> None:
    assert _classify_component(_make_fp(ref="R1")) == "passive"
    assert _classify_component(_make_fp(ref="C3")) == "passive"
    assert _classify_component(_make_fp(ref="L2")) == "passive"


def test_classify_module() -> None:
    fp = _make_fp(ref="U1", lib_id="RF_Module:ESP32-S3-WROOM-1")
    assert _classify_component(fp) == "module"


def test_classify_connector() -> None:
    fp = _make_fp(ref="J1", lib_id="Connector_PinHeader:PinHeader_1x04")
    assert _classify_component(fp) == "connector"


def test_classify_ic() -> None:
    fp = _make_fp(ref="U2", lib_id="Package_SO:SOIC-8")
    assert _classify_component(fp) == "ic"


# ---------------------------------------------------------------------------
# _pad_centroid
# ---------------------------------------------------------------------------

def test_pad_centroid_symmetric() -> None:
    cx, cy = _pad_centroid(_make_fp())
    assert abs(cx) < 0.01
    assert abs(cy) < 0.01


def test_pad_centroid_asymmetric() -> None:
    pads = (
        Pad(number="1", pad_type="smd", shape="rect",
            position=Point(0.0, 0.0), size_x=1.0, size_y=1.0, layers=("F.Cu",)),
        Pad(number="2", pad_type="smd", shape="rect",
            position=Point(4.0, 0.0), size_x=1.0, size_y=1.0, layers=("F.Cu",)),
        Pad(number="3", pad_type="smd", shape="rect",
            position=Point(2.0, 6.0), size_x=1.0, size_y=1.0, layers=("F.Cu",)),
    )
    cx, cy = _pad_centroid(_make_fp(pads=pads))
    assert abs(cx - 2.0) < 0.01
    assert abs(cy - 2.0) < 0.01


# ---------------------------------------------------------------------------
# _check_3d_alignment
# ---------------------------------------------------------------------------

def test_3d_alignment_small_offset_passes() -> None:
    """Normal passive with small offset should produce no issues."""
    fp = _make_fp(models=(_make_model(offset=(0.1, 0.1, 0.0)),))
    issues = _check_3d_alignment(_make_pcb([fp]))
    assert not issues


def test_3d_alignment_large_passive_offset_fails() -> None:
    """Passive with offset > 1.5mm should flag."""
    fp = _make_fp(models=(_make_model(offset=(2.0, 0.0, 0.0)),))
    issues = _check_3d_alignment(_make_pcb([fp]))
    assert any(i.category == "3d_alignment" and i.ref == "R1" for i in issues)


def test_3d_alignment_module_large_offset_ok() -> None:
    """Module with offset < 8mm should pass."""
    fp = _make_fp(
        ref="U1", lib_id="RF_Module:ESP32-S3-WROOM-1",
        models=(_make_model(offset=(-0.35, 4.39, 0.0)),),
    )
    issues = _check_3d_alignment(_make_pcb([fp]))
    assert not any(i.category == "3d_alignment" for i in issues)


def test_3d_alignment_z_offset_smd_fails() -> None:
    """SMD component with non-zero Z should flag."""
    fp = _make_fp(models=(_make_model(offset=(0.0, 0.0, 1.5)),))
    issues = _check_3d_alignment(_make_pcb([fp]))
    z_issues = [i for i in issues if "Z-offset" in i.message]
    assert len(z_issues) == 1


def test_3d_alignment_centroid_far_no_offset_warns() -> None:
    """Footprint with centroid far from origin but zero 3D offset → warning."""
    # Simulate a JLCPCB footprint where origin is at pin 1, not center
    pads = (
        Pad(number="1", pad_type="smd", shape="rect",
            position=Point(-8.75, -8.89), size_x=1.0, size_y=0.9, layers=("F.Cu",)),
        Pad(number="2", pad_type="smd", shape="rect",
            position=Point(8.75, -8.89), size_x=1.0, size_y=0.9, layers=("F.Cu",)),
        Pad(number="3", pad_type="smd", shape="rect",
            position=Point(0.0, 8.0), size_x=1.0, size_y=0.9, layers=("F.Cu",)),
    )
    fp = _make_fp(
        ref="U1", lib_id="RF_Module:ESP32-S3-WROOM-1",
        pads=pads,
        models=(_make_model(offset=(0.0, 0.0, 0.0)),),
    )
    issues = _check_3d_alignment(_make_pcb([fp]))
    centroid_issues = [i for i in issues if "centroid" in i.message]
    assert len(centroid_issues) == 1


def test_3d_alignment_no_models_skipped() -> None:
    """Components without models should be skipped (caught by check 3)."""
    fp = _make_fp(models=())
    issues = _check_3d_alignment(_make_pcb([fp]))
    assert not issues


def test_3d_alignment_mounting_holes_skipped() -> None:
    """Mounting holes (H*) should be skipped."""
    fp = _make_fp(ref="H1", models=(_make_model(offset=(10.0, 10.0, 0.0)),))
    issues = _check_3d_alignment(_make_pcb([fp]))
    assert not issues


# ---------------------------------------------------------------------------
# ESP32 parametric footprint integration test
# ---------------------------------------------------------------------------

def test_esp32_parametric_3d_alignment() -> None:
    """ESP32 parametric footprint should pass 3D alignment checks."""
    from kicad_pipeline.pcb.footprints import make_esp32_wroom

    fp = make_esp32_wroom("U1", "ESP32-S3-WROOM-1")
    issues = _check_3d_alignment(_make_pcb([fp]))
    alignment_issues = [i for i in issues if i.severity in ("critical", "major")]
    assert not alignment_issues, f"ESP32 3D alignment issues: {alignment_issues}"


def test_esp32_parametric_offset_not_hardcoded() -> None:
    """ESP32 offset should be dynamically computed, not hardcoded (0, 3.63, 0)."""
    from kicad_pipeline.pcb.footprints import make_esp32_wroom

    fp = make_esp32_wroom("U1", "ESP32-S3-WROOM-1")
    model = fp.models[0]
    # Old hardcoded offset was (0.0, 3.63, 0.0) — parametric path computes
    # from pad centroid so it should NOT be the old hardcoded value.
    assert abs(model.offset[1] - 3.63) > 0.5 or model.offset[1] == 0.0, (
        f"Y offset {model.offset[1]} looks like old hardcoded 3.63"
    )


# ---------------------------------------------------------------------------
# Check 6: Duplicate pad numbers
# ---------------------------------------------------------------------------

def test_duplicate_pads_detected() -> None:
    """Footprint with two pads numbered '8' should be flagged critical."""
    pads = (
        Pad(number="1", pad_type="smd", shape="rect",
            position=Point(-3.4, -1.9), size_x=1.0, size_y=0.5, layers=("F.Cu",)),
        Pad(number="8", pad_type="smd", shape="rect",
            position=Point(3.4, -1.9), size_x=1.0, size_y=0.5, layers=("F.Cu",)),
        Pad(number="8", pad_type="smd", shape="rect",
            position=Point(0.0, 0.0), size_x=4.0, size_y=2.5, layers=("F.Cu",)),
    )
    fp = _make_fp(ref="U1", lib_id="SOIC-8", pads=pads)
    issues = _check_duplicate_pads(_make_pcb([fp]))
    assert len(issues) == 1
    assert issues[0].severity == "critical"
    assert "pad 8 x2" in issues[0].message


def test_no_duplicate_pads_passes() -> None:
    """Normal footprint with unique pad numbers should pass."""
    fp = _make_fp()  # default 2-pad footprint
    issues = _check_duplicate_pads(_make_pcb([fp]))
    assert not issues


# ---------------------------------------------------------------------------
# Check 7: Mounting hole clearance
# ---------------------------------------------------------------------------

def _make_mounting_hole(ref: str, x: float, y: float) -> Footprint:
    """Create a mounting hole footprint."""
    return Footprint(
        lib_id="MountingHole:MountingHole_3.2mm", ref=ref, value="MH",
        position=Point(x, y),
        pads=(
            Pad(number="1", pad_type="np_thru_hole", shape="circle",
                position=Point(0.0, 0.0), size_x=3.2, size_y=3.2, layers=("F.Cu",)),
        ),
        attr="exclude_from_pos_files",
    )


def test_component_on_mounting_hole_detected() -> None:
    """Component overlapping a mounting hole should be flagged critical."""
    fp = _make_fp(ref="R1")
    fp = Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=Point(10.0, 10.0), pads=fp.pads, models=fp.models, attr=fp.attr,
    )
    hole = _make_mounting_hole("H1", 10.0, 10.0)  # same position!
    issues = _check_mounting_hole_clearance(_make_pcb([fp, hole]))
    assert len(issues) >= 1
    assert issues[0].severity == "critical"
    assert "H1" in issues[0].message


def test_component_far_from_mounting_hole_passes() -> None:
    """Component far from all mounting holes should pass."""
    fp = _make_fp(ref="R1")
    fp = Footprint(
        lib_id=fp.lib_id, ref=fp.ref, value=fp.value,
        position=Point(50.0, 50.0), pads=fp.pads, models=fp.models, attr=fp.attr,
    )
    hole = _make_mounting_hole("H1", 10.0, 10.0)
    issues = _check_mounting_hole_clearance(_make_pcb([fp, hole]))
    assert not issues


# ---------------------------------------------------------------------------
# Check 8: Connector edge placement
# ---------------------------------------------------------------------------

def test_connector_mid_board_detected() -> None:
    """THT connector in the middle of the board should be flagged."""
    pads = (
        Pad(number="1", pad_type="thru_hole", shape="circle",
            position=Point(-1.27, 0.0), size_x=1.7, size_y=1.7, layers=("*.Cu",)),
        Pad(number="2", pad_type="thru_hole", shape="circle",
            position=Point(1.27, 0.0), size_x=1.7, size_y=1.7, layers=("*.Cu",)),
    )
    fp = Footprint(
        lib_id="Connector:PinHeader_1x02", ref="J2", value="CONN",
        position=Point(50.0, 50.0), pads=pads, attr="thru_hole",
    )
    issues = _check_connector_edge_placement(_make_pcb([fp]))
    assert len(issues) == 1
    assert issues[0].severity == "major"
    assert "J2" in issues[0].message


def test_connector_at_edge_passes() -> None:
    """THT connector at board edge should pass."""
    pads = (
        Pad(number="1", pad_type="thru_hole", shape="circle",
            position=Point(-1.27, 0.0), size_x=1.7, size_y=1.7, layers=("*.Cu",)),
        Pad(number="2", pad_type="thru_hole", shape="circle",
            position=Point(1.27, 0.0), size_x=1.7, size_y=1.7, layers=("*.Cu",)),
    )
    fp = Footprint(
        lib_id="Connector:PinHeader_1x02", ref="J1", value="CONN",
        position=Point(3.0, 50.0), pads=pads, attr="thru_hole",  # 3mm from left edge
    )
    issues = _check_connector_edge_placement(_make_pcb([fp]))
    assert not issues


# ---------------------------------------------------------------------------
# Check 9: Board utilization
# ---------------------------------------------------------------------------

def test_low_utilization_detected() -> None:
    """A huge board with tiny components should be flagged."""
    # 100x100mm board with one tiny resistor = very low utilization
    fp = _make_fp(ref="R1")
    issues = _check_board_utilization(_make_pcb([fp]))
    assert len(issues) == 1
    assert "utilization" in issues[0].message.lower()


# ---------------------------------------------------------------------------
# Check 10: Orphan passives
# ---------------------------------------------------------------------------

def test_orphan_passive_detected() -> None:
    """A passive far from all ICs should be flagged."""
    ic = Footprint(
        lib_id="Package_SO:SOIC-8", ref="U1", value="TPS54331",
        position=Point(10.0, 10.0),
        pads=(Pad(number="1", pad_type="smd", shape="rect",
                  position=Point(0.0, 0.0), size_x=1.0, size_y=0.5, layers=("F.Cu",)),),
        attr="smd",
    )
    orphan_cap = Footprint(
        lib_id="Capacitor_SMD:C_0805", ref="C6", value="100nF",
        position=Point(90.0, 90.0),  # 113mm from IC!
        pads=(Pad(number="1", pad_type="smd", shape="rect",
                  position=Point(0.0, 0.0), size_x=1.0, size_y=0.5, layers=("F.Cu",)),),
        attr="smd",
    )
    issues = _check_orphan_passives(_make_pcb([ic, orphan_cap]))
    assert len(issues) == 1
    assert issues[0].ref == "C6"


def test_passive_near_ic_passes() -> None:
    """A passive within 15mm of an IC should pass."""
    ic = Footprint(
        lib_id="Package_SO:SOIC-8", ref="U1", value="TPS54331",
        position=Point(50.0, 50.0),
        pads=(Pad(number="1", pad_type="smd", shape="rect",
                  position=Point(0.0, 0.0), size_x=1.0, size_y=0.5, layers=("F.Cu",)),),
        attr="smd",
    )
    nearby_cap = Footprint(
        lib_id="Capacitor_SMD:C_0805", ref="C1", value="100nF",
        position=Point(55.0, 50.0),  # 5mm from IC
        pads=(Pad(number="1", pad_type="smd", shape="rect",
                  position=Point(0.0, 0.0), size_x=1.0, size_y=0.5, layers=("F.Cu",)),),
        attr="smd",
    )
    issues = _check_orphan_passives(_make_pcb([ic, nearby_cap]))
    assert not issues


# ---------------------------------------------------------------------------
# Check 11: Power chain gap
# ---------------------------------------------------------------------------

def test_power_chain_gap_detected() -> None:
    """Two regulator ICs 35mm apart should be flagged."""
    from kicad_pipeline.models.requirements import (
        Component,
        FeatureBlock,
        Net,
        ProjectInfo,
        ProjectRequirements,
    )

    u1 = Footprint(
        lib_id="Package_SO:SOIC-8", ref="U1", value="TPS54331",
        position=Point(10.0, 20.0),
        pads=(Pad(number="1", pad_type="smd", shape="rect",
                  position=Point(0.0, 0.0), size_x=1.0, size_y=0.5, layers=("F.Cu",)),),
        attr="smd",
    )
    u2 = Footprint(
        lib_id="Package_SO:SOT-223", ref="U2", value="AMS1117-3.3",
        position=Point(50.0, 20.0),  # 40mm from U1
        pads=(Pad(number="1", pad_type="smd", shape="rect",
                  position=Point(0.0, 0.0), size_x=1.0, size_y=0.5, layers=("F.Cu",)),),
        attr="smd",
    )
    req = ProjectRequirements(
        project=ProjectInfo(name="test", revision="1"),
        features=(FeatureBlock(name="Power", description="", components=("U1", "U2"), nets=(), subcircuits=()),),
        components=(
            Component(ref="U1", value="TPS54331", footprint="SOIC-8"),
            Component(ref="U2", value="AMS1117-3.3", footprint="SOT-223"),
        ),
        nets=(),
    )
    issues = _check_power_chain_gap(_make_pcb([u1, u2]), req)
    assert len(issues) == 1
    assert "U1/U2" in issues[0].ref

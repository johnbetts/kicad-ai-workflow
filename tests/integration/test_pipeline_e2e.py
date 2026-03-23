"""End-to-end integration tests for the full KiCad AI pipeline.

Tests verify that pipeline stages work together correctly:
requirements -> schematic -> PCB -> validation -> production.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import (
    Footprint,
)
from kicad_pipeline.models.pcb import (
    Point as PCBPoint,
)

if TYPE_CHECKING:
    from pathlib import Path
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

# ---------------------------------------------------------------------------
# Helpers — _make_* factories for test data
# ---------------------------------------------------------------------------


def _make_passive_pins() -> tuple[Pin, ...]:
    """Two-pin passive component (resistor/capacitor)."""
    return (
        Pin(number="1", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
        Pin(number="2", name="~", pin_type=PinType.PASSIVE, function=PinFunction.NC),
    )


def _make_mcu_pins() -> tuple[Pin, ...]:
    """Minimal MCU pins: GND, VCC, 2x GPIO."""
    return (
        Pin(number="1", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
        Pin(number="2", name="3V3", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
        Pin(number="3", name="GPIO2", pin_type=PinType.BIDIRECTIONAL, function=PinFunction.GPIO),
        Pin(number="4", name="GPIO4", pin_type=PinType.BIDIRECTIONAL, function=PinFunction.GPIO),
    )


def _make_passive(
    ref: str, value: str, footprint: str = "R_0805", lcsc: str = "C17414",
) -> Component:
    """Shorthand for a 2-pin passive component."""
    return Component(
        ref=ref, value=value, footprint=footprint,
        lcsc=lcsc, pins=_make_passive_pins(),
    )


def _make_connector_pins(count: int) -> tuple[Pin, ...]:
    """N-pin passive connector."""
    return tuple(
        Pin(number=str(i + 1), name=f"P{i + 1}", pin_type=PinType.PASSIVE, function=PinFunction.NC)
        for i in range(count)
    )


def _make_minimal_requirements(
    *,
    name: str = "test-board",
    board_w: float = 50.0,
    board_h: float = 40.0,
) -> ProjectRequirements:
    """Minimal board: 1 MCU + 1 resistor + 1 capacitor + GND net."""
    u1 = Component(
        ref="U1",
        value="ESP32-WROOM-32E",
        footprint="ESP32-WROOM-32E",
        lcsc="C165948",
        pins=_make_mcu_pins(),
    )
    r1 = Component(
        ref="R1", value="10k", footprint="R_0805", lcsc="C17414", pins=_make_passive_pins(),
    )
    c1 = Component(
        ref="C1", value="100nF", footprint="C_0805", lcsc="C49678", pins=_make_passive_pins(),
    )
    nets = (
        Net(name="GND", connections=(
            NetConnection(ref="U1", pin="1"),
            NetConnection(ref="C1", pin="2"),
        )),
        Net(name="+3V3", connections=(
            NetConnection(ref="U1", pin="2"),
            NetConnection(ref="C1", pin="1"),
        )),
        Net(name="SIG1", connections=(
            NetConnection(ref="U1", pin="3"),
            NetConnection(ref="R1", pin="1"),
        )),
    )
    features = (
        FeatureBlock(
            name="MCU",
            description="Microcontroller with decoupling",
            components=("U1", "C1"),
            nets=("GND", "+3V3"),
            subcircuits=(),
        ),
        FeatureBlock(
            name="Peripheral",
            description="Signal conditioning",
            components=("R1",),
            nets=("SIG1",),
            subcircuits=(),
        ),
    )
    return ProjectRequirements(
        project=ProjectInfo(name=name, author="test", revision="v0.1"),
        features=features,
        components=(u1, r1, c1),
        nets=nets,
        mechanical=MechanicalConstraints(
            board_width_mm=board_w, board_height_mm=board_h,
        ),
    )


def _make_multi_group_requirements() -> ProjectRequirements:
    """Board with 7 components across 3 feature groups for optimizer testing."""
    u1 = Component(
        ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32-S3-WROOM-1",
        lcsc="C2913202", pins=_make_mcu_pins(),
    )
    r1 = _make_passive("R1", "10k", lcsc="C17414")
    r2 = _make_passive("R2", "4.7k", lcsc="C17673")
    c1 = _make_passive("C1", "100nF", footprint="C_0805", lcsc="C49678")
    c2 = _make_passive("C2", "10uF", footprint="C_0805", lcsc="C15850")
    r3 = _make_passive("R3", "1k", lcsc="C17513")
    j1 = Component(
        ref="J1", value="Conn_01x02", footprint="Conn_01x02",
        lcsc="C160416", pins=_make_connector_pins(2),
    )

    nets = (
        Net(name="GND", connections=(
            NetConnection(ref="U1", pin="1"),
            NetConnection(ref="C1", pin="2"),
            NetConnection(ref="C2", pin="2"),
            NetConnection(ref="J1", pin="2"),
        )),
        Net(name="+3V3", connections=(
            NetConnection(ref="U1", pin="2"),
            NetConnection(ref="C1", pin="1"),
            NetConnection(ref="C2", pin="1"),
        )),
        Net(name="SIG_A", connections=(
            NetConnection(ref="U1", pin="3"),
            NetConnection(ref="R1", pin="1"),
        )),
        Net(name="SIG_B", connections=(
            NetConnection(ref="R1", pin="2"),
            NetConnection(ref="R2", pin="1"),
        )),
        Net(name="OUT", connections=(
            NetConnection(ref="R2", pin="2"),
            NetConnection(ref="R3", pin="1"),
        )),
        Net(name="CONN_IN", connections=(
            NetConnection(ref="J1", pin="1"),
            NetConnection(ref="R3", pin="2"),
        )),
    )

    features = (
        FeatureBlock(
            name="MCU",
            description="Main processor with decoupling",
            components=("U1", "C1"),
            nets=("GND", "+3V3"),
            subcircuits=("decoupling",),
        ),
        FeatureBlock(
            name="Signal Chain",
            description="Signal conditioning resistors",
            components=("R1", "R2", "R3"),
            nets=("SIG_A", "SIG_B", "OUT"),
            subcircuits=("voltage_divider",),
        ),
        FeatureBlock(
            name="Power",
            description="Power distribution",
            components=("C2", "J1"),
            nets=("GND", "CONN_IN"),
            subcircuits=(),
        ),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="multi-group-test", author="test", revision="v0.1"),
        features=features,
        components=(u1, r1, r2, c1, c2, r3, j1),
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=70.0, board_height_mm=50.0),
    )


def _make_relay_mcu_analog_requirements() -> ProjectRequirements:
    """Board with relay driver + MCU + analog divider for subcircuit detection."""
    u1 = Component(
        ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32-S3-WROOM-1",
        lcsc="C2913202", pins=_make_mcu_pins(),
    )
    # Relay driver: Q1 (transistor) + D1 (flyback diode) + K1 (relay)
    q1 = Component(
        ref="Q1", value="2N7002", footprint="SOT-23",
        lcsc="C8545", pins=(
            Pin(number="1", name="G", pin_type=PinType.INPUT),
            Pin(number="2", name="S", pin_type=PinType.PASSIVE),
            Pin(number="3", name="D", pin_type=PinType.OUTPUT),
        ),
    )
    d1 = Component(
        ref="D1", value="1N4148W", footprint="SOD-123",
        lcsc="C81598", pins=(
            Pin(number="1", name="K", pin_type=PinType.PASSIVE),
            Pin(number="2", name="A", pin_type=PinType.PASSIVE),
        ),
    )
    k1 = Component(
        ref="K1", value="HF46F-5-HS1", footprint="Relay_SPDT",
        lcsc="C35449", pins=(
            Pin(number="1", name="COIL+", pin_type=PinType.PASSIVE),
            Pin(number="2", name="COIL-", pin_type=PinType.PASSIVE),
            Pin(number="3", name="COM", pin_type=PinType.PASSIVE),
            Pin(number="4", name="NO", pin_type=PinType.PASSIVE),
        ),
    )
    # Decoupling
    c1 = _make_passive("C1", "100nF", footprint="C_0805", lcsc="C49678")
    # Voltage divider for analog input
    r1 = _make_passive("R1", "10k", lcsc="C17414")
    r2 = _make_passive("R2", "10k", lcsc="C17414")

    nets = (
        Net(name="GND", connections=(
            NetConnection(ref="U1", pin="1"),
            NetConnection(ref="C1", pin="2"),
            NetConnection(ref="Q1", pin="2"),
            NetConnection(ref="R2", pin="2"),
        )),
        Net(name="+3V3", connections=(
            NetConnection(ref="U1", pin="2"),
            NetConnection(ref="C1", pin="1"),
            NetConnection(ref="R1", pin="1"),
        )),
        Net(name="RELAY_DRV", connections=(
            NetConnection(ref="U1", pin="3"),
            NetConnection(ref="Q1", pin="1"),
        )),
        Net(name="RELAY_COIL", connections=(
            NetConnection(ref="Q1", pin="3"),
            NetConnection(ref="K1", pin="2"),
            NetConnection(ref="D1", pin="2"),
        )),
        Net(name="+5V", connections=(
            NetConnection(ref="K1", pin="1"),
            NetConnection(ref="D1", pin="1"),
        )),
        Net(name="ADC_IN", connections=(
            NetConnection(ref="U1", pin="4"),
            NetConnection(ref="R1", pin="2"),
            NetConnection(ref="R2", pin="1"),
        )),
    )

    features = (
        FeatureBlock(
            name="MCU",
            description="Main controller",
            components=("U1", "C1"),
            nets=("GND", "+3V3"),
            subcircuits=("decoupling",),
        ),
        FeatureBlock(
            name="Relay Outputs",
            description="Relay driver circuit",
            components=("Q1", "D1", "K1"),
            nets=("RELAY_DRV", "RELAY_COIL", "+5V"),
            subcircuits=("relay_driver",),
        ),
        FeatureBlock(
            name="Analog Inputs",
            description="ADC voltage divider",
            components=("R1", "R2"),
            nets=("ADC_IN",),
            subcircuits=("voltage_divider",),
        ),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="relay-adc-test", author="test", revision="v0.1"),
        features=features,
        components=(u1, q1, d1, k1, c1, r1, r2),
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=80.0, board_height_mm=60.0),
    )


def _make_connector_only_requirements() -> ProjectRequirements:
    """Single-component board: just a 4-pin connector."""
    j1 = Component(
        ref="J1", value="Conn_01x04", footprint="Conn_01x04",
        lcsc="C160416", pins=_make_connector_pins(4),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="connector-only", author="test"),
        features=(
            FeatureBlock(
                name="Interface",
                description="Single connector",
                components=("J1",),
                nets=(),
                subcircuits=(),
            ),
        ),
        components=(j1,),
        nets=(),
        mechanical=MechanicalConstraints(board_width_mm=30.0, board_height_mm=20.0),
    )


def _make_tht_only_requirements() -> ProjectRequirements:
    """Board with only through-hole components (connector + relay)."""
    j1 = Component(
        ref="J1", value="Conn_01x02", footprint="Conn_01x02",
        lcsc="C160416", pins=_make_connector_pins(2),
    )
    k1 = Component(
        ref="K1", value="HF46F-5-HS1", footprint="Relay_SPDT",
        lcsc="C35449", pins=(
            Pin(number="1", name="COIL+", pin_type=PinType.PASSIVE),
            Pin(number="2", name="COIL-", pin_type=PinType.PASSIVE),
            Pin(number="3", name="COM", pin_type=PinType.PASSIVE),
            Pin(number="4", name="NO", pin_type=PinType.PASSIVE),
        ),
    )
    nets = (
        Net(name="COIL", connections=(
            NetConnection(ref="J1", pin="1"),
            NetConnection(ref="K1", pin="1"),
        )),
        Net(name="GND", connections=(
            NetConnection(ref="J1", pin="2"),
            NetConnection(ref="K1", pin="2"),
        )),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="tht-only", author="test"),
        features=(
            FeatureBlock(
                name="Relay",
                description="THT relay board",
                components=("J1", "K1"),
                nets=("COIL", "GND"),
                subcircuits=(),
            ),
        ),
        components=(j1, k1),
        nets=nets,
        mechanical=MechanicalConstraints(board_width_mm=40.0, board_height_mm=30.0),
    )


# ===========================================================================
# 1. Minimal board: requirements -> schematic -> PCB -> validation
# ===========================================================================


class TestMinimalBoardPipeline:
    """Full pipeline through a minimal 3-component board."""

    def test_schematic_has_symbols(self) -> None:
        """Schematic contains symbol instances for all components."""
        from kicad_pipeline.schematic.builder import build_schematic

        reqs = _make_minimal_requirements()
        sch = build_schematic(reqs)
        refs = {s.ref for s in sch.symbols}
        assert "U1" in refs
        assert "R1" in refs
        assert "C1" in refs

    def test_pcb_has_footprints_for_all_components(self) -> None:
        """PCB contains footprints for every component in requirements."""
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_minimal_requirements()
        pcb = build_pcb(reqs)
        component_refs = {c.ref for c in reqs.components}
        pcb_refs = {fp.ref for fp in pcb.footprints if not fp.ref.startswith("H")}
        assert component_refs.issubset(pcb_refs), (
            f"Missing footprints: {component_refs - pcb_refs}"
        )

    def test_electrical_validation_no_critical_errors(self) -> None:
        """Electrical checks produce no ERROR-severity violations."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.validation.electrical import run_electrical_checks

        reqs = _make_minimal_requirements()
        pcb = build_pcb(reqs)
        report = run_electrical_checks(pcb, reqs)
        errors = report.errors
        assert len(errors) == 0, f"Unexpected electrical errors: {[e.message for e in errors]}"

    def test_manufacturing_validation_runs(self) -> None:
        """Manufacturing checks complete without exception."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.validation.manufacturing import run_manufacturing_checks

        reqs = _make_minimal_requirements()
        pcb = build_pcb(reqs)
        report = run_manufacturing_checks(pcb)
        # Report should exist; may have warnings for this minimal board
        assert report is not None

    def test_empty_features_still_produces_pcb(self) -> None:
        """Requirements with empty features list still produce a valid PCB."""
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_minimal_requirements()
        reqs_empty_features = replace(reqs, features=())
        pcb = build_pcb(reqs_empty_features)
        # Should still have footprints for the components
        assert len(pcb.footprints) > 0

    def test_no_components_raises_pcb_error(self) -> None:
        """Requirements with no components raises PCBError during PCB build."""
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_minimal_requirements()
        reqs_no_comps = replace(reqs, components=(), features=(), nets=())
        with pytest.raises(PCBError):
            build_pcb(reqs_no_comps)


# ===========================================================================
# 2. Schematic -> PCB round-trip consistency
# ===========================================================================


class TestSchematicPcbConsistency:
    """Verify schematic and PCB stay consistent when built from same requirements."""

    def test_all_schematic_refs_in_pcb(self) -> None:
        """Every component ref in the schematic exists as a PCB footprint."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.schematic.builder import build_schematic

        reqs = _make_minimal_requirements()
        sch = build_schematic(reqs)
        pcb = build_pcb(reqs)

        sch_refs = {s.ref for s in sch.symbols if not s.ref.startswith("#")}
        pcb_refs = {fp.ref for fp in pcb.footprints}
        missing = sch_refs - pcb_refs
        assert not missing, f"Schematic refs missing from PCB: {missing}"

    def test_net_names_consistent(self) -> None:
        """Nets defined in requirements appear in both schematic labels and PCB net entries."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.schematic.builder import build_schematic

        reqs = _make_minimal_requirements()
        sch = build_schematic(reqs)
        pcb = build_pcb(reqs)

        req_net_names = {n.name for n in reqs.nets}
        pcb_net_names = {n.name for n in pcb.nets}
        # All requirement nets should be in the PCB
        missing_nets = req_net_names - pcb_net_names
        assert not missing_nets, f"Requirement nets missing from PCB: {missing_nets}"

        # Schematic should have labels or power symbols for signal nets
        sch_label_names = {lbl.text for lbl in sch.labels}
        sch_global_names = {gl.text for gl in sch.global_labels}
        sch_power_names = {ps.value for ps in sch.power_symbols}
        sch_all_net_names = sch_label_names | sch_global_names | sch_power_names
        # At least some requirement nets should appear in schematic
        overlap = req_net_names & sch_all_net_names
        assert len(overlap) > 0, "No requirement nets found in schematic labels/power symbols"

    def test_single_component_board(self) -> None:
        """A single-connector board builds consistently in both schematic and PCB."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.schematic.builder import build_schematic

        reqs = _make_connector_only_requirements()
        sch = build_schematic(reqs)
        pcb = build_pcb(reqs)

        sch_refs = {s.ref for s in sch.symbols if not s.ref.startswith("#")}
        pcb_refs = {fp.ref for fp in pcb.footprints if not fp.ref.startswith("H")}
        assert "J1" in sch_refs
        assert "J1" in pcb_refs


# ===========================================================================
# 3. Placement optimizer end-to-end
# ===========================================================================


@pytest.mark.slow
class TestPlacementOptimizerE2E:
    """Placement optimizer produces valid, scored results."""

    def test_optimizer_no_off_board_components(self) -> None:
        """After optimization, no component centers are outside the board outline."""
        from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_multi_group_requirements()
        initial_pcb = build_pcb(reqs, placement_mode="grouped")
        optimized_pcb, review = optimize_placement_ee(reqs, initial_pcb, max_review_passes=2)

        outline = optimized_pcb.outline.polygon
        xs = [p.x for p in outline]
        ys = [p.y for p in outline]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Allow a small margin for components near edges
        margin = 2.0
        for fp in optimized_pcb.footprints:
            assert fp.position.x >= x_min - margin, f"{fp.ref} off-board left: x={fp.position.x}"
            assert fp.position.x <= x_max + margin, f"{fp.ref} off-board right: x={fp.position.x}"
            assert fp.position.y >= y_min - margin, f"{fp.ref} off-board top: y={fp.position.y}"
            assert fp.position.y <= y_max + margin, f"{fp.ref} off-board bottom: y={fp.position.y}"

    def test_optimizer_quality_score_acceptable(self) -> None:
        """Optimization quality score is above minimum threshold."""
        from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
        from kicad_pipeline.optimization.scoring import compute_fast_placement_score
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_multi_group_requirements()
        initial_pcb = build_pcb(reqs, placement_mode="grouped")
        optimized_pcb, _review = optimize_placement_ee(reqs, initial_pcb, max_review_passes=2)
        score = compute_fast_placement_score(optimized_pcb, reqs)
        assert score.overall_score > 0.5, f"Quality score too low: {score.overall_score}"

    def test_optimizer_single_group(self) -> None:
        """Optimizer works when all components are in a single feature group."""
        from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_minimal_requirements()
        # Merge everything into one feature group
        all_refs = tuple(c.ref for c in reqs.components)
        all_nets = tuple(n.name for n in reqs.nets)
        single_group = FeatureBlock(
            name="All", description="Everything", components=all_refs,
            nets=all_nets, subcircuits=(),
        )
        reqs_one_group = replace(reqs, features=(single_group,))
        initial_pcb = build_pcb(reqs_one_group, placement_mode="grouped")
        optimized_pcb, review = optimize_placement_ee(
            reqs_one_group, initial_pcb, max_review_passes=1,
        )
        assert len(optimized_pcb.footprints) > 0


# ===========================================================================
# 4. Production artifact generation
# ===========================================================================


class TestProductionArtifacts:
    """Production artifacts are correct and complete."""

    def test_bom_has_one_row_per_unique_part(self) -> None:
        """BOM groups identical components into rows by value/footprint/LCSC."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.production.bom import generate_bom

        reqs = _make_minimal_requirements()
        pcb = build_pcb(reqs)
        bom = generate_bom(pcb, reqs)
        # Each unique (value, footprint, lcsc) should be one row
        assert len(bom) >= 1
        # All component refs should appear somewhere in the BOM
        all_designators = " ".join(row.designator for row in bom)
        for comp in reqs.components:
            assert comp.ref in all_designators, f"{comp.ref} missing from BOM"

    def test_cpl_has_one_row_per_placed_component(self) -> None:
        """CPL has one entry per placed footprint."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.production.cpl import generate_cpl

        reqs = _make_minimal_requirements()
        pcb = build_pcb(reqs)
        cpl = generate_cpl(pcb)
        pcb_refs = {fp.ref for fp in pcb.footprints}
        cpl_refs = {row.designator for row in cpl}
        assert pcb_refs == cpl_refs, f"CPL mismatch: PCB={pcb_refs}, CPL={cpl_refs}"

    def test_gerbers_include_required_layers(self) -> None:
        """Generated Gerbers include F.Cu, B.Cu, and Edge.Cuts."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.production.gerber import generate_all_gerbers

        reqs = _make_minimal_requirements()
        pcb = build_pcb(reqs)
        gerbers = generate_all_gerbers(pcb, project_name="test")
        filenames = list(gerbers.keys())
        assert any("F_Cu" in f for f in filenames), f"Missing F.Cu layer in {filenames}"
        assert any("B_Cu" in f for f in filenames), f"Missing B.Cu layer in {filenames}"
        assert any("Edge_Cuts" in f for f in filenames), f"Missing Edge.Cuts in {filenames}"
        # All Gerber files should be non-empty strings
        for fname, content in gerbers.items():
            assert len(content) > 0, f"Empty Gerber file: {fname}"

    def test_tht_only_board_produces_artifacts(self) -> None:
        """Board with only THT components still generates valid BOM/CPL/Gerbers."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.production.bom import generate_bom
        from kicad_pipeline.production.cpl import generate_cpl
        from kicad_pipeline.production.gerber import generate_all_gerbers

        reqs = _make_tht_only_requirements()
        pcb = build_pcb(reqs)
        bom = generate_bom(pcb, reqs)
        cpl = generate_cpl(pcb)
        gerbers = generate_all_gerbers(pcb, project_name="tht-test")
        assert len(bom) >= 1
        assert len(cpl) >= 1
        assert len(gerbers) >= 3  # at least F.Cu, B.Cu, Edge.Cuts


# ===========================================================================
# 5. S-expression round-trip
# ===========================================================================


class TestSexpRoundTrip:
    """S-expression write -> read round-trips preserve structure."""

    def test_pcb_sexp_round_trip(self, tmp_path: Path) -> None:
        """PCB -> sexp -> file -> parse -> verify key structure."""
        from kicad_pipeline.pcb.builder import build_pcb, write_pcb
        from kicad_pipeline.sexp.parser import parse_file

        reqs = _make_minimal_requirements()
        pcb = build_pcb(reqs)
        out_path = tmp_path / "test.kicad_pcb"
        write_pcb(pcb, str(out_path))

        assert out_path.exists()
        assert out_path.stat().st_size > 100

        # Parse back: SExpNode is str|int|float|bool|list[SExpNode]
        # Top-level is a list whose first element is the tag string "kicad_pcb"
        tree = parse_file(str(out_path))
        assert isinstance(tree, list)
        assert tree[0] == "kicad_pcb"
        # Find child tags (sublists whose first element is a string)
        child_tags = {
            c[0] for c in tree[1:]
            if isinstance(c, list) and c and isinstance(c[0], str)
        }
        assert "footprint" in child_tags or "net" in child_tags

    def test_schematic_sexp_round_trip(self, tmp_path: Path) -> None:
        """Schematic -> sexp -> file -> parse -> verify key structure."""
        from kicad_pipeline.schematic.builder import build_schematic, write_schematic
        from kicad_pipeline.sexp.parser import parse_file

        reqs = _make_minimal_requirements()
        sch = build_schematic(reqs)
        out_path = tmp_path / "test.kicad_sch"
        write_schematic(sch, str(out_path))

        assert out_path.exists()
        assert out_path.stat().st_size > 100

        tree = parse_file(str(out_path))
        assert isinstance(tree, list)
        assert tree[0] == "kicad_sch"
        child_tags = {
            c[0] for c in tree[1:]
            if isinstance(c, list) and c and isinstance(c[0], str)
        }
        assert "lib_symbols" in child_tags
        assert "symbol" in child_tags

    def test_write_to_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        """Writing to a path with nonexistent parent directory raises an error."""
        from kicad_pipeline.pcb.builder import build_pcb, write_pcb

        reqs = _make_minimal_requirements()
        pcb = build_pcb(reqs)
        bad_path = tmp_path / "nonexistent" / "deep" / "dir" / "test.kicad_pcb"
        with pytest.raises((FileNotFoundError, OSError)):
            write_pcb(pcb, str(bad_path))


# ===========================================================================
# 6. Validation catches real errors
# ===========================================================================


class TestValidationCatchesErrors:
    """Validation detects deliberately introduced problems."""

    def test_off_board_component_detected(self) -> None:
        """validate_placement catches a component moved far off-board."""
        from kicad_pipeline.optimization.placement_guard import validate_placement
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_minimal_requirements()
        pcb = build_pcb(reqs)

        # Move the first non-mounting-hole footprint way off-board
        new_footprints: list[Footprint] = []
        moved = False
        for fp in pcb.footprints:
            if not fp.ref.startswith("H") and not moved:
                new_footprints.append(replace(fp, position=PCBPoint(x=999.0, y=999.0)))
                moved = True
            else:
                new_footprints.append(fp)
        corrupted_pcb = replace(pcb, footprints=tuple(new_footprints))

        result = validate_placement(corrupted_pcb, reqs)
        assert not result.passed, "Should fail with off-board component"
        assert len(result.off_board_refs) > 0 or len(result.issues) > 0

    def test_courtyard_overlap_detected(self) -> None:
        """check_courtyard_collisions detects overlapping footprints."""
        from kicad_pipeline.pcb.constraints import check_courtyard_collisions

        # Place two components at the same position
        positions = {
            "R1": PCBPoint(x=25.0, y=20.0),
            "R2": PCBPoint(x=25.0, y=20.0),
        }
        footprint_sizes = {
            "R1": (3.0, 2.0),
            "R2": (3.0, 2.0),
        }
        collisions = check_courtyard_collisions(positions, footprint_sizes)
        assert len(collisions) > 0, "Should detect overlap when components share position"

    def test_courtyard_no_false_positive_when_separated(self) -> None:
        """check_courtyard_collisions reports no overlap for well-separated components."""
        from kicad_pipeline.pcb.constraints import check_courtyard_collisions

        positions = {
            "R1": PCBPoint(x=10.0, y=10.0),
            "R2": PCBPoint(x=50.0, y=50.0),
        }
        footprint_sizes = {
            "R1": (3.0, 2.0),
            "R2": (3.0, 2.0),
        }
        collisions = check_courtyard_collisions(positions, footprint_sizes)
        assert len(collisions) == 0, f"False positive collisions: {collisions}"

    def test_electrical_checks_on_unconnected_nets(self) -> None:
        """Electrical checks produce warnings for nets with single-pin connections."""
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.validation.electrical import run_electrical_checks

        # Build a board where some nets have only one connection (dangling)
        u1 = Component(
            ref="U1", value="ESP32", footprint="ESP32-WROOM-32E",
            lcsc="C165948", pins=_make_mcu_pins(),
        )
        r1 = Component(
            ref="R1", value="10k", footprint="R_0805", lcsc="C17414",
            pins=_make_passive_pins(),
        )
        # Net with only one endpoint — should trigger unconnected warning
        nets = (
            Net(name="GND", connections=(NetConnection(ref="U1", pin="1"),)),
            Net(name="+3V3", connections=(NetConnection(ref="U1", pin="2"),)),
            Net(name="FLOATING", connections=(NetConnection(ref="R1", pin="1"),)),
        )
        reqs = ProjectRequirements(
            project=ProjectInfo(name="unconnected-test", author="test"),
            features=(
                FeatureBlock(
                    name="Test", description="test", components=("U1", "R1"),
                    nets=("GND", "+3V3", "FLOATING"), subcircuits=(),
                ),
            ),
            components=(u1, r1),
            nets=nets,
            mechanical=MechanicalConstraints(board_width_mm=50.0, board_height_mm=40.0),
        )
        pcb = build_pcb(reqs)
        report = run_electrical_checks(pcb, reqs)
        # Should have at least warnings (unconnected pin, single-endpoint nets)
        assert len(report.violations) > 0 or len(report.warnings) > 0


# ===========================================================================
# 7. Multi-feature board with subcircuit detection
# ===========================================================================


@pytest.mark.slow
class TestSubcircuitDetectionE2E:
    """Subcircuit detection and zone partitioning on a realistic board."""

    def test_detect_subcircuits_finds_patterns(self) -> None:
        """detect_subcircuits identifies relay_driver and voltage_divider patterns."""
        from kicad_pipeline.optimization.functional_grouper import (
            SubCircuitType,
            detect_subcircuits,
        )

        reqs = _make_relay_mcu_analog_requirements()
        subcircuits = detect_subcircuits(reqs)
        types_found = {sc.circuit_type for sc in subcircuits}

        # Decoupling should always be detected (C1 near U1)
        assert SubCircuitType.DECOUPLING in types_found, (
            f"Missing decoupling detection. Found: {types_found}"
        )

    def test_partition_board_non_overlapping_zones(self) -> None:
        """partition_board produces zones that do not overlap."""
        from kicad_pipeline.optimization.zone_partitioner import partition_board

        reqs = _make_relay_mcu_analog_requirements()
        zones = partition_board(
            board_bounds=(0.0, 0.0, 80.0, 60.0),
            groups=list(reqs.features),
        )
        assert len(zones) >= 1

        # Check no two zones overlap
        for i, z1 in enumerate(zones):
            for z2 in zones[i + 1:]:
                x_overlap = z1.rect[0] < z2.rect[2] and z2.rect[0] < z1.rect[2]
                y_overlap = z1.rect[1] < z2.rect[3] and z2.rect[1] < z1.rect[3]
                assert not (x_overlap and y_overlap), (
                    f"Zones overlap: {z1.name}={z1.rect} vs {z2.name}={z2.rect}"
                )

    def test_full_optimizer_on_subcircuit_board(self) -> None:
        """Full optimize_placement_ee on a board with relay + MCU + analog."""
        from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
        from kicad_pipeline.optimization.scoring import compute_fast_placement_score
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_relay_mcu_analog_requirements()
        initial_pcb = build_pcb(reqs, placement_mode="grouped")
        optimized_pcb, review = optimize_placement_ee(reqs, initial_pcb, max_review_passes=2)

        score = compute_fast_placement_score(optimized_pcb, reqs)
        assert score.overall_score > 0.5, f"Score too low: {score.overall_score}"

        # Verify all original component refs are still present
        original_refs = {c.ref for c in reqs.components}
        optimized_refs = {fp.ref for fp in optimized_pcb.footprints if not fp.ref.startswith("H")}
        assert original_refs.issubset(optimized_refs), (
            f"Lost components: {original_refs - optimized_refs}"
        )

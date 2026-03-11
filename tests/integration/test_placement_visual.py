"""Integration test: visual placement verification for a complex board.

Builds a multi-domain board (relays + MCU + analog + RF) from requirements,
runs the EE placement optimizer, renders to PNG, and verifies placement
quality scores and violation counts.

This test serves as both a regression gate and a visual review artifact.
"""

from __future__ import annotations

import math
import sys
import types

import pytest

# Skip if matplotlib not available
matplotlib = pytest.importorskip("matplotlib")

# Skip if nl-s-3c build script not available
NL_S3C_PATH = "/Users/johnbetts/Dropbox/Source/nl-s-3c-complete/build_with_pipeline.py"


def _load_nl_s3c_module() -> types.ModuleType:
    """Load the nl-s-3c build script as a module (without running main)."""
    from pathlib import Path

    build_path = Path(NL_S3C_PATH)
    if not build_path.exists():
        pytest.skip("nl-s-3c build script not found")

    mod = types.ModuleType("build_nl_s3c")
    mod.__file__ = str(build_path)
    sys.modules["build_nl_s3c"] = mod

    source = build_path.read_text()
    cut = source.find("if __name__")
    if cut > 0:
        source = source[:cut]
    exec(compile(source, str(build_path), "exec"), mod.__dict__)
    return mod


def _build_requirements() -> object:
    """Build ProjectRequirements from nl-s-3c definitions."""
    from kicad_pipeline.models.requirements import (
        BoardContext,
        MechanicalConstraints,
        ProjectInfo,
        ProjectRequirements,
    )

    mod = _load_nl_s3c_module()
    components = mod._make_components()
    nets = mod._make_nets(components)
    features = mod._make_features(components)

    return ProjectRequirements(
        project=ProjectInfo(
            name="NL-S-3C-Complete",
            author="Test",
            revision="v0.1",
            description="Integration test board",
        ),
        features=tuple(features),
        components=tuple(components),
        nets=tuple(nets),
        mechanical=MechanicalConstraints(
            board_width_mm=140.0, board_height_mm=80.0,
        ),
        board_context=BoardContext(
            target_system="Test",
            shared_grounds=True,
            notes=(),
        ),
    )


@pytest.fixture(scope="module")
def placement_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build PCB and run EE optimizer (cached for all tests in module)."""
    from kicad_pipeline.optimization.functional_grouper import (
        classify_voltage_domains,
        compute_power_flow_topology,
        detect_cross_domain_affinities,
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.placement_optimizer import (
        _count_collisions,
        _fp_courtyard_sizes,
        optimize_placement_ee,
    )
    from kicad_pipeline.optimization.scoring import compute_fast_placement_score
    from kicad_pipeline.pcb.builder import build_pcb
    from kicad_pipeline.visualization.placement_render import render_placement

    requirements = _build_requirements()

    pcb = build_pcb(
        requirements, auto_route=False, placement_mode="grouped",
        layer_count=4, preserve_routing=False, skip_inner_zones=True,
    )

    pcb_opt, review = optimize_placement_ee(requirements, pcb)
    score = compute_fast_placement_score(pcb_opt, requirements)

    subcircuits = detect_subcircuits(requirements)
    domain_map = classify_voltage_domains(requirements)
    affinities = detect_cross_domain_affinities(requirements, domain_map)
    topology = compute_power_flow_topology(subcircuits)

    # Count post-optimization collisions
    fp_sizes = _fp_courtyard_sizes(pcb_opt)
    positions = {
        fp.ref: (fp.position.x, fp.position.y, fp.rotation)
        for fp in pcb_opt.footprints
    }
    post_collisions = _count_collisions(positions, fp_sizes)

    # Render to temp directory
    out_dir = tmp_path_factory.mktemp("placement")
    render_path = render_placement(
        pcb_opt, requirements, out_dir / "placement.png",
        title="NL-S-3C Integration Test",
        score=score, domain_map=domain_map,
    )

    return {
        "pcb": pcb_opt,
        "requirements": requirements,
        "review": review,
        "score": score,
        "subcircuits": subcircuits,
        "domain_map": domain_map,
        "affinities": affinities,
        "topology": topology,
        "post_collisions": post_collisions,
        "render_path": render_path,
    }


class TestPlacementQuality:
    """Verify placement quality meets minimum thresholds."""

    def test_overall_score_above_threshold(self, placement_result: dict) -> None:
        score = placement_result["score"]
        assert score.overall_score >= 0.85, (
            f"Overall score {score.overall_score:.3f} below 0.85 threshold"
        )

    def test_grade_is_acceptable(self, placement_result: dict) -> None:
        score = placement_result["score"]
        assert score.grade in ("A", "B"), f"Grade {score.grade} is below B"

    def test_collision_score_acceptable(self, placement_result: dict) -> None:
        score = placement_result["score"]
        collision = next(
            e for e in score.breakdown if e.category == "Collisions"
        )
        assert collision.score >= 0.5, (
            f"Collision score {collision.score:.3f} too low"
        )

    def test_cohesion_score_acceptable(self, placement_result: dict) -> None:
        score = placement_result["score"]
        cohesion = next(
            e for e in score.breakdown if "Cohesion" in e.category
        )
        assert cohesion.score >= 0.8, (
            f"Cohesion score {cohesion.score:.3f} too low"
        )

    def test_all_components_on_board(self, placement_result: dict) -> None:
        pcb = placement_result["pcb"]
        outline = pcb.outline
        if not outline or not outline.polygon:
            pytest.skip("No outline")

        xs = [p.x for p in outline.polygon]
        ys = [p.y for p in outline.polygon]
        bx0, bx1 = min(xs), max(xs)
        by0, by1 = min(ys), max(ys)

        margin = 5.0  # allow small overshoot
        off_board = []
        for fp in pcb.footprints:
            if (fp.position.x < bx0 - margin or fp.position.x > bx1 + margin
                    or fp.position.y < by0 - margin or fp.position.y > by1 + margin):
                off_board.append(fp.ref)

        assert not off_board, f"Components off board: {off_board}"

    def test_critical_violations_limited(self, placement_result: dict) -> None:
        review = placement_result["review"]
        critical = [v for v in review.violations if v.severity == "critical"]
        assert len(critical) <= 5, (
            f"{len(critical)} critical violations (max 5)"
        )

    def test_no_post_optimization_collisions(self, placement_result: dict) -> None:
        """Phase 4.5 should resolve all collisions after review fixes."""
        post_collisions = placement_result["post_collisions"]
        assert len(post_collisions) == 0, (
            f"{len(post_collisions)} collisions remain after optimization: "
            f"{post_collisions[:5]}"
        )


class TestSubcircuitDetection:
    """Verify subcircuit detection works on a real board."""

    def test_relay_drivers_detected(self, placement_result: dict) -> None:
        from kicad_pipeline.optimization.functional_grouper import SubCircuitType
        subcircuits = placement_result["subcircuits"]
        relay_scs = [
            sc for sc in subcircuits
            if sc.circuit_type == SubCircuitType.RELAY_DRIVER
        ]
        assert len(relay_scs) >= 4, f"Expected 4+ relay drivers, got {len(relay_scs)}"

    def test_relay_layout_hint_is_row(self, placement_result: dict) -> None:
        from kicad_pipeline.optimization.functional_grouper import SubCircuitType
        subcircuits = placement_result["subcircuits"]
        for sc in subcircuits:
            if sc.circuit_type == SubCircuitType.RELAY_DRIVER:
                assert sc.layout_hint == "row", (
                    f"Relay {sc.anchor_ref} has hint '{sc.layout_hint}', expected 'row'"
                )

    def test_buck_converters_detected(self, placement_result: dict) -> None:
        from kicad_pipeline.optimization.functional_grouper import SubCircuitType
        subcircuits = placement_result["subcircuits"]
        bucks = [
            sc for sc in subcircuits
            if sc.circuit_type == SubCircuitType.BUCK_CONVERTER
        ]
        assert len(bucks) >= 1, "Expected at least 1 buck converter"

    def test_rf_antenna_detected(self, placement_result: dict) -> None:
        from kicad_pipeline.optimization.functional_grouper import SubCircuitType
        subcircuits = placement_result["subcircuits"]
        rf = [
            sc for sc in subcircuits
            if sc.circuit_type == SubCircuitType.RF_ANTENNA
        ]
        assert len(rf) >= 1, "Expected RF antenna subcircuit"

    def test_cross_domain_affinities_detected(self, placement_result: dict) -> None:
        affinities = placement_result["affinities"]
        assert len(affinities) >= 1, "Expected cross-domain affinities"


class TestTopologyOrdering:
    """Verify v3 power flow topology and zone ordering."""

    def test_topology_has_domain_order(self, placement_result: dict) -> None:
        """Power flow topology should produce a non-empty domain order."""
        topology = placement_result["topology"]
        assert len(topology.domain_order) >= 2, (
            f"Expected 2+ domains in topology, got {len(topology.domain_order)}"
        )

    def test_topology_highest_voltage_first(self, placement_result: dict) -> None:
        """First domain in topology should have higher voltage than last."""
        from kicad_pipeline.optimization.functional_grouper import (
            _voltage_magnitude,
        )
        topology = placement_result["topology"]
        first_v = _voltage_magnitude(topology.domain_order[0])
        last_v = _voltage_magnitude(topology.domain_order[-1])
        assert first_v >= last_v, (
            f"First domain {topology.domain_order[0].value} ({first_v}V) should have "
            f">= voltage than last {topology.domain_order[-1].value} ({last_v}V)"
        )

    def test_topology_regulators_detected(self, placement_result: dict) -> None:
        """Board with buck/LDO should have regulator boundaries."""
        topology = placement_result["topology"]
        assert len(topology.regulator_boundaries) >= 1, (
            "Expected at least 1 regulator boundary in topology"
        )

    def test_high_voltage_components_separated_from_low(
        self, placement_result: dict,
    ) -> None:
        """24V components should be spatially separated from 3.3V components."""
        from kicad_pipeline.optimization.functional_grouper import VoltageDomain

        pcb = placement_result["pcb"]
        domain_map = placement_result["domain_map"]

        positions = {fp.ref: (fp.position.x, fp.position.y) for fp in pcb.footprints}

        # Compute centroids for 24V and 3.3V domains
        v24_positions = [
            positions[r] for r in domain_map
            if domain_map[r] == VoltageDomain.VIN_24V and r in positions
        ]
        v33_positions = [
            positions[r] for r in domain_map
            if domain_map[r] == VoltageDomain.DIGITAL_3V3 and r in positions
        ]

        if not v24_positions or not v33_positions:
            pytest.skip("Need both 24V and 3.3V domains")

        c24_x = sum(p[0] for p in v24_positions) / len(v24_positions)
        c24_y = sum(p[1] for p in v24_positions) / len(v24_positions)
        c33_x = sum(p[0] for p in v33_positions) / len(v33_positions)
        c33_y = sum(p[1] for p in v33_positions) / len(v33_positions)

        separation = math.sqrt((c24_x - c33_x) ** 2 + (c24_y - c33_y) ** 2)
        assert separation >= 15.0, (
            f"24V and 3.3V domain centroids only {separation:.1f}mm apart "
            "(expected 15mm+ separation)"
        )


class TestRelayTerminalEdge:
    """Verify relay terminal connectors are on the same edge as the relay bank."""

    def test_relay_terminal_connectors_near_relays(
        self, placement_result: dict,
    ) -> None:
        """Connectors in relay subcircuits should be near the relay bank."""
        from kicad_pipeline.optimization.functional_grouper import SubCircuitType

        pcb = placement_result["pcb"]
        subcircuits = placement_result["subcircuits"]
        positions = {fp.ref: (fp.position.x, fp.position.y) for fp in pcb.footprints}

        # Find relay anchors
        relay_anchors = []
        for sc in subcircuits:
            if sc.circuit_type == SubCircuitType.RELAY_DRIVER and sc.anchor_ref in positions:
                relay_anchors.append(positions[sc.anchor_ref])

        if not relay_anchors:
            pytest.skip("No relay drivers in layout")

        # Relay bank centroid
        relay_cx = sum(p[0] for p in relay_anchors) / len(relay_anchors)
        relay_cy = sum(p[1] for p in relay_anchors) / len(relay_anchors)

        # Find connectors in any relay subcircuit
        relay_connectors = []
        for sc in subcircuits:
            if sc.circuit_type == SubCircuitType.RELAY_DRIVER:
                for ref in sc.refs:
                    if ref.startswith("J") and ref in positions:
                        relay_connectors.append((ref, positions[ref]))

        if not relay_connectors:
            pytest.skip("No connectors in relay subcircuits")

        # Each relay connector should be within 30mm of relay centroid
        for ref, (cx, cy) in relay_connectors:
            dist = math.sqrt((cx - relay_cx) ** 2 + (cy - relay_cy) ** 2)
            assert dist <= 30.0, (
                f"Relay connector {ref} is {dist:.1f}mm from relay bank centroid "
                "(max 30mm)"
            )


class TestRenderOutput:
    """Verify the render produces a valid image."""

    def test_render_file_exists(self, placement_result: dict) -> None:
        from pathlib import Path
        render_path = placement_result["render_path"]
        assert Path(render_path).exists()

    def test_render_file_has_content(self, placement_result: dict) -> None:
        from pathlib import Path
        render_path = placement_result["render_path"]
        assert Path(render_path).stat().st_size > 10000

"""Integration test: visual placement verification for a complex board.

Builds a multi-domain board (relays + MCU + analog + RF) from requirements,
runs the EE placement optimizer, renders to PNG, and verifies placement
quality scores and violation counts.

This test serves as both a regression gate and a visual review artifact.
"""

from __future__ import annotations

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
        detect_cross_domain_affinities,
        detect_subcircuits,
    )
    from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
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

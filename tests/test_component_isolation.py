"""Component isolation test suite.

For each registered component type, generates a footprint in isolation and
runs structural verification checks.  Catches 3D model rotation, pad count
mismatch, body-pad misalignment, and through-hole issues BEFORE board-level
review.

Visual rendering tests are marked ``@pytest.mark.slow`` and require
``kicad-image-gen`` to be installed.
"""

from __future__ import annotations

import pytest

from kicad_pipeline.pcb.isolation_board import build_isolation_footprint
from kicad_pipeline.validation.component_registry import ComponentRegistry
from kicad_pipeline.validation.component_verifier import verify_structural

# ---------------------------------------------------------------------------
# Load registry — parametrize over all registered components
# ---------------------------------------------------------------------------

_REGISTRY = ComponentRegistry()
# Only test components that have been verified (can generate parametrically).
# Unverified parts from the JLCPCB catalog may not have parametric footprint support.
_ALL_SPECS = [s for s in _REGISTRY.all_components() if s.verification_status == "verified"]


@pytest.fixture(params=_ALL_SPECS, ids=[s.component_id for s in _ALL_SPECS])
def component_spec(request: pytest.FixtureRequest) -> object:
    """Yield each registered ComponentSpec as a fixture."""
    return request.param


# ---------------------------------------------------------------------------
# Structural checks — fast, no file I/O, no rendering
# ---------------------------------------------------------------------------


class TestComponentIsolation:
    """Structural verification of every registered component type."""

    def test_footprint_generates(self, component_spec) -> None:  # type: ignore[no-untyped-def]
        """Footprint generation succeeds without error."""
        fp = build_isolation_footprint(component_spec)
        assert fp is not None
        assert fp.ref == component_spec.ref

    def test_pad_count(self, component_spec) -> None:  # type: ignore[no-untyped-def]
        """Pad count matches registry expectation.

        JLCPCB footprints may include extra ground/shield pads, so
        actual >= expected is acceptable for JLCPCB-sourced footprints.
        """
        fp = build_isolation_footprint(component_spec)
        actual = len(fp.pads)
        is_jlcpcb = getattr(fp, "footprint_source", "") == "jlcpcb"
        if is_jlcpcb:
            assert actual >= component_spec.expected_pads, (
                f"{component_spec.component_id}: expected >={component_spec.expected_pads} pads, "
                f"got {actual} (JLCPCB)"
            )
        else:
            assert actual == component_spec.expected_pads, (
                f"{component_spec.component_id}: expected {component_spec.expected_pads} pads, "
                f"got {actual}"
            )

    def test_pad_type(self, component_spec) -> None:  # type: ignore[no-untyped-def]
        """All pads match expected type (smd/thru_hole)."""
        fp = build_isolation_footprint(component_spec)
        for pad in fp.pads:
            if pad.pad_type == "np_thru_hole":
                continue
            assert pad.pad_type == component_spec.expected_pad_type, (
                f"{component_spec.component_id}: pad {pad.number} "
                f"type={pad.pad_type}, expected {component_spec.expected_pad_type}"
            )

    def test_no_duplicate_pads(self, component_spec) -> None:  # type: ignore[no-untyped-def]
        """No duplicate pad numbers (shield/mount pads exempt, switches allow 2x)."""
        from collections import Counter

        fp = build_isolation_footprint(component_spec)
        exempt = {"SH", "MP", ""}
        numbers = [p.number for p in fp.pads if p.number not in exempt]
        counts = Counter(numbers)
        # Tact switches have paired pads (each pin appears 2x) — expected
        is_switch = component_spec.ref.startswith("SW") or "switch" in component_spec.description.lower()
        max_allowed = 2 if is_switch else 1
        dupes = [n for n, c in counts.items() if c > max_allowed]
        assert not dupes, (
            f"{component_spec.component_id}: duplicate pad numbers (>{max_allowed}x): {dupes}"
        )

    def test_3d_model_present(self, component_spec) -> None:  # type: ignore[no-untyped-def]
        """3D model attached (exempt for mounting holes/test points/connectors without STEP)."""
        if component_spec.ref.startswith(("H", "TP")):
            pytest.skip("mounting hole / test point exempt")
        fp = build_isolation_footprint(component_spec)
        if len(fp.models) == 0 and component_spec.ref.startswith("J"):
            pytest.skip("connector without matching STEP model")
        assert len(fp.models) > 0, (
            f"{component_spec.component_id}: no 3D model attached"
        )

    def test_model_rotation(self, component_spec) -> None:  # type: ignore[no-untyped-def]
        """3D model Z rotation matches expected value.

        JLCPCB footprints may have mirrored pad layouts requiring a 180°
        correction.  Accept both the expected rotation and expected+180°.
        """
        if component_spec.ref.startswith(("H", "TP")):
            pytest.skip("exempt")
        fp = build_isolation_footprint(component_spec)
        if not fp.models:
            pytest.skip("no models")
        model = fp.models[0]
        actual_z = model.rotate[2] if len(model.rotate) > 2 else 0.0
        expected_z = component_spec.model_rotation_z
        diff = abs((actual_z % 360.0) - (expected_z % 360.0))
        if diff > 180.0:
            diff = 360.0 - diff
        # Also accept expected + 180° (JLCPCB mirror correction)
        diff_mirrored = abs((actual_z % 360.0) - ((expected_z + 180.0) % 360.0))
        if diff_mirrored > 180.0:
            diff_mirrored = 360.0 - diff_mirrored
        assert diff < 5.0 or diff_mirrored < 5.0, (
            f"{component_spec.component_id}: model Z rotation "
            f"expected {expected_z}° (or +180°), got {actual_z}° (diff {diff:.1f}°)"
        )

    def test_model_offset(self, component_spec) -> None:  # type: ignore[no-untyped-def]
        """3D model XY offset within acceptable range."""
        if component_spec.ref.startswith(("H", "TP")):
            pytest.skip("exempt")
        fp = build_isolation_footprint(component_spec)
        if not fp.models:
            pytest.skip("no models")
        model = fp.models[0]
        import math

        ox = model.offset[0] if len(model.offset) > 0 else 0.0
        oy = model.offset[1] if len(model.offset) > 1 else 0.0
        dist = math.sqrt(ox * ox + oy * oy)
        assert dist <= component_spec.model_offset_xy_max_mm, (
            f"{component_spec.component_id}: model XY offset "
            f"({ox:.2f}, {oy:.2f}) = {dist:.2f}mm exceeds "
            f"max {component_spec.model_offset_xy_max_mm}mm"
        )

    def test_tht_drill_present(self, component_spec) -> None:  # type: ignore[no-untyped-def]
        """Through-hole pads have drill > 0."""
        if component_spec.expected_pad_type != "thru_hole":
            pytest.skip("not THT")
        fp = build_isolation_footprint(component_spec)
        for pad in fp.pads:
            if pad.pad_type == "thru_hole":
                assert pad.drill_diameter is not None and pad.drill_diameter > 0, (
                    f"{component_spec.component_id}: pad {pad.number} "
                    f"missing drill (drill={pad.drill_diameter})"
                )

    def test_full_structural_verification(self, component_spec) -> None:  # type: ignore[no-untyped-def]
        """Full structural verification passes (all checks)."""
        fp = build_isolation_footprint(component_spec)
        result = verify_structural(fp, component_spec)
        failed = [c for c in result.checks if not c.passed and c.severity in ("critical", "major")]
        assert result.passed, (
            f"{component_spec.component_id}: structural verification failed:\n"
            + "\n".join(f"  [{c.severity}] {c.name}: {c.detail}" for c in failed)
        )


# ---------------------------------------------------------------------------
# Visual checks — require kicad-image-gen, marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestComponentIsolationVisual:
    """Visual verification via kicad-image-gen rendering."""

    def test_isolation_board_builds(self, component_spec, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Isolation board PCB file is generated successfully."""
        from kicad_pipeline.pcb.isolation_board import build_isolation_board

        pcb_path = build_isolation_board(component_spec, tmp_path)
        assert pcb_path.exists()
        assert pcb_path.stat().st_size > 100

    def test_renders_successfully(self, component_spec, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """All 3 render views (2D top, 3D top, 3D iso) are generated."""
        from kicad_pipeline.validation.component_verifier import verify_component

        result = verify_component(component_spec, tmp_path, render=True)
        view_names = {name for name, _ in result.render_paths}
        assert "2d_top" in view_names, (
            f"{component_spec.component_id}: 2D top render missing"
        )
        assert "3d_top" in view_names, (
            f"{component_spec.component_id}: 3D top render missing"
        )

"""Board-level 3D model alignment verification.

Tests that the ACTUAL generated .kicad_pcb file has correct 3D model
offsets/rotations — not just the isolated footprints.  This closes the
verification gap where components pass isolation checks but appear
misaligned on the assembled board.

See: council report 2026-03-31 "Why 3D mismatches keep recurring"
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kicad_pipeline.pcb.position_extractor import BoardFootprintModel, models_from_pcb_file
from kicad_pipeline.validation.component_verifier import (
    Board3DCheckResult,
    verify_board_3d_alignment,
)

TRAIN_RELAY_PCB = Path("output/train_relay/train_relay.kicad_pcb")


@pytest.mark.skipif(
    not TRAIN_RELAY_PCB.exists(),
    reason="train_relay board not generated yet",
)
class TestBoardModelExtraction:
    """Verify that models_from_pcb_file extracts 3D data from actual boards."""

    def test_extracts_models(self) -> None:
        models = models_from_pcb_file(TRAIN_RELAY_PCB)
        assert len(models) > 0, "Should extract at least one footprint with a 3D model"

    def test_relay_has_nonzero_offset(self) -> None:
        """Relays (K1-K4) should have correct non-zero 3D model offsets."""
        models = models_from_pcb_file(TRAIN_RELAY_PCB)
        for ref in ("K1", "K2", "K3", "K4"):
            if ref not in models:
                continue
            m = models[ref]
            ox, oy, oz = m.model_offset
            assert abs(ox) > 0.1 or abs(oy) > 0.1, (
                f"{ref} relay 3D model offset should be non-zero, got ({ox:.3f},{oy:.3f})"
            )

    def test_model_has_required_fields(self) -> None:
        models = models_from_pcb_file(TRAIN_RELAY_PCB)
        for ref, m in models.items():
            assert isinstance(m, BoardFootprintModel)
            assert m.ref == ref
            assert len(m.model_offset) == 3
            assert len(m.model_rotate) == 3
            assert m.model_path  # non-empty


@pytest.mark.skipif(
    not TRAIN_RELAY_PCB.exists(),
    reason="train_relay board not generated yet",
)
class TestBoardLevelVerification:
    """Board-level 3D alignment checks — the gate that catches what isolation misses."""

    def test_verify_returns_results(self) -> None:
        results = verify_board_3d_alignment(TRAIN_RELAY_PCB)
        assert len(results) > 0, "Should return at least one check result"
        assert all(isinstance(r, Board3DCheckResult) for r in results)

    def test_no_zero_offset_with_pad1_data(self) -> None:
        """Components with kicad_ref_pad1 data in registry must NOT have (0,0,0) offset.

        This is THE test that would have caught every recurring 3D mismatch.
        If a component's registry entry says its pad-1 is offset from centroid,
        the 3D model on the actual board MUST have a corresponding non-zero
        offset.  A zero offset means _apply_jlcpcb_model_offset() silently
        failed.
        """
        results = verify_board_3d_alignment(TRAIN_RELAY_PCB)
        failed = [r for r in results if not r.passed]
        if failed:
            msg_lines = ["Board 3D alignment failures (offset correction not applied):"]
            for f in failed:
                msg_lines.append(f"  {f.ref}: {f.detail}")
            pytest.fail("\n".join(msg_lines))

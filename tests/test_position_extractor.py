"""Tests for PCB position extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from kicad_pipeline.pcb.position_extractor import (
    positions_from_pcb_file,
    positions_from_source,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal PCB S-expression fixtures
# ---------------------------------------------------------------------------

_MINIMAL_PCB = """\
(kicad_pcb (version 20241229) (generator "test")
  (footprint "Resistor_SMD:R_0603"
    (layer "F.Cu")
    (at 25.0 30.0)
    (property "Reference" "R1"
      (at 0 -1.5) (layer "F.SilkS")
      (effects (font (size 1 1) (thickness 0.15))))
    (pad "1" smd rect (at -0.775 0) (size 0.9 0.95) (layers "F.Cu" "F.Mask" "F.Paste"))
    (pad "2" smd rect (at 0.775 0) (size 0.9 0.95) (layers "F.Cu" "F.Mask" "F.Paste"))
  )
  (footprint "Capacitor_SMD:C_0603"
    (layer "F.Cu")
    (at 40.0 50.0 90.0)
    (property "Reference" "C1"
      (at 0 -1.5) (layer "F.SilkS")
      (effects (font (size 1 1) (thickness 0.15))))
    (pad "1" smd rect (at -0.775 0) (size 0.9 0.95) (layers "F.Cu" "F.Mask" "F.Paste"))
    (pad "2" smd rect (at 0.775 0) (size 0.9 0.95) (layers "F.Cu" "F.Mask" "F.Paste"))
  )
)
"""

_LEGACY_PCB = """\
(kicad_pcb (version 20241229) (generator "test")
  (footprint "Package_SO:SOIC-8"
    (layer "F.Cu")
    (at 10.0 20.0 45.0)
    (fp_text reference "U1" (at 0 -3) (layer "F.SilkS")
      (effects (font (size 1 1) (thickness 0.15))))
    (pad "1" smd rect (at -2.7 -1.905) (size 1.5 0.6) (layers "F.Cu" "F.Mask" "F.Paste"))
  )
)
"""

_EMPTY_PCB = """\
(kicad_pcb (version 20241229) (generator "test"))
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_positions_from_minimal_pcb(tmp_path: Path) -> None:
    """Extract positions from a minimal PCB with two footprints."""
    pcb_file = tmp_path / "test.kicad_pcb"
    pcb_file.write_text(_MINIMAL_PCB)

    result = positions_from_pcb_file(pcb_file)

    assert result == {
        "R1": (25.0, 30.0, 0.0),
        "C1": (40.0, 50.0, 90.0),
    }


def test_positions_handles_missing_rotation(tmp_path: Path) -> None:
    """Footprint without rotation in (at x y) gets rotation=0.0."""
    pcb_file = tmp_path / "test.kicad_pcb"
    pcb_file.write_text(_MINIMAL_PCB)

    result = positions_from_pcb_file(pcb_file)
    assert result["R1"][2] == 0.0


def test_positions_handles_empty_board(tmp_path: Path) -> None:
    """Empty board returns empty dict."""
    pcb_file = tmp_path / "test.kicad_pcb"
    pcb_file.write_text(_EMPTY_PCB)

    result = positions_from_pcb_file(pcb_file)
    assert result == {}


def test_positions_handles_legacy_fp_text(tmp_path: Path) -> None:
    """Legacy fp_text reference format is parsed correctly."""
    pcb_file = tmp_path / "test.kicad_pcb"
    pcb_file.write_text(_LEGACY_PCB)

    result = positions_from_pcb_file(pcb_file)
    assert result == {"U1": (10.0, 20.0, 45.0)}


def test_positions_from_source_dispatches_path(tmp_path: Path) -> None:
    """Path argument dispatches to file parser."""
    pcb_file = tmp_path / "test.kicad_pcb"
    pcb_file.write_text(_MINIMAL_PCB)

    result = positions_from_source(pcb_file)
    assert "R1" in result
    assert "C1" in result


def test_positions_from_source_nonexistent_file(tmp_path: Path) -> None:
    """Non-existent file returns empty dict."""
    result = positions_from_source(tmp_path / "nonexistent.kicad_pcb")
    assert result == {}


def test_positions_from_source_dispatches_ipc() -> None:
    """KiCadConnection-like object dispatches to pull_footprint_positions."""
    mock_conn = MagicMock()
    expected = {"R1": (10.0, 20.0, 0.0)}

    # Patch at module level
    import kicad_pipeline.ipc.board_ops as board_ops_mod

    original = board_ops_mod.pull_footprint_positions
    board_ops_mod.pull_footprint_positions = MagicMock(return_value=expected)
    try:
        result = positions_from_source(mock_conn)
        assert result == expected
        board_ops_mod.pull_footprint_positions.assert_called_once_with(mock_conn)
    finally:
        board_ops_mod.pull_footprint_positions = original


def test_positions_from_source_string_path(tmp_path: Path) -> None:
    """String path argument works the same as Path."""
    pcb_file = tmp_path / "test.kicad_pcb"
    pcb_file.write_text(_MINIMAL_PCB)

    result = positions_from_source(str(pcb_file))
    assert "R1" in result

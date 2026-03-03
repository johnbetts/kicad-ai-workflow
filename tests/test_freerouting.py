"""Tests for kicad_pipeline.routing.freerouting."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    NetEntry,
    PCBDesign,
    Point,
)
from kicad_pipeline.routing.freerouting import (
    FreeRoutingResult,
    find_freerouting_jar,
    route_with_freerouting,
    ses_to_tracks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outline() -> BoardOutline:
    return BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(50.0, 0.0),
            Point(50.0, 30.0),
            Point(0.0, 30.0),
        )
    )


def _make_pcb(
    nets: tuple[NetEntry, ...] | None = None,
) -> PCBDesign:
    if nets is None:
        nets = (
            NetEntry(number=0, name=""),
            NetEntry(number=1, name="GND"),
            NetEntry(number=2, name="VCC"),
        )
    return PCBDesign(
        outline=_make_outline(),
        design_rules=DesignRules(),
        nets=nets,
        footprints=(),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


# ---------------------------------------------------------------------------
# FreeRoutingResult dataclass
# ---------------------------------------------------------------------------


def test_freerouting_result_frozen() -> None:
    result = FreeRoutingResult(
        success=True,
        ses_file="/tmp/out.ses",
        stdout="ok",
        stderr="",
    )
    with pytest.raises(AttributeError):
        result.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# find_freerouting_jar
# ---------------------------------------------------------------------------


def test_find_freerouting_jar_returns_none_when_missing() -> None:
    """Returns None when searching non-existent directories."""
    result = find_freerouting_jar(search_dirs=["/nonexistent/path/abc123"])
    # Also override default dirs by patching _DEFAULT_SEARCH_DIRS
    with patch("kicad_pipeline.routing.freerouting._DEFAULT_SEARCH_DIRS", ()):
        result = find_freerouting_jar(search_dirs=["/nonexistent/path/abc123"])
    assert result is None


def test_find_freerouting_jar_finds_in_dir(tmp_path: Path) -> None:
    """Returns the jar path when freerouting.jar exists in a search dir."""
    jar = tmp_path / "freerouting.jar"
    jar.write_bytes(b"fake jar")
    result = find_freerouting_jar(search_dirs=[str(tmp_path)])
    assert result is not None
    assert result.endswith("freerouting.jar")


def test_find_freerouting_jar_default_dirs_searched(tmp_path: Path) -> None:
    """Default search dirs are used when no search_dirs provided."""
    jar = tmp_path / "freerouting.jar"
    jar.write_bytes(b"fake")
    with patch(
        "kicad_pipeline.routing.freerouting._DEFAULT_SEARCH_DIRS",
        (str(tmp_path),),
    ):
        result = find_freerouting_jar()
    assert result is not None


# ---------------------------------------------------------------------------
# route_with_freerouting
# ---------------------------------------------------------------------------


def test_freerouting_no_jar_returns_failure() -> None:
    """Returns a failure result when no JAR is found."""
    with patch("kicad_pipeline.routing.freerouting.find_freerouting_jar", return_value=None):
        result = route_with_freerouting("/tmp/test.dsn")
    assert result.success is False
    assert "not found" in result.error.lower()
    assert result.ses_file is None


def test_freerouting_timeout_returns_failure(tmp_path: Path) -> None:
    """TimeoutExpired should produce a failure result."""
    jar = tmp_path / "freerouting.jar"
    jar.write_bytes(b"fake")
    dsn = tmp_path / "test.dsn"
    dsn.write_text("(pcb)", encoding="utf-8")

    exc = subprocess.TimeoutExpired(cmd=["java"], timeout=1, output="", stderr="")
    with patch("subprocess.run", side_effect=exc):
        result = route_with_freerouting(str(dsn), jar_path=str(jar), timeout_seconds=1)
    assert result.success is False
    assert "timed out" in result.error.lower()


def test_freerouting_success_path(tmp_path: Path) -> None:
    """When subprocess succeeds and ses file exists, result is successful."""
    jar = tmp_path / "freerouting.jar"
    jar.write_bytes(b"fake")
    dsn = tmp_path / "test.dsn"
    dsn.write_text("(pcb)", encoding="utf-8")
    ses = tmp_path / "test.ses"
    ses.write_text("(session)", encoding="utf-8")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "done"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = route_with_freerouting(str(dsn), jar_path=str(jar))

    assert result.success is True
    assert result.ses_file == str(ses)
    assert result.stdout == "done"


def test_freerouting_nonzero_exit_is_failure(tmp_path: Path) -> None:
    """Non-zero exit code should produce a failure result."""
    jar = tmp_path / "freerouting.jar"
    jar.write_bytes(b"fake")
    dsn = tmp_path / "test.dsn"
    dsn.write_text("(pcb)", encoding="utf-8")

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "error"

    with patch("subprocess.run", return_value=mock_result):
        result = route_with_freerouting(str(dsn), jar_path=str(jar))

    assert result.success is False


def test_freerouting_java_not_found(tmp_path: Path) -> None:
    """FileNotFoundError (java missing) produces a failure result."""
    jar = tmp_path / "freerouting.jar"
    jar.write_bytes(b"fake")
    dsn = tmp_path / "test.dsn"
    dsn.write_text("(pcb)", encoding="utf-8")

    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = route_with_freerouting(str(dsn), jar_path=str(jar))

    assert result.success is False
    assert "java" in result.error.lower()


# ---------------------------------------------------------------------------
# ses_to_tracks
# ---------------------------------------------------------------------------

_SIMPLE_SES = """\
(session "test.ses"
  (base_design "test.dsn")
  (routes
    (resolution mm 1000)
    (network_out
      (net "GND"
        (wire (path "F.Cu" 0.25 10.0 20.0 15.0 20.0)
        ))
      )))
"""

_MULTI_NET_SES = """\
(session "test.ses"
  (base_design "test.dsn")
  (routes
    (network_out
      (net "GND"
        (wire (path "F.Cu" 0.5 0.0 0.0 5.0 0.0)
        ))
      (net "VCC"
        (wire (path "B.Cu" 0.25 1.0 1.0 6.0 1.0)
        ))
      )))
"""

_MULTI_SEGMENT_SES = """\
(session "test.ses"
  (routes
    (network_out
      (net "GND"
        (wire (path "F.Cu" 0.25 0 0 5 0 5 5)
        ))
      )))
"""


def test_ses_to_tracks_empty_content() -> None:
    """Empty or minimal SES content should produce an empty tuple."""
    pcb = _make_pcb()
    result = ses_to_tracks("", pcb)
    assert result == ()


def test_ses_to_tracks_minimal_ses() -> None:
    pcb = _make_pcb()
    result = ses_to_tracks("(session)", pcb)
    assert result == ()


def test_ses_to_tracks_parses_wire_path() -> None:
    """A wire path in SES should produce Track objects."""
    pcb = _make_pcb()
    tracks = ses_to_tracks(_SIMPLE_SES, pcb)
    assert len(tracks) >= 1


def test_ses_to_tracks_track_fields() -> None:
    """Parsed tracks have correct layer and width."""
    pcb = _make_pcb()
    tracks = ses_to_tracks(_SIMPLE_SES, pcb)
    assert len(tracks) >= 1
    t = tracks[0]
    assert t.layer == "F.Cu"
    assert abs(t.width - 0.25) < 1e-9


def test_ses_to_tracks_net_lookup() -> None:
    """Net name 'GND' should map to net_number=1 via pcb.nets."""
    pcb = _make_pcb()
    tracks = ses_to_tracks(_SIMPLE_SES, pcb)
    assert len(tracks) >= 1
    assert tracks[0].net_number == 1


def test_ses_to_tracks_unknown_net() -> None:
    """Unknown net names should get net_number=0."""
    ses = """\
(session "test.ses"
  (routes
    (network_out
      (net "UNKNOWN_NET_XYZ"
        (wire (path "F.Cu" 0.25 0 0 10 0)
        ))
      )))
"""
    pcb = _make_pcb()
    tracks = ses_to_tracks(ses, pcb)
    assert len(tracks) >= 1
    assert tracks[0].net_number == 0


def test_ses_to_tracks_coordinate_values() -> None:
    """Parsed track start/end coordinates should match SES wire path values."""
    pcb = _make_pcb()
    tracks = ses_to_tracks(_SIMPLE_SES, pcb)
    assert len(tracks) >= 1
    t = tracks[0]
    assert abs(t.start.x - 10.0) < 1e-6
    assert abs(t.start.y - 20.0) < 1e-6
    assert abs(t.end.x - 15.0) < 1e-6
    assert abs(t.end.y - 20.0) < 1e-6


def test_ses_to_tracks_multi_segment() -> None:
    """A path with 3 points (0,0 5,0 5,5) should produce 2 track segments."""
    pcb = _make_pcb()
    tracks = ses_to_tracks(_MULTI_SEGMENT_SES, pcb)
    assert len(tracks) == 2

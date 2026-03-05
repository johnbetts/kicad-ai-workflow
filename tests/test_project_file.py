"""Tests for kicad_pipeline.project_file."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import NetClass
from kicad_pipeline.project_file import build_project_file, write_project_file

if TYPE_CHECKING:
    from pathlib import Path


def test_build_project_file_defaults() -> None:
    """Default project file should have valid structure."""
    data = build_project_file("test_project")
    assert data["meta"]["filename"] == "test_project.kicad_pro"
    assert data["net_settings"]["classes"][0]["name"] == "Default"


def test_via_drill_meets_minimum() -> None:
    """All via drill values should be >= 0.508mm (min_through_hole_diameter)."""
    data = build_project_file("test")
    via_dims = data["board"]["design_settings"]["via_dimensions"]
    for via in via_dims:
        if via["drill"] > 0:
            assert via["drill"] >= 0.508, f"Via drill {via['drill']} < 0.508mm"

    default_class = data["net_settings"]["classes"][0]
    assert default_class["via_drill"] >= 0.508


def test_build_project_file_with_netclasses() -> None:
    """Netclasses should be serialised into net_settings."""
    classes = (
        NetClass(name="Default", trace_width_mm=0.25, clearance_mm=0.2),
        NetClass(
            name="Power",
            trace_width_mm=0.5,
            clearance_mm=0.3,
            via_diameter_mm=0.8,
            via_drill_mm=0.508,
            nets=("GND", "+3V3"),
        ),
        NetClass(
            name="HighVoltageAnalog",
            trace_width_mm=0.4,
            clearance_mm=0.5,
            nets=("SENS_IN", "AIN0"),
        ),
    )
    data = build_project_file("test", netclasses=classes)
    net_classes = data["net_settings"]["classes"]

    # Default + 2 custom
    assert len(net_classes) == 3
    names = [c["name"] for c in net_classes]
    assert "Default" in names
    assert "Power" in names
    assert "HighVoltageAnalog" in names

    # Check Power class values
    power = next(c for c in net_classes if c["name"] == "Power")
    assert power["track_width"] == 0.5
    assert power["clearance"] == 0.3

    # Check netclass assignments
    assignments = data["net_settings"]["netclass_assignments"]
    assert assignments is not None
    assert assignments["GND"] == "Power"
    assert assignments["+3V3"] == "Power"
    assert assignments["SENS_IN"] == "HighVoltageAnalog"


def test_build_project_file_without_netclasses() -> None:
    """Without netclasses, assignments should remain None."""
    data = build_project_file("test")
    assert data["net_settings"]["netclass_assignments"] is None


def test_write_project_file(tmp_path: Path) -> None:
    """write_project_file should create a valid .kicad_pro JSON file."""
    path = write_project_file("myproj", tmp_path)
    assert path.exists()
    assert path.suffix == ".kicad_pro"
    import json

    content = json.loads(path.read_text())
    assert content["meta"]["filename"] == "myproj.kicad_pro"


def test_write_project_file_with_netclasses(tmp_path: Path) -> None:
    """write_project_file should pass netclasses through."""
    classes = (
        NetClass(name="Power", trace_width_mm=0.5, nets=("GND",)),
    )
    path = write_project_file("myproj", tmp_path, netclasses=classes)
    import json

    content = json.loads(path.read_text())
    net_classes = content["net_settings"]["classes"]
    assert len(net_classes) == 2  # Default + Power


# ---------------------------------------------------------------------------
# Fix 7: Solder mask and edge clearance settings
# ---------------------------------------------------------------------------


def test_solder_mask_clearance_nonzero() -> None:
    """Solder mask clearance should be set to JLCPCB default (0.05mm)."""
    data = build_project_file("test")
    rules = data["board"]["design_settings"]["rules"]
    assert rules["solder_mask_clearance"] == 0.05


def test_solder_mask_min_width_nonzero() -> None:
    """Solder mask min width should be set to 0.1mm."""
    data = build_project_file("test")
    rules = data["board"]["design_settings"]["rules"]
    assert rules["solder_mask_min_width"] == 0.1


def test_solder_mask_to_copper_clearance() -> None:
    """Solder mask to copper clearance should be 0.05mm."""
    data = build_project_file("test")
    rules = data["board"]["design_settings"]["rules"]
    assert rules["solder_mask_to_copper_clearance"] == 0.05


def test_copper_edge_clearance_matches_jlcpcb() -> None:
    """min_copper_edge_clearance should be 0.3mm (JLCPCB standard)."""
    data = build_project_file("test")
    rules = data["board"]["design_settings"]["rules"]
    assert rules["min_copper_edge_clearance"] == 0.3

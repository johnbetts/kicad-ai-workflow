"""Tests for kicad_pipeline.routing.dsn_export."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.routing.dsn_export import pcb_to_dsn, write_dsn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pad(number: str, net_number: int = 1, net_name: str = "GND") -> Pad:
    return Pad(
        number=number,
        pad_type="smd",
        shape="rect",
        position=Point(x=0.0, y=0.0),
        size_x=1.5,
        size_y=1.5,
        layers=("F.Cu",),
        net_number=net_number,
        net_name=net_name,
    )


def _make_footprint(ref: str, x: float = 10.0, y: float = 10.0) -> Footprint:
    return Footprint(
        lib_id="R_SMD:R_0805",
        ref=ref,
        value="10k",
        position=Point(x=x, y=y),
        rotation=0.0,
        layer="F.Cu",
        pads=(_make_pad("1"), _make_pad("2", net_number=2, net_name="VCC")),
    )


def _make_outline(width: float = 50.0, height: float = 30.0) -> BoardOutline:
    return BoardOutline(
        polygon=(
            Point(0.0, 0.0),
            Point(width, 0.0),
            Point(width, height),
            Point(0.0, height),
        )
    )


def _make_pcb(
    footprints: tuple[Footprint, ...] = (),
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
        footprints=footprints,
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pcb_to_dsn_returns_string() -> None:
    pcb = _make_pcb()
    result = pcb_to_dsn(pcb)
    assert isinstance(result, str)


def test_dsn_contains_pcb_keyword() -> None:
    pcb = _make_pcb()
    result = pcb_to_dsn(pcb)
    assert "(pcb" in result


def test_dsn_contains_structure() -> None:
    pcb = _make_pcb()
    result = pcb_to_dsn(pcb)
    assert "(structure" in result


def test_dsn_contains_network() -> None:
    pcb = _make_pcb(footprints=(_make_footprint("R1"),))
    result = pcb_to_dsn(pcb)
    assert "(network" in result


def test_dsn_contains_placement() -> None:
    pcb = _make_pcb(footprints=(_make_footprint("R1"),))
    result = pcb_to_dsn(pcb)
    assert "(placement" in result


def test_dsn_contains_wiring() -> None:
    pcb = _make_pcb()
    result = pcb_to_dsn(pcb)
    assert "(wiring" in result


def test_dsn_contains_boundary() -> None:
    pcb = _make_pcb()
    result = pcb_to_dsn(pcb)
    assert "(boundary" in result


def test_dsn_net_names() -> None:
    """Net names from pcb.nets should appear in the DSN string."""
    pcb = _make_pcb(footprints=(_make_footprint("R1"),))
    result = pcb_to_dsn(pcb)
    assert "GND" in result
    assert "VCC" in result


def test_dsn_footprint_refs() -> None:
    """Footprint reference designators should appear in the placement section."""
    fp1 = _make_footprint("R1", x=10.0, y=10.0)
    fp2 = _make_footprint("C1", x=20.0, y=10.0)
    pcb = _make_pcb(footprints=(fp1, fp2))
    result = pcb_to_dsn(pcb)
    assert '"R1"' in result
    assert '"C1"' in result


def test_write_dsn(tmp_path: Path) -> None:
    """write_dsn should create a file whose content matches pcb_to_dsn."""
    fp1 = _make_footprint("R1")
    pcb = _make_pcb(footprints=(fp1,))
    out_file = tmp_path / "test.dsn"
    write_dsn(pcb, out_file)
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == pcb_to_dsn(pcb)


def test_dsn_layer_names() -> None:
    """Both 'F.Cu' and 'B.Cu' should appear in the DSN output."""
    pcb = _make_pcb()
    result = pcb_to_dsn(pcb)
    assert "F.Cu" in result
    assert "B.Cu" in result


def test_dsn_empty_footprints() -> None:
    """A PCB with no footprints should still produce a valid DSN string."""
    pcb = _make_pcb(footprints=())
    result = pcb_to_dsn(pcb)
    assert isinstance(result, str)
    assert "(pcb" in result
    assert "(structure" in result
    assert "(wiring" in result


def test_dsn_resolution_and_unit() -> None:
    """DSN output should declare resolution and unit."""
    pcb = _make_pcb()
    result = pcb_to_dsn(pcb)
    assert "(resolution mm" in result
    assert "(unit mm)" in result


def test_dsn_via_template() -> None:
    """A via template should be present in the structure section."""
    pcb = _make_pcb()
    result = pcb_to_dsn(pcb)
    assert "(via" in result

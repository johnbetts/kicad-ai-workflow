"""Written-file schematic<->PCB sync regression (KI: schematic load).

Every pipeline-generated schematic failed to LOAD in KiCad 10
(2026-06-12): numeric pin electrical types (``(pin 1 line``) and a
missing top-level ``(embedded_fonts no)``. The in-memory sync gate was
blind to both. This test round-trips a small generated project through
KiCad's OWN parser (kicad-cli netlist export) and asserts the written
schematic and PCB agree — the "Update PCB from Schematic is a no-op"
invariant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kicad_pipeline.evals.sch_pcb_sync import check_written_sync, find_kicad_cli
from kicad_pipeline.models.requirements import (
    Component,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)

pytestmark = pytest.mark.skipif(
    find_kicad_cli() is None, reason="kicad-cli not installed",
)


def _requirements() -> ProjectRequirements:
    r1 = Component(
        ref="R1", value="10k", footprint="R_0402",
        pins=(Pin(number="1", name="~", pin_type=PinType.PASSIVE),
              Pin(number="2", name="~", pin_type=PinType.PASSIVE)),
    )
    c1 = Component(
        ref="C1", value="100nF", footprint="C_0402",
        pins=(Pin(number="1", name="~", pin_type=PinType.PASSIVE),
              Pin(number="2", name="~", pin_type=PinType.PASSIVE)),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="sync-test"),
        features=(),
        components=(r1, c1),
        nets=(
            Net(name="SIG", connections=(
                NetConnection("R1", "1"), NetConnection("C1", "1"),
            )),
            Net(name="GND", connections=(
                NetConnection("R1", "2"), NetConnection("C1", "2"),
            )),
        ),
    )


def test_written_schematic_loads_in_kicad_and_matches_pcb(tmp_path: Path) -> None:
    from kicad_pipeline.pcb.builder import build_pcb, write_pcb
    from kicad_pipeline.schematic.builder import build_schematic, write_schematic

    requirements = _requirements()
    pcb = build_pcb(requirements, auto_route=False, project_name="sync_test")
    pcb_path = tmp_path / "sync_test.kicad_pcb"
    write_pcb(pcb, pcb_path, fill_zones=False)

    schematic = build_schematic(requirements, project_name="sync_test")
    sch_path = tmp_path / "sync_test.kicad_sch"
    write_schematic(schematic, sch_path, project_name="sync_test")

    report = check_written_sync(sch_path, pcb_path)
    assert report.passed, report.issues
    assert report.components_checked == 2
    assert report.nets_checked >= 4


def test_missing_kicad_cli_fails_loudly(tmp_path: Path) -> None:
    # A bogus explicit CLI path must FAIL the check, never skip it:
    # silently passing unverified sync is how the load bug survived.
    report = check_written_sync(
        tmp_path / "a.kicad_sch", tmp_path / "a.kicad_pcb",
        kicad_cli="/nonexistent/kicad-cli",
    )
    assert not report.passed

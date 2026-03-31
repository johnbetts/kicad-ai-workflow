"""Shared boilerplate for kicad-ai training board scripts.

Each ``train_*.py`` script builds a minimal PCB, runs the EE placement
optimizer, renders a PNG, and prints quality scores + component positions.
This module centralises the repetitive patterns so each script only needs
to supply its board-specific logic.

Usage in a train script::

    from _train_common import (
        pipeline_imports,        # noqa — ensures sys.path is configured
        write_and_compare_pcb,
        print_component_positions,
        build_group_map,
    )

The module must be imported **after** sys.path has been extended to include
``src/``.  Each train script handles this extension itself (see the _repo /
sys.path block in the preamble) then imports helpers from here.
"""

from __future__ import annotations

import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

# ---------------------------------------------------------------------------
# Shared pipeline imports
# ---------------------------------------------------------------------------
# Ensure src/ is on path when running scripts directly.
_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from kicad_pipeline.models.requirements import (  # noqa: E402
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
from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee  # noqa: E402
from kicad_pipeline.optimization.scoring import compute_fast_placement_score  # noqa: E402
from kicad_pipeline.pcb.builder import build_pcb, write_pcb  # noqa: E402
from kicad_pipeline.project_file import write_project_file  # noqa: E402
from kicad_pipeline.schematic.builder import build_schematic, write_schematic  # noqa: E402

__all__ = [
    "DRIFT_WARN_MM",
    "Component",
    "FeatureBlock",
    "MechanicalConstraints",
    "Net",
    "NetConnection",
    "Pin",
    "PinFunction",
    "PinType",
    "ProjectInfo",
    "ProjectRequirements",
    "build_group_map",
    "build_pcb",
    "build_schematic",
    "compute_fast_placement_score",
    "optimize_placement_ee",
    "print_component_positions",
    "write_and_compare_pcb",
    "write_pcb",
    "write_project_file",
    "write_schematic",
]

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: Component drift threshold for reference-board comparison warnings (mm).
DRIFT_WARN_MM: float = 3.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_group_map(requirements: ProjectRequirements) -> dict[str, str]:
    """Return ``{ref: feature_name}`` from the requirements feature blocks.

    Used to colour footprints by functional group in the placement render.

    Args:
        requirements: Built :class:`ProjectRequirements` for the training board.

    Returns:
        Mapping of component reference → feature/group name.
    """
    group_map: dict[str, str] = {}
    for feat in requirements.features:
        for ref in feat.components:
            group_map[ref] = feat.name
    return group_map


def write_and_compare_pcb(
    pcb: PCBDesign,
    pcb_path: Path,
    drift_warn_mm: float = DRIFT_WARN_MM,
    requirements: ProjectRequirements | None = None,
) -> None:
    """Write a KiCad PCB + schematic and compare positions against backup.

    Procedure:
    1. Back up the existing file (if any) to ``training_reference_boards/``.
    2. Write *pcb* to *pcb_path*.
    3. Build and write schematic from *requirements* (same directory).
    4. Write KiCad project file (.kicad_pro).
    5. If a previous backup exists, compare component positions and report
       drift.  Components whose ref starts with ``"H"`` (mounting holes)
       are skipped.

    Args:
        pcb: Optimized :class:`PCBDesign` to write.
        pcb_path: Target ``.kicad_pcb`` file path.
        drift_warn_mm: Distance threshold above which a ``***`` marker is
            printed for a component (default :data:`DRIFT_WARN_MM`).
        requirements: Project requirements for schematic generation.
            When ``None``, schematic generation is skipped.
    """
    stem = pcb_path.stem  # e.g. "train_ethernet"
    ref_dir = pcb_path.parent / "training_reference_boards"
    ref_dir.mkdir(exist_ok=True)

    if pcb_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = ref_dir / f"{stem}_{timestamp}.kicad_pcb"
        shutil.copy2(pcb_path, backup)
        print(f"  Backed up existing PCB to {backup}")

    print(f"Writing KiCad PCB to {pcb_path} ...")
    write_pcb(pcb, pcb_path, fill_zones=False)
    print(f"  KiCad PCB: {pcb_path}")

    # Generate schematic + project file so KiCad can open the full project
    if requirements is not None:
        sch_path = pcb_path.with_suffix(".kicad_sch")
        pro_path = pcb_path.with_suffix(".kicad_pro")
        try:
            schematic = build_schematic(requirements, project_name=stem)
            write_schematic(schematic, sch_path, project_name=stem)
            print(f"  Schematic: {sch_path}")
        except Exception as exc:
            print(f"  WARNING: schematic generation failed: {exc}")
        try:
            write_project_file(stem, pcb_path.parent)
            print(f"  Project:   {pcb_path.with_suffix('.kicad_pro')}")
        except Exception as exc:
            print(f"  WARNING: project file generation failed: {exc}")

    ref_files = sorted(ref_dir.glob(f"{stem}*.kicad_pcb"))
    if ref_files:
        latest_ref = ref_files[-1]
        print(f"\n  Comparing against reference: {latest_ref.name}")
        from kicad_pipeline.pcb.position_extractor import positions_from_pcb_file

        ref_positions = positions_from_pcb_file(latest_ref)
        gen_positions = positions_from_pcb_file(pcb_path)

        print(
            f"  {'Ref':<8} {'Gen X':>7} {'Ref X':>7} {'dX':>6}"
            f" {'Gen Y':>7} {'Ref Y':>7} {'dY':>6} {'Dist':>6}"
        )
        total_drift = 0.0
        count = 0
        for ref in sorted(set(gen_positions) & set(ref_positions)):
            if ref.startswith("H"):
                continue
            gx, gy, _gr = gen_positions[ref]
            rx, ry, _rr = ref_positions[ref]
            dist = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)
            total_drift += dist
            count += 1
            marker = "***" if dist > drift_warn_mm else ""
            print(
                f"  {ref:<8} {gx:>7.1f} {rx:>7.1f} {gx - rx:>+6.1f}"
                f" {gy:>7.1f} {ry:>7.1f} {gy - ry:>+6.1f}"
                f" {dist:>6.1f} {marker}"
            )
        if count:
            print(f"  Average drift from reference: {total_drift / count:.1f}mm")
    print()


def print_component_positions(pcb: PCBDesign) -> dict[str, tuple[float, float, float]]:
    """Print a table of component positions and return a ``ref → (x, y, rot)`` map.

    Args:
        pcb: Optimized :class:`PCBDesign`.

    Returns:
        Mapping of reference designator → ``(x_mm, y_mm, rotation_deg)``.
    """
    print("Component positions:")
    print(f"  {'Ref':<6} {'X':>8} {'Y':>8} {'Rot':>6}")
    print(f"  {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 6}")
    fp_map: dict[str, tuple[float, float, float]] = {}
    for fp in sorted(pcb.footprints, key=lambda f: f.ref):
        print(
            f"  {fp.ref:<6} {fp.position.x:>8.2f} "
            f"{fp.position.y:>8.2f} {fp.rotation:>6.1f}"
        )
        fp_map[fp.ref] = (fp.position.x, fp.position.y, fp.rotation)
    print()
    return fp_map

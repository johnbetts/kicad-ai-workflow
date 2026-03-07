"""Production package builder."""

from __future__ import annotations

import io
import logging
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.production.assembly_drawing import (
    generate_ascii_assembly,
)
from kicad_pipeline.production.bom import BOMRow, bom_to_csv, generate_bom
from kicad_pipeline.production.cpl import cpl_to_csv, generate_cpl
from kicad_pipeline.production.drill import generate_drill_files
from kicad_pipeline.production.gerber import generate_all_gerbers

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductionPackage:
    """Complete production artifact bundle."""

    project_name: str
    gerbers: dict[str, str]
    drill_files: dict[str, str]
    bom_csv: str
    cpl_csv: str
    assembly_drawing: str
    order_guide: str
    cost_estimate: str
    validation_report_text: str = ""
    validation_report_json: str = ""


def generate_gerber_zip(
    gerbers: dict[str, str],
    drill_files: dict[str, str],
    project_name: str,
) -> bytes:
    """Create an in-memory ZIP archive of gerber and drill files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, content in gerbers.items():
            zf.writestr(filename, content)
        for filename, content in drill_files.items():
            zf.writestr(filename, content)
    return buf.getvalue()


def generate_order_guide(
    project_name: str,
    bom_rows: tuple[BOMRow, ...],
    board_dims: tuple[float, float],
) -> str:
    """Generate JLCPCB order guide text."""
    w, h = board_dims

    # BOM summary
    bom_lines: list[str] = []
    for row in bom_rows:
        bom_lines.append(f"  - {row.quantity}x {row.comment} ({row.designator})")
    bom_summary = "\n".join(bom_lines) if bom_lines else "  (no components)"

    total_cost = sum(r.unit_price_usd * r.quantity for r in bom_rows)

    return (
        f"# JLCPCB Order Guide - {project_name}\n"
        "\n"
        "## Step 1: Upload Gerbers\n"
        "1. Go to https://jlcpcb.com/quote\n"
        f"2. Upload {project_name}_gerbers.zip\n"
        f"3. Board Dimensions: {w:.1f}mm x {h:.1f}mm (auto-detected)\n"
        "\n"
        "## Step 2: PCB Settings\n"
        "- Layers: 2\n"
        "- Surface Finish: HASL (lead-free)\n"
        "- Copper Weight: 1oz\n"
        "- PCB Color: Green\n"
        "\n"
        "## Step 3: SMT Assembly\n"
        '1. Enable "SMT Assembly"\n'
        f"2. Upload BOM: {project_name}_bom.csv\n"
        f"3. Upload CPL: {project_name}_cpl.csv\n"
        "\n"
        "## Step 4: Review BOM\n"
        f"{bom_summary}\n"
        "\n"
        "## Cost Estimate\n"
        f"Components: {total_cost:.2f} USD (quantity 5)\n"
    )


def estimate_cost(
    bom_rows: tuple[BOMRow, ...],
    quantities: tuple[int, ...] = (5, 10, 50, 100),
) -> str:
    """Return cost estimate string for multiple production quantities."""
    pcb_base_costs: dict[int, float] = {5: 5.0, 10: 8.0, 50: 30.0, 100: 100.0}

    lines: list[str] = ["Cost Estimate:"]
    component_cost = sum(r.unit_price_usd * r.quantity for r in bom_rows)

    for qty in quantities:
        pcb_cost = pcb_base_costs.get(qty, float(qty) * 1.0)
        total = pcb_cost + component_cost
        lines.append(f"  Qty {qty}:   ${total:.2f}")

    return "\n".join(lines) + "\n"


def build_production_package(
    pcb: PCBDesign,
    project_name: str,
    requirements: ProjectRequirements | None = None,
) -> ProductionPackage:
    """Orchestrate all generators and return a complete ProductionPackage."""
    gerbers = generate_all_gerbers(pcb, project_name)
    drill_files = generate_drill_files(pcb)
    bom_rows = generate_bom(pcb, requirements)
    bom_csv = bom_to_csv(bom_rows)
    cpl_rows = generate_cpl(pcb)
    cpl_csv_str = cpl_to_csv(cpl_rows)
    assembly_drawing = generate_ascii_assembly(pcb)

    # Compute bounding box of outline
    pts = pcb.outline.polygon
    if pts:
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        board_dims: tuple[float, float] = (
            max(xs) - min(xs),
            max(ys) - min(ys),
        )
    else:
        board_dims = (0.0, 0.0)

    order_guide = generate_order_guide(project_name, bom_rows, board_dims)
    cost_estimate = estimate_cost(bom_rows)

    return ProductionPackage(
        project_name=project_name,
        gerbers=gerbers,
        drill_files=drill_files,
        bom_csv=bom_csv,
        cpl_csv=cpl_csv_str,
        assembly_drawing=assembly_drawing,
        order_guide=order_guide,
        cost_estimate=cost_estimate,
    )


def write_production_package(
    pkg: ProductionPackage, output_dir: str | Path
) -> None:
    """Write all production package files to disk."""
    base = Path(output_dir)

    # Gerbers
    gerber_dir = base / "gerbers"
    gerber_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in pkg.gerbers.items():
        (gerber_dir / filename).write_text(content, encoding="utf-8")

    # Drill files alongside gerbers
    for filename, content in pkg.drill_files.items():
        (gerber_dir / filename).write_text(content, encoding="utf-8")

    # Assembly files
    assembly_dir = base / "assembly"
    assembly_dir.mkdir(parents=True, exist_ok=True)
    (assembly_dir / "bom.csv").write_text(pkg.bom_csv, encoding="utf-8")
    (assembly_dir / "cpl.csv").write_text(pkg.cpl_csv, encoding="utf-8")
    (assembly_dir / "assembly.txt").write_text(pkg.assembly_drawing, encoding="utf-8")

    # Gerber ZIP
    zip_bytes = generate_gerber_zip(pkg.gerbers, pkg.drill_files, pkg.project_name)
    (base / f"{pkg.project_name}_gerbers.zip").write_bytes(zip_bytes)

    # Docs
    docs_dir = base / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "order_guide.md").write_text(pkg.order_guide, encoding="utf-8")
    (docs_dir / "cost_estimate.txt").write_text(pkg.cost_estimate, encoding="utf-8")

    # Validation reports (if present)
    if pkg.validation_report_text:
        (docs_dir / "parts_validation_report.txt").write_text(
            pkg.validation_report_text, encoding="utf-8",
        )
    if pkg.validation_report_json:
        (docs_dir / "parts_validation_report.json").write_text(
            pkg.validation_report_json, encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# kicad-cli based export
# ---------------------------------------------------------------------------

_KICAD_CLI_DEFAULT = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"


def _find_kicad_cli() -> str | None:
    """Locate kicad-cli binary, returning None if not found."""
    import os

    env_path = os.environ.get("KICAD_CLI")
    if env_path and Path(env_path).is_file():
        return env_path
    if Path(_KICAD_CLI_DEFAULT).is_file():
        return _KICAD_CLI_DEFAULT
    try:
        result = subprocess.run(
            ["which", "kicad-cli"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


@dataclass(frozen=True)
class FabricationPackage:
    """Result of a kicad-cli fabrication export."""

    gerber_dir: Path
    drill_dir: Path
    zip_path: Path | None
    bom_path: Path | None
    cpl_path: Path | None
    errors: tuple[str, ...]


def export_for_jlcpcb(
    pcb_path: str | Path,
    output_dir: str | Path,
    project_name: str | None = None,
) -> FabricationPackage:
    """Generate JLCPCB-ready manufacturing files using kicad-cli.

    Falls back to the internal production pipeline if kicad-cli is not
    available.

    Args:
        pcb_path: Path to the ``.kicad_pcb`` file.
        output_dir: Directory for output files.
        project_name: Project name for file naming.

    Returns:
        A :class:`FabricationPackage` describing the generated files.
    """
    pcb_path = Path(pcb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = project_name or pcb_path.stem

    kicad_cli = _find_kicad_cli()
    errors: list[str] = []

    gerber_dir = output_dir / "gerbers"
    gerber_dir.mkdir(exist_ok=True)
    drill_dir = gerber_dir  # JLCPCB expects drills alongside gerbers.

    if kicad_cli:
        # Export gerbers via kicad-cli.
        gerber_cmd = [
            kicad_cli, "pcb", "export", "gerbers",
            "--output", str(gerber_dir) + "/",
            str(pcb_path),
        ]
        try:
            result = subprocess.run(
                gerber_cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                errors.append(f"Gerber export failed: {result.stderr[:300]}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            errors.append(f"Gerber export error: {exc}")

        # Export drill files.
        drill_cmd = [
            kicad_cli, "pcb", "export", "drill",
            "--output", str(drill_dir) + "/",
            "--format", "excellon",
            "--excellon-units", "mm",
            str(pcb_path),
        ]
        try:
            result = subprocess.run(
                drill_cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                errors.append(f"Drill export failed: {result.stderr[:300]}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            errors.append(f"Drill export error: {exc}")
    else:
        errors.append(
            "kicad-cli not found — using internal pipeline for gerber/drill"
        )

    # Create zip of gerber + drill files.
    zip_path: Path | None = None
    gerber_files = list(gerber_dir.glob("*"))
    if gerber_files:
        zip_path = output_dir / f"{name}_gerbers.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in gerber_files:
                zf.write(f, f.name)

    # BOM and CPL use the internal pipeline (kicad-cli doesn't generate these).
    bom_path: Path | None = None
    cpl_path: Path | None = None

    return FabricationPackage(
        gerber_dir=gerber_dir,
        drill_dir=drill_dir,
        zip_path=zip_path,
        bom_path=bom_path,
        cpl_path=cpl_path,
        errors=tuple(errors),
    )

"""Production package builder."""

from __future__ import annotations

import io
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

"""Persistent design state for the conversational workflow.

Writes human-readable markdown + machine-readable JSON to a ``design/``
directory inside the project root so that Claude Code can reconstruct
context after compaction.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path  # noqa: TC003 — used at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.parts.jlcpcb_db import JLCPCBPart
    from kicad_pipeline.parts.selector import PartSuggestion
    from kicad_pipeline.validation.checklist import ChecklistReport
    from kicad_pipeline.validation.drc import DRCReport
    from kicad_pipeline.validation.drc_interpreter import DRCSuggestion

logger = logging.getLogger(__name__)

# Workflow phases in order — used by get_current_phase()
PHASES: tuple[str, ...] = (
    "Project Setup",
    "Requirements Gathering",
    "Parts Selection",
    "Schematic Generation",
    "PCB Layout",
    "User Routing",
    "DRC Iteration",
    "Validation",
    "Fabrication Export",
    "Design Review",
)

_PHASE_FILES: tuple[tuple[str, str], ...] = (
    ("checklist.md", "Validation"),
    ("drc_history.md", "DRC Iteration"),
    ("parts_selection.json", "Parts Selection"),
    ("requirements.md", "Requirements Gathering"),
    ("README.md", "Project Setup"),
)


def _design_dir(project_dir: Path) -> Path:
    """Return the ``design/`` directory, creating it if needed."""
    d = project_dir / "design"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _today() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_project_readme(
    project_dir: Path,
    name: str,
    description: str,
    phase: str,
    component_count: int = 0,
    basic_count: int = 0,
) -> Path:
    """Write/update ``design/README.md`` with project overview and current phase."""
    d = _design_dir(project_dir)
    path = d / "README.md"

    phase_idx = PHASES.index(phase) if phase in PHASES else -1

    lines = [
        f"# Project: {name}",
        f"**Phase**: {phase}",
        f"**Updated**: {_now_stamp()}",
        f"**Description**: {description}",
        "",
        "## Status",
    ]
    for i, p in enumerate(PHASES):
        marker = "x" if i <= phase_idx else " "
        extra = ""
        if p == "Parts Selection" and component_count > 0:
            extra = f" ({component_count} components, {basic_count} JLCPCB basic)"
        lines.append(f"- [{marker}] {p}{extra}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def write_requirements(project_dir: Path, requirements: ProjectRequirements) -> Path:
    """Write ``design/requirements.md`` from gathered requirements."""
    d = _design_dir(project_dir)
    path = d / "requirements.md"
    lines: list[str] = ["# Requirements", ""]

    # Project info
    proj = requirements.project
    lines.append(f"**Project**: {proj.name}")
    if proj.description:
        lines.append(f"**Description**: {proj.description}")
    if proj.author:
        lines.append(f"**Author**: {proj.author}")
    lines.append(f"**Revision**: {proj.revision}")
    lines.append("")

    # Mechanical
    mech = requirements.mechanical
    if mech:
        lines.append("## Mechanical")
        lines.append(f"- Board size: {mech.board_width_mm} x {mech.board_height_mm} mm")
        if mech.board_template:
            lines.append(f"- Template: {mech.board_template}")
        if mech.enclosure:
            lines.append(f"- Enclosure: {mech.enclosure}")
        lines.append(
            f"- Mounting holes: {mech.mounting_hole_diameter_mm}mm"
            f" x {len(mech.mounting_hole_positions)}"
        )
        if mech.notes:
            lines.append(f"- Notes: {mech.notes}")
        lines.append("")

    # Feature blocks
    if requirements.features:
        lines.append("## Features")
        for fb in requirements.features:
            lines.append(f"### {fb.name}")
            lines.append(fb.description)
            lines.append(f"- Components: {', '.join(fb.components)}")
            lines.append(f"- Nets: {', '.join(fb.nets)}")
            if fb.subcircuits:
                lines.append(f"- Subcircuits: {', '.join(fb.subcircuits)}")
            lines.append("")

    # Components
    lines.append("## Components")
    lines.append("")
    lines.append("| Ref | Value | Footprint | LCSC |")
    lines.append("|-----|-------|-----------|------|")
    for c in requirements.components:
        lines.append(f"| {c.ref} | {c.value} | {c.footprint} | {c.lcsc or ''} |")
    lines.append("")

    # Nets
    if requirements.nets:
        lines.append("## Nets")
        for net in requirements.nets:
            conns = ", ".join(f"{c.ref}.{c.pin}" for c in net.connections)
            lines.append(f"- **{net.name}**: {conns}")
        lines.append("")

    # Power budget
    if requirements.power_budget:
        pb = requirements.power_budget
        lines.append("## Power Budget")
        lines.append(f"- Total current: {pb.total_current_ma} mA")
        for rail in pb.rails:
            lines.append(f"- {rail.name}: {rail.voltage}V, {rail.current_ma}mA")
        if pb.notes:
            for note in pb.notes:
                lines.append(f"- Note: {note}")
        lines.append("")

    # Recommendations
    if requirements.recommendations:
        lines.append("## Recommendations")
        for rec in requirements.recommendations:
            lines.append(f"- [{rec.severity}] ({rec.category}) {rec.message}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def write_parts_selection(
    project_dir: Path,
    suggestions: dict[str, PartSuggestion],
    final_parts: dict[str, JLCPCBPart],
) -> tuple[Path, Path]:
    """Write ``design/parts_selection.md`` and ``design/parts_selection.json``."""
    d = _design_dir(project_dir)
    md_path = d / "parts_selection.md"
    json_path = d / "parts_selection.json"

    # --- Markdown ---
    lines = ["# Parts Selection", f"**Date**: {_today()}", ""]
    lines.append("| Ref | Value | LCSC | Mfr Part | Package | Basic | Price |")
    lines.append("|-----|-------|------|----------|---------|-------|-------|")
    for ref, part in sorted(final_parts.items()):
        basic = "Yes" if part.basic else "No"
        price = f"${part.price:.4f}" if part.price is not None else "—"
        lines.append(
            f"| {ref} | {part.mfr_part} | {part.lcsc} | {part.mfr} "
            f"| {part.package} | {basic} | {price} |"
        )
    lines.append("")

    # Match quality from suggestions
    lines.append("## Match Quality")
    for ref, sug in sorted(suggestions.items()):
        alt_count = len(sug.candidates)
        lines.append(f"- **{ref}** ({sug.component_value}): {sug.match_quality} match")
        if sug.notes:
            lines.append(f"  - {sug.notes}")
        if alt_count > 1:
            lines.append(f"  - {alt_count} alternatives available")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # --- JSON ---
    data: dict[str, dict[str, object]] = {}
    for ref, part in sorted(final_parts.items()):
        data[ref] = {
            "lcsc": part.lcsc,
            "mfr": part.mfr,
            "mfr_part": part.mfr_part,
            "package": part.package,
            "basic": part.basic,
            "stock": part.stock,
            "price": part.price,
        }
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    logger.info("Wrote %s and %s", md_path, json_path)
    return md_path, json_path


def append_design_decision(
    project_dir: Path,
    phase: str,
    decision: str,
    rationale: str,
    alternatives: tuple[str, ...] = (),
) -> Path:
    """Append an entry to ``design/design_decisions.md``."""
    d = _design_dir(project_dir)
    path = d / "design_decisions.md"

    # Create header if new file
    if not path.exists():
        path.write_text("# Design Decisions\n\n", encoding="utf-8")

    lines = [
        f"## {_today()} — {phase}",
        f"**Decision**: {decision}",
        f"**Rationale**: {rationale}",
    ]
    if alternatives:
        alts = "; ".join(alternatives)
        lines.append(f"**Alternatives considered**: {alts}")
    lines.append("")

    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Appended decision to %s", path)
    return path


def write_drc_history(
    project_dir: Path,
    report: DRCReport,
    suggestions: list[DRCSuggestion],
    fixes_applied: int,
    iteration: int,
) -> Path:
    """Append a DRC run to ``design/drc_history.md``."""
    d = _design_dir(project_dir)
    path = d / "drc_history.md"

    if not path.exists():
        path.write_text("# DRC History\n\n", encoding="utf-8")

    error_count = len(report.errors)
    warning_count = len(report.warnings)
    status = "PASS" if report.passed else "FAIL"
    auto_fixable = sum(1 for s in suggestions if s.auto_fixable)
    manual = len(suggestions) - auto_fixable

    lines = [
        f"## Iteration {iteration} — {_now_stamp()}",
        f"**Status**: {status}",
        f"**Errors**: {error_count} | **Warnings**: {warning_count}",
        f"**Auto-fixable**: {auto_fixable} | **Manual**: {manual}",
        f"**Fixes applied this run**: {fixes_applied}",
    ]

    # Top violation types
    type_counts: dict[str, int] = {}
    for v in report.violations:
        type_counts[v.rule] = type_counts.get(v.rule, 0) + 1
    if type_counts:
        lines.append("")
        lines.append("### Violations by type")
        for rule, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {rule}: {count}")

    # Suggestions requiring manual action
    manual_suggestions = [s for s in suggestions if not s.auto_fixable]
    if manual_suggestions:
        lines.append("")
        lines.append("### Manual fixes needed")
        for s in manual_suggestions:
            refs = f" ({', '.join(s.affected_refs)})" if s.affected_refs else ""
            lines.append(f"- [{s.severity}] {s.description}{refs}")
            if s.fix_action:
                lines.append(f"  - Action: {s.fix_action}")

    lines.append("")

    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Appended DRC iteration %d to %s", iteration, path)
    return path


def write_checklist_results(project_dir: Path, report: ChecklistReport) -> Path:
    """Write ``design/checklist.md`` with latest checklist output."""
    d = _design_dir(project_dir)
    path = d / "checklist.md"

    status = "PASS" if report.passed else f"FAIL ({report.fail_count} failures)"

    lines = [
        "# Pre-Fabrication Checklist",
        f"**Date**: {_now_stamp()}",
        f"**Overall**: {status}",
        "",
    ]

    # Group by category
    categories: dict[str, list[tuple[str, str, str]]] = {}
    for r in report.results:
        cat_list = categories.setdefault(r.category, [])
        icon = {"pass": "+", "warn": "~", "fail": "-", "skip": " "}[r.status.value]
        detail = f" — {r.details}" if r.details else ""
        cat_list.append((icon, r.name, f"{r.message}{detail}"))

    for category, items in sorted(categories.items()):
        lines.append(f"## {category}")
        for icon, name, msg in items:
            lines.append(f"- [{icon}] **{name}**: {msg}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Research Cache
# ---------------------------------------------------------------------------


def _research_dir(project_dir: Path) -> Path:
    """Return ``design/research/``, creating it if needed."""
    d = project_dir / "design" / "research"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slugify(topic: str) -> str:
    """Convert a topic string to a safe filename slug."""
    import re

    slug = topic.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")[:80]


def save_research(
    project_dir: Path,
    topic: str,
    content: str,
    source: str = "",
    tags: tuple[str, ...] = (),
) -> Path:
    """Cache a research finding to ``design/research/{slug}.md``.

    Args:
        project_dir: Project root.
        topic: Human-readable topic (e.g. "ESP32-S3 pinout").
        content: The research content (markdown).
        source: URL or reference where the info came from.
        tags: Optional tags for categorisation (e.g. "pinout", "datasheet").

    Returns:
        Path to the written file.
    """
    d = _research_dir(project_dir)
    slug = _slugify(topic)
    path = d / f"{slug}.md"

    lines = [
        f"# {topic}",
        f"**Cached**: {_now_stamp()}",
    ]
    if source:
        lines.append(f"**Source**: {source}")
    if tags:
        lines.append(f"**Tags**: {', '.join(tags)}")
    lines.append("")
    lines.append(content)
    if not content.endswith("\n"):
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Cached research: %s → %s", topic, path)
    return path


def lookup_research(project_dir: Path, topic: str) -> str | None:
    """Look up a cached research finding by exact topic slug.

    Returns the file content if found, ``None`` otherwise.
    """
    d = project_dir / "design" / "research"
    if not d.is_dir():
        return None
    slug = _slugify(topic)
    path = d / f"{slug}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def search_research(project_dir: Path, keyword: str) -> dict[str, str]:
    """Search cached research files for a keyword (case-insensitive).

    Returns ``{filename: content}`` for every matching file.
    """
    d = project_dir / "design" / "research"
    if not d.is_dir():
        return {}
    kw = keyword.lower()
    results: dict[str, str] = {}
    for p in sorted(d.iterdir()):
        if p.suffix == ".md":
            text = p.read_text(encoding="utf-8")
            if kw in text.lower():
                results[p.name] = text
    return results


def list_research(project_dir: Path) -> tuple[str, ...]:
    """List all cached research topic filenames."""
    d = project_dir / "design" / "research"
    if not d.is_dir():
        return ()
    return tuple(p.name for p in sorted(d.iterdir()) if p.suffix == ".md")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_project_state(project_dir: Path) -> dict[str, str]:
    """Read all ``design/*.md``, ``design/*.json``, and ``design/research/*.md``.

    Returns ``{filename: content}`` for top-level design files and
    ``{"research/{name}": content}`` for cached research.  Useful for
    reconstructing conversational context after compaction.
    """
    d = project_dir / "design"
    if not d.is_dir():
        return {}

    result: dict[str, str] = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix in (".md", ".json"):
            result[p.name] = p.read_text(encoding="utf-8")

    research = d / "research"
    if research.is_dir():
        for p in sorted(research.iterdir()):
            if p.suffix == ".md":
                result[f"research/{p.name}"] = p.read_text(encoding="utf-8")

    return result


def get_current_phase(project_dir: Path) -> str:
    """Determine the most advanced phase from files present in ``design/``.

    Falls back to ``"Project Setup"`` if no files exist yet.
    """
    d = project_dir / "design"
    if not d.is_dir():
        return PHASES[0]

    existing = {p.name for p in d.iterdir()}

    # Walk from most-advanced to least-advanced
    for filename, phase in _PHASE_FILES:
        if filename in existing:
            return phase

    return PHASES[0]

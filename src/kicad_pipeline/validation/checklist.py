"""Pre-fabrication validation checklist.

Comprehensive checks against JLCPCB manufacturing capabilities,
electrical correctness, safety standards, and mechanical requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    JLCPCB_MAX_BOARD_SIZE_MM,
    JLCPCB_MIN_BOARD_SIZE_MM,
    JLCPCB_MIN_TRACE_MM,
    JLCPCB_MIN_VIA_ANNULAR_RING_MM,
    JLCPCB_MIN_VIA_DRILL_MM,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.validation.kicad_drc import DRCReport

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Result status of a checklist item."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class CheckResult:
    """Result of a single checklist verification."""

    category: str
    name: str
    status: CheckStatus
    message: str
    details: str = ""


@dataclass(frozen=True)
class ChecklistReport:
    """Complete pre-fabrication checklist report."""

    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """True if no FAIL results."""
        return not any(r.status == CheckStatus.FAIL for r in self.results)

    @property
    def fail_count(self) -> int:
        """Number of failed checks."""
        return sum(1 for r in self.results if r.status == CheckStatus.FAIL)

    @property
    def warn_count(self) -> int:
        """Number of warning checks."""
        return sum(1 for r in self.results if r.status == CheckStatus.WARN)

    @property
    def pass_count(self) -> int:
        """Number of passed checks."""
        return sum(1 for r in self.results if r.status == CheckStatus.PASS)


def _check_board_dimensions(pcb: PCBDesign) -> list[CheckResult]:
    """Verify board dimensions are within JLCPCB limits."""
    results: list[CheckResult] = []

    if not pcb.outline.polygon:
        results.append(
            CheckResult(
                category="Mechanical",
                name="Board outline",
                status=CheckStatus.FAIL,
                message="No board outline defined",
            )
        )
        return results

    # Calculate bounding box from outline points.
    xs = [p.x for p in pcb.outline.polygon]
    ys = [p.y for p in pcb.outline.polygon]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    min_w, min_h = JLCPCB_MIN_BOARD_SIZE_MM
    max_w, max_h = JLCPCB_MAX_BOARD_SIZE_MM

    if width < min_w or height < min_h:
        results.append(
            CheckResult(
                category="Mechanical",
                name="Minimum board size",
                status=CheckStatus.FAIL,
                message=f"Board {width:.1f}x{height:.1f}mm below JLCPCB min {min_w}x{min_h}mm",
            )
        )
    elif width > max_w or height > max_h:
        results.append(
            CheckResult(
                category="Mechanical",
                name="Maximum board size",
                status=CheckStatus.FAIL,
                message=f"Board {width:.1f}x{height:.1f}mm exceeds JLCPCB max {max_w}x{max_h}mm",
            )
        )
    else:
        results.append(
            CheckResult(
                category="Mechanical",
                name="Board dimensions",
                status=CheckStatus.PASS,
                message=f"Board {width:.1f}x{height:.1f}mm within JLCPCB limits",
            )
        )

    # Check outline is closed.
    outline = pcb.outline.polygon
    if len(outline) >= 3:
        first = outline[0]
        last = outline[-1]
        if abs(first.x - last.x) > 0.01 or abs(first.y - last.y) > 0.01:
            results.append(
                CheckResult(
                    category="Mechanical",
                    name="Outline closure",
                    status=CheckStatus.FAIL,
                    message="Board outline is not closed (first point != last point)",
                )
            )
        else:
            results.append(
                CheckResult(
                    category="Mechanical",
                    name="Outline closure",
                    status=CheckStatus.PASS,
                    message="Board outline is properly closed",
                )
            )

    return results


def _check_trace_widths(pcb: PCBDesign) -> list[CheckResult]:
    """Verify minimum trace widths."""
    results: list[CheckResult] = []

    if not pcb.tracks:
        results.append(
            CheckResult(
                category="Manufacturing",
                name="Trace width",
                status=CheckStatus.SKIP,
                message="No tracks found (routing not yet done)",
            )
        )
        return results

    min_width = min(t.width for t in pcb.tracks)
    if min_width < JLCPCB_MIN_TRACE_MM:
        results.append(
            CheckResult(
                category="Manufacturing",
                name="Minimum trace width",
                status=CheckStatus.FAIL,
                message=(
                    f"Minimum trace width {min_width:.3f}mm "
                    f"below JLCPCB min {JLCPCB_MIN_TRACE_MM:.3f}mm"
                ),
            )
        )
    else:
        results.append(
            CheckResult(
                category="Manufacturing",
                name="Minimum trace width",
                status=CheckStatus.PASS,
                message=f"All traces >= {JLCPCB_MIN_TRACE_MM:.3f}mm (min: {min_width:.3f}mm)",
            )
        )

    return results


def _check_via_sizes(pcb: PCBDesign) -> list[CheckResult]:
    """Verify via drill and annular ring sizes."""
    results: list[CheckResult] = []

    if not pcb.vias:
        results.append(
            CheckResult(
                category="Manufacturing",
                name="Via sizes",
                status=CheckStatus.SKIP,
                message="No vias found",
            )
        )
        return results

    for via in pcb.vias:
        if via.drill < JLCPCB_MIN_VIA_DRILL_MM:
            results.append(
                CheckResult(
                    category="Manufacturing",
                    name="Via drill size",
                    status=CheckStatus.FAIL,
                    message=(
                        f"Via at ({via.position.x:.2f},{via.position.y:.2f}) "
                        f"drill {via.drill:.3f}mm "
                        f"below min {JLCPCB_MIN_VIA_DRILL_MM:.3f}mm"
                    ),
                )
            )
            break  # Report first violation only.

    annular_rings = [(v.size - v.drill) / 2.0 for v in pcb.vias]
    if annular_rings:
        min_ring = min(annular_rings)
        if min_ring < JLCPCB_MIN_VIA_ANNULAR_RING_MM:
            results.append(
                CheckResult(
                    category="Manufacturing",
                    name="Annular ring",
                    status=CheckStatus.FAIL,
                    message=(
                        f"Min annular ring {min_ring:.3f}mm "
                        f"below {JLCPCB_MIN_VIA_ANNULAR_RING_MM:.3f}mm"
                    ),
                )
            )
        else:
            results.append(
                CheckResult(
                    category="Manufacturing",
                    name="Annular ring",
                    status=CheckStatus.PASS,
                    message=f"All annular rings >= {JLCPCB_MIN_VIA_ANNULAR_RING_MM:.3f}mm",
                )
            )

    return results


def _check_nets(pcb: PCBDesign) -> list[CheckResult]:
    """Verify net consistency."""
    results: list[CheckResult] = []

    if not pcb.nets:
        results.append(
            CheckResult(
                category="Electrical",
                name="Net definitions",
                status=CheckStatus.WARN,
                message="No nets defined in PCB",
            )
        )
        return results

    # Check for empty net names (excluding net 0 which is unconnected).
    named_nets = [n for n in pcb.nets if n.number > 0]
    if not named_nets:
        results.append(
            CheckResult(
                category="Electrical",
                name="Net definitions",
                status=CheckStatus.WARN,
                message="No named nets found (only unconnected net 0)",
            )
        )
    else:
        results.append(
            CheckResult(
                category="Electrical",
                name="Net definitions",
                status=CheckStatus.PASS,
                message=f"{len(named_nets)} nets defined",
            )
        )

    return results


def _check_footprints(pcb: PCBDesign) -> list[CheckResult]:
    """Verify footprint placement."""
    results: list[CheckResult] = []

    if not pcb.footprints:
        results.append(
            CheckResult(
                category="Electrical",
                name="Components",
                status=CheckStatus.FAIL,
                message="No footprints placed on board",
            )
        )
        return results

    # Check for duplicate reference designators.
    refs = [fp.ref for fp in pcb.footprints]
    seen: set[str] = set()
    dupes: list[str] = []
    for r in refs:
        if r in seen:
            dupes.append(r)
        seen.add(r)

    if dupes:
        results.append(
            CheckResult(
                category="Electrical",
                name="Duplicate refs",
                status=CheckStatus.FAIL,
                message=f"Duplicate reference designators: {', '.join(dupes)}",
            )
        )
    else:
        results.append(
            CheckResult(
                category="Electrical",
                name="Reference designators",
                status=CheckStatus.PASS,
                message=f"All {len(refs)} references are unique",
            )
        )

    return results


def _check_drc_results(drc_report: DRCReport | None) -> list[CheckResult]:
    """Incorporate KiCad CLI DRC results if available."""
    results: list[CheckResult] = []

    if drc_report is None:
        results.append(
            CheckResult(
                category="DRC",
                name="KiCad DRC",
                status=CheckStatus.SKIP,
                message="KiCad CLI DRC not run — run for full validation",
            )
        )
        return results

    if drc_report.passed:
        results.append(
            CheckResult(
                category="DRC",
                name="KiCad DRC",
                status=CheckStatus.PASS,
                message=(
                    f"DRC passed with {drc_report.warning_count} warnings, "
                    f"{drc_report.unconnected_count} unconnected"
                ),
            )
        )
    else:
        results.append(
            CheckResult(
                category="DRC",
                name="KiCad DRC",
                status=CheckStatus.FAIL,
                message=(
                    f"DRC failed: {drc_report.error_count} errors, "
                    f"{drc_report.warning_count} warnings"
                ),
            )
        )

    if drc_report.unconnected_count > 0:
        results.append(
            CheckResult(
                category="DRC",
                name="Unconnected nets",
                status=CheckStatus.WARN,
                message=f"{drc_report.unconnected_count} unconnected net items",
            )
        )

    return results


def run_checklist(
    pcb: PCBDesign,
    drc_report: DRCReport | None = None,
) -> ChecklistReport:
    """Run the full pre-fabrication checklist.

    Args:
        pcb: The PCB design to validate.
        drc_report: Optional KiCad CLI DRC report for additional checks.

    Returns:
        A :class:`ChecklistReport` with all check results.
    """
    all_results: list[CheckResult] = []

    all_results.extend(_check_board_dimensions(pcb))
    all_results.extend(_check_trace_widths(pcb))
    all_results.extend(_check_via_sizes(pcb))
    all_results.extend(_check_nets(pcb))
    all_results.extend(_check_footprints(pcb))
    all_results.extend(_check_drc_results(drc_report))

    return ChecklistReport(results=tuple(all_results))


def format_checklist(report: ChecklistReport) -> str:
    """Format a checklist report as a human-readable string.

    Args:
        report: The checklist report to format.

    Returns:
        Multi-line formatted string.
    """
    status_icons = {
        CheckStatus.PASS: "PASS",
        CheckStatus.WARN: "WARN",
        CheckStatus.FAIL: "FAIL",
        CheckStatus.SKIP: "SKIP",
    }

    lines: list[str] = []
    overall = "READY" if report.passed else "NOT READY"
    lines.append(f"Pre-Fabrication Checklist: {overall}")
    lines.append(f"  Passed: {report.pass_count}  Warnings: {report.warn_count}  "
                 f"Failed: {report.fail_count}")
    lines.append("")

    current_category = ""
    for result in report.results:
        if result.category != current_category:
            current_category = result.category
            lines.append(f"[{current_category}]")
        icon = status_icons.get(result.status, "????")
        lines.append(f"  [{icon}] {result.name}: {result.message}")
        if result.details:
            lines.append(f"         {result.details}")

    return "\n".join(lines)

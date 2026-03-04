"""DRC (Design Rule Check) violation models and engine for KiCad PCB validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from kicad_pipeline.constants import JLCPCB_MIN_TRACE_MM

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import DesignRules, PCBDesign


class Severity(Enum):
    """Severity level for a DRC violation."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class DRCViolation:
    """A single DRC violation."""

    rule: str
    message: str
    severity: Severity
    ref: str = ""
    layer: str = ""


@dataclass(frozen=True)
class DRCReport:
    """Result of a DRC run."""

    violations: tuple[DRCViolation, ...]

    @property
    def errors(self) -> tuple[DRCViolation, ...]:
        """Return only ERROR-severity violations."""
        return tuple(v for v in self.violations if v.severity == Severity.ERROR)

    @property
    def warnings(self) -> tuple[DRCViolation, ...]:
        """Return only WARNING-severity violations."""
        return tuple(v for v in self.violations if v.severity == Severity.WARNING)

    @property
    def passed(self) -> bool:
        """True if there are no ERROR-severity violations."""
        return len(self.errors) == 0


def run_drc(
    pcb: PCBDesign,
    design_rules: DesignRules | None = None,
) -> DRCReport:
    """Run all design rule checks against a PCBDesign.

    Args:
        pcb: The PCB design to validate.
        design_rules: Optional override for design rules; defaults to pcb.design_rules.

    Returns:
        A DRCReport containing all violations found.
    """
    rules = design_rules if design_rules is not None else pcb.design_rules
    violations: list[DRCViolation] = []

    violations.extend(_check_min_trace_width(pcb, rules))
    violations.extend(_check_min_clearance(pcb, rules))
    violations.extend(_check_min_via_size(pcb, rules))
    violations.extend(_check_board_outline(pcb))
    violations.extend(_check_net_consistency(pcb))
    violations.extend(_check_duplicate_refs(pcb))
    violations.extend(_check_unconnected_pads(pcb))

    return DRCReport(violations=tuple(violations))


def _check_min_trace_width(
    pcb: PCBDesign,
    rules: DesignRules,
) -> list[DRCViolation]:
    """Check each track against absolute minimum and design-rule minimum widths."""
    violations: list[DRCViolation] = []
    for track in pcb.tracks:
        if track.width < JLCPCB_MIN_TRACE_MM:
            violations.append(
                DRCViolation(
                    rule="min_trace_width",
                    message=(
                        f"Track on {track.layer} width {track.width:.3f}mm below"
                        f" JLCPCB absolute minimum {JLCPCB_MIN_TRACE_MM:.3f}mm"
                    ),
                    severity=Severity.ERROR,
                    layer=track.layer,
                )
            )
        elif track.width < rules.default_trace_width_mm:
            violations.append(
                DRCViolation(
                    rule="min_trace_width",
                    message=(
                        f"Track on {track.layer} width {track.width:.3f}mm below"
                        f" design rule minimum {rules.default_trace_width_mm:.3f}mm"
                    ),
                    severity=Severity.WARNING,
                    layer=track.layer,
                )
            )
    return violations


def _check_min_clearance(
    pcb: PCBDesign,
    rules: DesignRules,
) -> list[DRCViolation]:
    """Check for duplicate track segments (simplified clearance check).

    For tracks on the same net and layer with identical start/end points, flag
    a WARNING for the duplicate segment.
    """
    violations: list[DRCViolation] = []
    seen: set[tuple[int, str, tuple[float, float], tuple[float, float]]] = set()
    for track in pcb.tracks:
        key = (
            track.net_number,
            track.layer,
            (track.start.x, track.start.y),
            (track.end.x, track.end.y),
        )
        if key in seen:
            violations.append(
                DRCViolation(
                    rule="min_clearance",
                    message=f"Duplicate track segment on {track.layer}",
                    severity=Severity.WARNING,
                    layer=track.layer,
                )
            )
        else:
            seen.add(key)
    return violations


def _check_min_via_size(
    pcb: PCBDesign,
    rules: DesignRules,
) -> list[DRCViolation]:
    """Check each via drill and diameter against design-rule minimums."""
    violations: list[DRCViolation] = []
    for via in pcb.vias:
        if via.drill < rules.min_via_drill_mm:
            violations.append(
                DRCViolation(
                    rule="min_via_size",
                    message=(
                        f"Via drill {via.drill:.3f}mm below minimum"
                        f" {rules.min_via_drill_mm:.3f}mm"
                    ),
                    severity=Severity.ERROR,
                )
            )
        if via.size < rules.min_via_diameter_mm:
            violations.append(
                DRCViolation(
                    rule="min_via_size",
                    message=(
                        f"Via diameter {via.size:.3f}mm below minimum"
                        f" {rules.min_via_diameter_mm:.3f}mm"
                    ),
                    severity=Severity.ERROR,
                )
            )
    return violations


def _check_board_outline(pcb: PCBDesign) -> list[DRCViolation]:
    """Check that the board outline polygon is valid."""
    violations: list[DRCViolation] = []
    pts = pcb.outline.polygon

    if len(pts) < 3:
        violations.append(
            DRCViolation(
                rule="board_outline_closed",
                message="Board outline has fewer than 3 points",
                severity=Severity.ERROR,
                layer="Edge.Cuts",
            )
        )
        return violations

    first = pts[0]
    last = pts[-1]
    tol = 1e-9
    if abs(first.x - last.x) > tol or abs(first.y - last.y) > tol:
        violations.append(
            DRCViolation(
                rule="board_outline_closed",
                message="Board outline polygon is not closed (first != last point)",
                severity=Severity.WARNING,
                layer="Edge.Cuts",
            )
        )

    return violations


def _check_net_consistency(pcb: PCBDesign) -> list[DRCViolation]:
    """Check that pad net numbers are all present in the PCB net list."""
    violations: list[DRCViolation] = []
    valid_net_numbers = {net.number for net in pcb.nets}

    for fp in pcb.footprints:
        for pad in fp.pads:
            if (
                pad.net_number is not None
                and pad.net_number != 0
                and pad.net_number not in valid_net_numbers
            ):
                violations.append(
                    DRCViolation(
                        rule="net_consistency",
                        message=(
                            f"Pad {fp.ref}.{pad.number} references net number"
                            f" {pad.net_number} not in net list"
                        ),
                        severity=Severity.ERROR,
                        ref=fp.ref,
                    )
                )
    return violations


def _check_duplicate_refs(pcb: PCBDesign) -> list[DRCViolation]:
    """Check for duplicate footprint reference designators."""
    violations: list[DRCViolation] = []
    seen: dict[str, int] = {}
    for fp in pcb.footprints:
        seen[fp.ref] = seen.get(fp.ref, 0) + 1

    for ref, count in seen.items():
        if count > 1:
            violations.append(
                DRCViolation(
                    rule="duplicate_refs",
                    message=f"Duplicate footprint reference {ref}",
                    severity=Severity.ERROR,
                    ref=ref,
                )
            )
    return violations


def is_intra_footprint_violation(
    violation: DRCViolation,
    pcb: PCBDesign,
) -> bool:
    """Return True if *violation* is an intra-footprint pad clearance issue.

    Many fine-pitch ICs (QFN, MSOP, TSSOP) have pads closer together than
    the default clearance rule.  KiCad's DRC flags these as violations but
    they are false positives — the pad spacing is fixed by the package.

    This function checks if a clearance violation refers to pads on the
    same footprint.  If so, it should be downgraded or excluded.

    Args:
        violation: The DRC violation to check.
        pcb: The PCB design for footprint lookup.

    Returns:
        ``True`` when the violation involves pads within a single footprint.
    """
    if violation.rule != "min_clearance":
        return False
    if not violation.ref:
        return False
    # If the ref field references a single footprint, it's intra-footprint
    fp = pcb.get_footprint(violation.ref)
    return fp is not None


def _check_unconnected_pads(pcb: PCBDesign) -> list[DRCViolation]:
    """Flag SMD/thru-hole pads with net_number == 0 as unconnected."""
    violations: list[DRCViolation] = []
    for fp in pcb.footprints:
        for pad in fp.pads:
            if pad.net_number == 0 and pad.pad_type != "np_thru_hole":
                violations.append(
                    DRCViolation(
                        rule="unconnected_pads",
                        message=(
                            f"Pad {fp.ref}.{pad.number} is unconnected (net 0)"
                        ),
                        severity=Severity.INFO,
                        ref=fp.ref,
                    )
                )
    return violations

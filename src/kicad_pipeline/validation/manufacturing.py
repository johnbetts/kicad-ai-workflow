"""JLCPCB manufacturing constraint validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    JLCPCB_MAX_BOARD_SIZE_MM,
    JLCPCB_MIN_TRACE_MM,
)
from kicad_pipeline.validation.drc import Severity

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.production import BOMEntry

# ---------------------------------------------------------------------------
# JLCPCB constants not already in constants.py
# ---------------------------------------------------------------------------

_JLCPCB_MIN_VIA_DRILL_MM: float = 0.2
_JLCPCB_MIN_ANNULAR_RING_MM: float = 0.1
_JLCPCB_MIN_PASTE_APERTURE_MM: float = 0.25

_LAYER_F_CU = "F.Cu"
_LAYER_B_CU = "B.Cu"
_LAYER_F_PASTE = "F.Paste"
_LAYER_B_PASTE = "B.Paste"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManufacturingViolation:
    """A single JLCPCB manufacturing constraint violation."""

    rule: str
    message: str
    severity: Severity


@dataclass(frozen=True)
class ManufacturingReport:
    """Result of JLCPCB manufacturing constraint checks."""

    violations: tuple[ManufacturingViolation, ...]

    @property
    def errors(self) -> tuple[ManufacturingViolation, ...]:
        """Return only ERROR-severity violations."""
        return tuple(v for v in self.violations if v.severity == Severity.ERROR)

    @property
    def warnings(self) -> tuple[ManufacturingViolation, ...]:
        """Return only WARNING-severity violations."""
        return tuple(v for v in self.violations if v.severity == Severity.WARNING)

    @property
    def passed(self) -> bool:
        """True if there are no ERROR-severity violations."""
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _board_dimensions(pcb: PCBDesign) -> tuple[float, float]:
    """Return (width, height) of the board outline bounding box in mm."""
    pts = pcb.outline.polygon
    if not pts:
        return 0.0, 0.0
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width, height


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_manufacturing_checks(
    pcb: PCBDesign,
    bom_entries: tuple[BOMEntry, ...] | None = None,
) -> ManufacturingReport:
    """Run JLCPCB manufacturing constraint checks on *pcb*.

    Args:
        pcb: The PCB design to check.
        bom_entries: Optional BOM entries used for LCSC part-number checks.

    Returns:
        A :class:`ManufacturingReport` with all detected violations.
    """
    violations: list[ManufacturingViolation] = []

    # ------------------------------------------------------------------
    # 1. trace_width_jlcpcb
    # ------------------------------------------------------------------
    for track in pcb.tracks:
        if track.width < JLCPCB_MIN_TRACE_MM:
            violations.append(
                ManufacturingViolation(
                    rule="trace_width_jlcpcb",
                    message=(
                        f"Trace width {track.width:.3f}mm below JLCPCB minimum"
                        f" {JLCPCB_MIN_TRACE_MM:.3f}mm"
                    ),
                    severity=Severity.ERROR,
                )
            )

    # ------------------------------------------------------------------
    # 2. via_drill_jlcpcb
    # ------------------------------------------------------------------
    for via in pcb.vias:
        if via.drill < _JLCPCB_MIN_VIA_DRILL_MM:
            violations.append(
                ManufacturingViolation(
                    rule="via_drill_jlcpcb",
                    message=(
                        f"Via drill {via.drill:.3f}mm below JLCPCB minimum"
                        f" {_JLCPCB_MIN_VIA_DRILL_MM:.3f}mm"
                    ),
                    severity=Severity.ERROR,
                )
            )

    # ------------------------------------------------------------------
    # 3. board_dimensions
    # ------------------------------------------------------------------
    max_w, max_h = JLCPCB_MAX_BOARD_SIZE_MM
    board_w, board_h = _board_dimensions(pcb)
    if board_w <= 0.0 or board_h <= 0.0:
        violations.append(
            ManufacturingViolation(
                rule="board_dimensions",
                message="Board has zero or negative dimensions",
                severity=Severity.ERROR,
            )
        )
    else:
        if board_w > max_w or board_h > max_h:
            violations.append(
                ManufacturingViolation(
                    rule="board_dimensions",
                    message=(
                        f"Board size {board_w:.1f}x{board_h:.1f}mm exceeds JLCPCB"
                        f" maximum {max_w:.0f}x{max_h:.0f}mm"
                    ),
                    severity=Severity.ERROR,
                )
            )

    # ------------------------------------------------------------------
    # 4. acid_trap_check
    # ------------------------------------------------------------------
    for track in pcb.tracks:
        if math.isclose(track.start.x, track.end.x) and math.isclose(
            track.start.y, track.end.y
        ):
            violations.append(
                ManufacturingViolation(
                    rule="acid_trap_check",
                    message=(
                        f"Potential acid trap at"
                        f" ({track.start.x:.2f},{track.start.y:.2f})"
                    ),
                    severity=Severity.WARNING,
                )
            )

    # ------------------------------------------------------------------
    # 5. paste_aperture_check
    # ------------------------------------------------------------------
    for fp in pcb.footprints:
        for pad in fp.pads:
            if pad.pad_type != "smd":
                continue
            has_paste = _LAYER_F_PASTE in pad.layers or _LAYER_B_PASTE in pad.layers
            if not has_paste:
                continue
            too_small = (
                pad.size_x < _JLCPCB_MIN_PASTE_APERTURE_MM
                or pad.size_y < _JLCPCB_MIN_PASTE_APERTURE_MM
            )
            if too_small:
                violations.append(
                    ManufacturingViolation(
                        rule="paste_aperture_check",
                        message=(
                            f"Paste aperture too small for pad {fp.ref}.{pad.number}"
                        ),
                        severity=Severity.WARNING,
                    )
                )

    # ------------------------------------------------------------------
    # 6. lcsc_check
    # ------------------------------------------------------------------
    if bom_entries is not None:
        for entry in bom_entries:
            if not entry.lcsc:
                for ref in entry.designators:
                    violations.append(
                        ManufacturingViolation(
                            rule="lcsc_check",
                            message=f"Component {ref} has no LCSC part number",
                            severity=Severity.WARNING,
                        )
                    )

    # ------------------------------------------------------------------
    # 7. smt_side_check
    # ------------------------------------------------------------------
    has_front_smd = any(
        fp.layer == _LAYER_F_CU and fp.attr == "smd" for fp in pcb.footprints
    )
    has_back_smd = any(
        fp.layer == _LAYER_B_CU and fp.attr == "smd" for fp in pcb.footprints
    )
    if has_front_smd and has_back_smd:
        violations.append(
            ManufacturingViolation(
                rule="smt_side_check",
                message=(
                    "SMD components on both sides"
                    " - JLCPCB double-sided assembly surcharge applies"
                ),
                severity=Severity.WARNING,
            )
        )

    return ManufacturingReport(violations=tuple(violations))

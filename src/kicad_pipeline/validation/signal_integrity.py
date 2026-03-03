"""Signal integrity checks for PCB designs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.validation.drc import Severity

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign, Track
    from kicad_pipeline.models.requirements import ProjectRequirements

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USB_DIFF_MAX_SKEW_MM: float = 0.5
_ANALOG_PARALLEL_TOLERANCE_MM: float = 0.5
_SPI_MAX_TRACE_MM: float = 100.0

_SPI_NET_KEYWORDS: tuple[str, ...] = ("SCK", "MOSI", "MISO", "CS")
_WIFI_COMPONENT_KEYWORDS: tuple[str, ...] = ("ESP32", "WiFi")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIViolation:
    """A single signal integrity violation."""

    rule: str
    message: str
    severity: Severity


@dataclass(frozen=True)
class SIReport:
    """Result of signal integrity checks."""

    violations: tuple[SIViolation, ...]

    @property
    def errors(self) -> tuple[SIViolation, ...]:
        """Return only ERROR-severity violations."""
        return tuple(v for v in self.violations if v.severity == Severity.ERROR)

    @property
    def warnings(self) -> tuple[SIViolation, ...]:
        """Return only WARNING-severity violations."""
        return tuple(v for v in self.violations if v.severity == Severity.WARNING)

    @property
    def passed(self) -> bool:
        """True if there are no ERROR-severity violations."""
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _track_length(track: Track) -> float:
    """Return the Euclidean length of a track segment in mm."""
    dx = track.end.x - track.start.x
    dy = track.end.y - track.start.y
    return math.sqrt(dx * dx + dy * dy)


def _net_total_length(pcb: PCBDesign, net_number: int) -> float:
    """Return the total routed length for a given net number in mm."""
    return sum(_track_length(t) for t in pcb.tracks if t.net_number == net_number)


def _net_number_by_name(pcb: PCBDesign, name: str) -> int | None:
    """Return net number for exact net name match, or None."""
    for net in pcb.nets:
        if net.name == name:
            return net.number
    return None


def _nets_containing(pcb: PCBDesign, keyword: str) -> list[tuple[int, str]]:
    """Return list of (net_number, net_name) for nets whose name contains *keyword*."""
    return [(n.number, n.name) for n in pcb.nets if keyword in n.name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_si_checks(
    pcb: PCBDesign,
    requirements: ProjectRequirements | None = None,
) -> SIReport:
    """Run signal integrity checks on *pcb*.

    Args:
        pcb: The PCB design to check.
        requirements: Optional project requirements (currently unused).

    Returns:
        A :class:`SIReport` with all detected violations.
    """
    violations: list[SIViolation] = []

    # ------------------------------------------------------------------
    # 1. usb_diff_pair_check
    # ------------------------------------------------------------------
    dp_net = _net_number_by_name(pcb, "D+")
    dm_net = _net_number_by_name(pcb, "D-")

    if dp_net is not None and dm_net is not None:
        len_dp = _net_total_length(pcb, dp_net)
        len_dm = _net_total_length(pcb, dm_net)
        if abs(len_dp - len_dm) > _USB_DIFF_MAX_SKEW_MM:
            violations.append(
                SIViolation(
                    rule="usb_diff_pair_check",
                    message=(
                        f"USB differential pair length mismatch:"
                        f" D+={len_dp:.2f}mm D-={len_dm:.2f}mm"
                        f" (max skew {_USB_DIFF_MAX_SKEW_MM:.1f}mm)"
                    ),
                    severity=Severity.WARNING,
                )
            )

    # ------------------------------------------------------------------
    # 2. analog_digital_parallel_check
    # ------------------------------------------------------------------
    analog_nets = _nets_containing(pcb, "ANALOG") + _nets_containing(pcb, "ADC")
    analog_net_numbers = {num for num, _ in analog_nets}

    analog_tracks = [
        t
        for t in pcb.tracks
        if t.layer == "F.Cu" and t.net_number in analog_net_numbers
    ]
    other_tracks = [
        t
        for t in pcb.tracks
        if t.layer == "F.Cu" and t.net_number not in analog_net_numbers
    ]

    if analog_tracks and other_tracks:
        for a_track in analog_tracks:
            a_y = (a_track.start.y + a_track.end.y) / 2.0
            for o_track in other_tracks:
                o_y = (o_track.start.y + o_track.end.y) / 2.0
                if abs(a_y - o_y) < _ANALOG_PARALLEL_TOLERANCE_MM:
                    violations.append(
                        SIViolation(
                            rule="analog_digital_parallel_check",
                            message=(
                                "Potential analog/digital parallel coupling detected"
                            ),
                            severity=Severity.WARNING,
                        )
                    )
                    # Only report once
                    break
            else:
                continue
            break

    # ------------------------------------------------------------------
    # 3. antenna_keepout_check
    # ------------------------------------------------------------------
    has_wifi_component = any(
        any(kw in fp.value for kw in _WIFI_COMPONENT_KEYWORDS)
        for fp in pcb.footprints
    )

    if has_wifi_component:
        has_antenna_keepout = any(
            ko.no_copper and ko.no_vias for ko in pcb.keepouts
        )
        if not has_antenna_keepout:
            violations.append(
                SIViolation(
                    rule="antenna_keepout_check",
                    message=(
                        "No antenna keepout zone found"
                        " - ESP32/WiFi antenna area should be keepout"
                    ),
                    severity=Severity.WARNING,
                )
            )

    # ------------------------------------------------------------------
    # 4. trace_length_check (SPI nets)
    # ------------------------------------------------------------------
    for keyword in _SPI_NET_KEYWORDS:
        spi_nets = _nets_containing(pcb, keyword)
        for net_num, net_name in spi_nets:
            total_length = _net_total_length(pcb, net_num)
            if total_length > _SPI_MAX_TRACE_MM:
                violations.append(
                    SIViolation(
                        rule="trace_length_check",
                        message=(
                            f"Long SPI trace {net_name}: {total_length:.1f}mm"
                            " - consider impedance matching"
                        ),
                        severity=Severity.WARNING,
                    )
                )

    return SIReport(violations=tuple(violations))

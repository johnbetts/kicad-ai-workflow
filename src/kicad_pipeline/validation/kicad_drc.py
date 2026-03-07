"""Run DRC via ``kicad-cli pcb drc`` and parse the JSON report.

This module shells out to the KiCad 9 CLI to perform a full design-rule
check, then parses the resulting JSON into typed dataclasses.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kicad_pipeline.exceptions import DRCError, ValidationError

logger = logging.getLogger(__name__)

# Default kicad-cli path (macOS).  Override via KICAD_CLI env var.
_KICAD_CLI_DEFAULT = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"


@dataclass(frozen=True)
class DRCPosition:
    """Board-space position of a DRC item."""

    x: float
    y: float


@dataclass(frozen=True)
class DRCItem:
    """An object involved in a DRC violation."""

    description: str
    ref: str = ""
    pos: DRCPosition | None = None
    layer: str = ""


@dataclass(frozen=True)
class DRCViolation:
    """A single DRC violation from kicad-cli output."""

    type: str
    severity: str
    description: str
    items: tuple[DRCItem, ...] = ()
    excluded: bool = False


@dataclass(frozen=True)
class DRCReport:
    """Parsed results of a ``kicad-cli pcb drc`` run."""

    violations: tuple[DRCViolation, ...] = ()
    unconnected: tuple[DRCViolation, ...] = ()
    schema_version: int = 0
    source: str = ""
    coordinate_units: str = "mm"

    @property
    def passed(self) -> bool:
        """True if there are no error-severity violations."""
        return self.error_count == 0

    @property
    def error_count(self) -> int:
        """Count of error-severity violations (excluding unconnected)."""
        return sum(
            1 for v in self.violations if v.severity == "error" and not v.excluded
        )

    @property
    def warning_count(self) -> int:
        """Count of warning-severity violations."""
        return sum(
            1 for v in self.violations if v.severity == "warning" and not v.excluded
        )

    @property
    def unconnected_count(self) -> int:
        """Count of unconnected net items."""
        return sum(1 for v in self.unconnected if not v.excluded)


def _find_kicad_cli() -> str:
    """Locate the kicad-cli binary.

    Returns:
        Path to kicad-cli.

    Raises:
        DRCError: If kicad-cli cannot be found.
    """
    import os

    env_path = os.environ.get("KICAD_CLI")
    if env_path and Path(env_path).is_file():
        return env_path
    if Path(_KICAD_CLI_DEFAULT).is_file():
        return _KICAD_CLI_DEFAULT
    # Try PATH.
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
    msg = "Cannot find kicad-cli. Install KiCad 9 or set KICAD_CLI env var."
    raise DRCError(msg)


def _parse_position(pos_data: dict[str, object]) -> DRCPosition | None:
    """Parse a position object from DRC JSON."""
    x = pos_data.get("x")
    y = pos_data.get("y")
    if x is not None and y is not None:
        return DRCPosition(x=float(str(x)), y=float(str(y)))
    return None


def _parse_item(item_data: dict[str, object]) -> DRCItem:
    """Parse a DRC item from JSON."""
    desc = str(item_data.get("description", ""))
    ref = str(item_data.get("uuid", ""))
    layer = ""
    pos: DRCPosition | None = None

    pos_data = item_data.get("pos")
    if isinstance(pos_data, dict):
        pos = _parse_position(pos_data)

    return DRCItem(description=desc, ref=ref, pos=pos, layer=layer)


def _parse_violation(v_data: dict[str, object]) -> DRCViolation:
    """Parse a single violation from JSON."""
    vtype = str(v_data.get("type", "unknown"))
    severity = str(v_data.get("severity", "error"))
    desc = str(v_data.get("description", ""))
    excluded = bool(v_data.get("excluded", False))

    items_data = v_data.get("items", [])
    items: list[DRCItem] = []
    if isinstance(items_data, list):
        for item in items_data:
            if isinstance(item, dict):
                items.append(_parse_item(item))

    return DRCViolation(
        type=vtype,
        severity=severity,
        description=desc,
        items=tuple(items),
        excluded=excluded,
    )


def parse_drc_json(json_text: str) -> DRCReport:
    """Parse a kicad-cli DRC JSON report.

    Args:
        json_text: Raw JSON string from ``kicad-cli pcb drc --format json``.

    Returns:
        A :class:`DRCReport` with parsed violations.

    Raises:
        ValidationError: If the JSON cannot be parsed.
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        msg = f"Failed to parse DRC JSON: {exc}"
        raise ValidationError(msg) from exc

    schema_version = int(data.get("$schema_version", 0))
    source = str(data.get("source", ""))
    coord_units = str(data.get("coordinate_units", "mm"))

    violations: list[DRCViolation] = []
    for v in data.get("violations", []):
        if isinstance(v, dict):
            violations.append(_parse_violation(v))

    unconnected: list[DRCViolation] = []
    for u in data.get("unconnected_items", []):
        if isinstance(u, dict):
            unconnected.append(_parse_violation(u))

    return DRCReport(
        violations=tuple(violations),
        unconnected=tuple(unconnected),
        schema_version=schema_version,
        source=source,
        coordinate_units=coord_units,
    )


def run_drc(
    pcb_path: str | Path,
    output_dir: str | Path | None = None,
    severity_all: bool = True,
) -> DRCReport:
    """Run ``kicad-cli pcb drc`` on a PCB file and return parsed results.

    Args:
        pcb_path: Path to the ``.kicad_pcb`` file.
        output_dir: Directory for the JSON output file.  Uses a temp dir
            if not specified.
        severity_all: Include all severities (default ``True``).

    Returns:
        A :class:`DRCReport` with all violations parsed.

    Raises:
        DRCError: If kicad-cli fails to execute.
    """
    pcb_path = Path(pcb_path)
    if not pcb_path.is_file():
        msg = f"PCB file not found: {pcb_path}"
        raise DRCError(msg)

    kicad_cli = _find_kicad_cli()

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{pcb_path.stem}_drc.json"
        return _execute_drc(kicad_cli, pcb_path, json_path, severity_all)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "drc_report.json"
        return _execute_drc(kicad_cli, pcb_path, json_path, severity_all)


def _execute_drc(
    kicad_cli: str,
    pcb_path: Path,
    json_path: Path,
    severity_all: bool,
) -> DRCReport:
    """Execute kicad-cli and parse output."""
    cmd = [
        kicad_cli,
        "pcb",
        "drc",
        "--format",
        "json",
        "--output",
        str(json_path),
    ]
    if severity_all:
        cmd.append("--severity-all")
    cmd.append(str(pcb_path))

    logger.info("Running DRC: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"kicad-cli DRC timed out after 120s on {pcb_path}"
        raise DRCError(msg) from exc
    except FileNotFoundError as exc:
        msg = f"kicad-cli not found at {kicad_cli}"
        raise DRCError(msg) from exc

    if not json_path.is_file():
        msg = (
            f"kicad-cli DRC did not produce output file. "
            f"Exit code: {result.returncode}. "
            f"stderr: {result.stderr[:500]}"
        )
        raise DRCError(msg)

    json_text = json_path.read_text(encoding="utf-8")
    report = parse_drc_json(json_text)
    logger.info(
        "DRC complete: %d errors, %d warnings, %d unconnected",
        report.error_count,
        report.warning_count,
        report.unconnected_count,
    )
    return report


def summarize_drc(report: DRCReport) -> str:
    """Generate a human-readable DRC summary.

    Args:
        report: Parsed DRC report.

    Returns:
        Multi-line summary string suitable for display.
    """
    lines: list[str] = []
    status = "PASSED" if report.passed else "FAILED"
    lines.append(f"DRC Result: {status}")
    lines.append(f"  Errors:      {report.error_count}")
    lines.append(f"  Warnings:    {report.warning_count}")
    lines.append(f"  Unconnected: {report.unconnected_count}")

    if report.violations:
        lines.append("")
        lines.append("Violations by type:")
        type_counts: dict[str, int] = {}
        for v in report.violations:
            if not v.excluded:
                type_counts[v.type] = type_counts.get(v.type, 0) + 1
        for vtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {vtype}: {count}")

    if report.unconnected:
        lines.append("")
        lines.append(f"Unconnected items: {report.unconnected_count}")

    return "\n".join(lines)


def categorize_violations(
    report: DRCReport,
) -> dict[str, list[DRCViolation]]:
    """Group violations into auto-fixable vs manual-fix categories.

    Args:
        report: Parsed DRC report.

    Returns:
        Dict with keys ``"auto_fixable"`` and ``"manual"`` containing
        lists of violations.
    """
    auto_fixable_types = {
        "silk_over_copper",
        "silk_overlap",
        "silk_edge_clearance",
        "courtyard_overlap",
        "courtyards_overlap",
    }

    result: dict[str, list[DRCViolation]] = {
        "auto_fixable": [],
        "manual": [],
    }

    for v in report.violations:
        if v.excluded:
            continue
        if v.type in auto_fixable_types:
            result["auto_fixable"].append(v)
        else:
            result["manual"].append(v)

    return result

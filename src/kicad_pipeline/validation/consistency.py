"""Schematic-PCB consistency validator.

Extracts components from generated ``.kicad_sch`` and ``.kicad_pcb`` files and
cross-validates that both contain the same set of components with matching
footprints.  This catches divergence caused by independent requirements
enrichment in the schematic and PCB builders.

The :func:`check_consistency` function is the main entry point — it returns a
:class:`ConsistencyReport` with DRC-style violations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.sexp.parser import parse_file
from kicad_pipeline.validation.drc import DRCViolation, Severity

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.sexp.writer import SExpNode

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchematicComponent:
    """A component extracted from a ``.kicad_sch`` file."""

    ref: str
    value: str
    footprint: str
    source_file: str


@dataclass(frozen=True)
class PCBComponent:
    """A component extracted from a ``.kicad_pcb`` file."""

    ref: str
    value: str
    lib_id: str


@dataclass(frozen=True)
class ConsistencyReport:
    """Result of a schematic-PCB consistency check."""

    violations: tuple[DRCViolation, ...]
    schematic_refs: tuple[str, ...]
    pcb_refs: tuple[str, ...]

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


# ---------------------------------------------------------------------------
# S-expression helpers
# ---------------------------------------------------------------------------


def _find_children(node: SExpNode, tag: str) -> list[list[SExpNode]]:
    """Find child lists whose first element matches *tag*.

    Args:
        node: An S-expression node (expected to be a list).
        tag: The tag string to match against each child's first element.

    Returns:
        List of matching child nodes (each a list).
    """
    if not isinstance(node, list):
        return []
    results: list[list[SExpNode]] = []
    for child in node:
        if isinstance(child, list) and len(child) > 0 and child[0] == tag:
            results.append(child)
    return results


def _get_property(node: list[SExpNode], name: str) -> str | None:
    """Extract the value of a named ``(property "name" "value" ...)`` child.

    Args:
        node: An S-expression list node containing property children.
        name: The property name to look for.

    Returns:
        The property value string, or ``None`` if not found.
    """
    for child in node:
        if (
            isinstance(child, list)
            and len(child) >= 3
            and child[0] == "property"
            and child[1] == name
        ):
            val = child[2]
            return str(val) if val is not None else None
    return None


# ---------------------------------------------------------------------------
# Extraction: schematic
# ---------------------------------------------------------------------------


def extract_schematic_components(
    sch_path: Path,
) -> tuple[SchematicComponent, ...]:
    """Extract placed components from a single ``.kicad_sch`` file.

    Skips power symbols (lib_id containing ``"power:"``).

    Args:
        sch_path: Path to the ``.kicad_sch`` file.

    Returns:
        Tuple of extracted components.
    """
    tree = parse_file(sch_path)
    if not isinstance(tree, list):
        return ()

    components: list[SchematicComponent] = []
    source = str(sch_path.name)

    for child in tree:
        if not isinstance(child, list) or len(child) < 2:
            continue
        if child[0] != "symbol":
            continue

        # Find lib_id
        lib_id_nodes = _find_children(child, "lib_id")
        if not lib_id_nodes:
            continue
        lib_id = str(lib_id_nodes[0][1]) if len(lib_id_nodes[0]) > 1 else ""

        # Skip power symbols
        if "power:" in lib_id.lower():
            continue

        ref = _get_property(child, "Reference")
        value = _get_property(child, "Value")
        footprint = _get_property(child, "Footprint")

        if ref is None:
            continue

        # Skip virtual ref designators (e.g. "#PWR01")
        if ref.startswith("#"):
            continue

        components.append(
            SchematicComponent(
                ref=ref,
                value=value or "",
                footprint=footprint or "",
                source_file=source,
            )
        )

    return tuple(components)


def extract_schematic_components_recursive(
    root_sch_path: Path,
) -> tuple[SchematicComponent, ...]:
    """Extract components from a root schematic and all sub-sheets.

    Args:
        root_sch_path: Path to the root ``.kicad_sch`` file.

    Returns:
        Tuple of all extracted components across all sheets.
    """
    all_components: list[SchematicComponent] = []
    visited: set[str] = set()

    def _walk(sch_path: Path) -> None:
        resolved = str(sch_path.resolve())
        if resolved in visited:
            return
        visited.add(resolved)

        all_components.extend(extract_schematic_components(sch_path))

        # Find sub-sheet references
        tree = parse_file(sch_path)
        if not isinstance(tree, list):
            return
        for child in tree:
            if not isinstance(child, list) or len(child) < 2:
                continue
            if child[0] != "sheet":
                continue
            sheet_file = _get_property(child, "Sheetfile")
            if sheet_file:
                sub_path = sch_path.parent / sheet_file
                if sub_path.exists():
                    _walk(sub_path)

    _walk(root_sch_path)
    return tuple(all_components)


# ---------------------------------------------------------------------------
# Extraction: PCB
# ---------------------------------------------------------------------------


def extract_pcb_components(pcb_path: Path) -> tuple[PCBComponent, ...]:
    """Extract footprint components from a ``.kicad_pcb`` file.

    Args:
        pcb_path: Path to the ``.kicad_pcb`` file.

    Returns:
        Tuple of extracted PCB components.
    """
    tree = parse_file(pcb_path)
    if not isinstance(tree, list):
        return ()

    components: list[PCBComponent] = []

    for child in tree:
        if not isinstance(child, list) or len(child) < 2:
            continue
        if child[0] != "footprint":
            continue

        lib_id = str(child[1])
        ref = _get_property(child, "Reference")
        value = _get_property(child, "Value")

        if ref is None:
            continue

        components.append(
            PCBComponent(
                ref=ref,
                value=value or "",
                lib_id=lib_id,
            )
        )

    return tuple(components)


# ---------------------------------------------------------------------------
# Footprint normalization
# ---------------------------------------------------------------------------


def normalize_footprint(fp: str) -> str:
    """Strip library prefix from a footprint name.

    ``"Resistor_SMD:R_0805_2012Metric"`` becomes ``"R_0805_2012Metric"``.

    Args:
        fp: Footprint string, possibly with library prefix.

    Returns:
        The footprint name without its library prefix.
    """
    if ":" in fp:
        return fp.split(":", 1)[1]
    return fp


def footprints_match(sch_fp: str, pcb_fp: str) -> bool:
    """Compare footprints after normalization.

    Handles prefix matching: ``"R_0805"`` matches ``"R_0805_2012Metric"``
    (one is a prefix of the other).

    Args:
        sch_fp: Footprint from the schematic.
        pcb_fp: Footprint from the PCB.

    Returns:
        True if the footprints are considered matching.
    """
    norm_sch = normalize_footprint(sch_fp)
    norm_pcb = normalize_footprint(pcb_fp)

    if norm_sch == norm_pcb:
        return True

    # Prefix match: one is prefix of the other
    return norm_sch.startswith(norm_pcb) or norm_pcb.startswith(norm_sch)


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def check_consistency(
    sch_path: Path,
    pcb_path: Path,
) -> ConsistencyReport:
    """Cross-validate schematic and PCB component lists.

    Checks for:
    - Components in schematic but missing from PCB (ERROR)
    - Components in PCB but missing from schematic (ERROR)
    - Same ref with different footprint (ERROR)
    - Same ref with different value (WARNING)

    Args:
        sch_path: Path to the root ``.kicad_sch`` file.
        pcb_path: Path to the ``.kicad_pcb`` file.

    Returns:
        A :class:`ConsistencyReport` with all violations found.
    """
    sch_comps = extract_schematic_components_recursive(sch_path)
    pcb_comps = extract_pcb_components(pcb_path)

    sch_by_ref: dict[str, SchematicComponent] = {}
    for comp in sch_comps:
        # Keep first occurrence (in case of duplicates across sheets)
        if comp.ref not in sch_by_ref:
            sch_by_ref[comp.ref] = comp

    pcb_by_ref: dict[str, PCBComponent] = {}
    for pcb_comp in pcb_comps:
        if pcb_comp.ref not in pcb_by_ref:
            pcb_by_ref[pcb_comp.ref] = pcb_comp

    violations: list[DRCViolation] = []

    # Components in schematic but not PCB
    for ref in sorted(sch_by_ref):
        if ref not in pcb_by_ref:
            sch_comp = sch_by_ref[ref]
            violations.append(
                DRCViolation(
                    rule="consistency_missing_in_pcb",
                    message=(
                        f"{ref} ({sch_comp.value}) present in schematic "
                        f"({sch_comp.source_file}) but missing from PCB"
                    ),
                    severity=Severity.ERROR,
                    ref=ref,
                )
            )

    # Components in PCB but not schematic
    # Skip mechanical-only components (mounting holes) which are added by the
    # PCB builder and have no schematic representation.
    _MECHANICAL_VALUES = {"MountingHole", "MountingHole_Pad"}
    for ref in sorted(pcb_by_ref):
        if ref not in sch_by_ref:
            pcb_comp = pcb_by_ref[ref]
            if pcb_comp.value in _MECHANICAL_VALUES:
                continue
            violations.append(
                DRCViolation(
                    rule="consistency_missing_in_schematic",
                    message=(
                        f"{ref} ({pcb_comp.value}) present in PCB "
                        f"but missing from schematic"
                    ),
                    severity=Severity.ERROR,
                    ref=ref,
                )
            )

    # Check matched refs for footprint / value mismatches
    for ref in sorted(set(sch_by_ref) & set(pcb_by_ref)):
        sch_comp = sch_by_ref[ref]
        pcb_comp = pcb_by_ref[ref]

        # Footprint mismatch
        if (
            sch_comp.footprint
            and pcb_comp.lib_id
            and not footprints_match(sch_comp.footprint, pcb_comp.lib_id)
        ):
                violations.append(
                    DRCViolation(
                        rule="consistency_footprint_mismatch",
                        message=(
                            f"{ref} footprint mismatch: schematic has "
                            f"'{sch_comp.footprint}', PCB has '{pcb_comp.lib_id}'"
                        ),
                        severity=Severity.ERROR,
                        ref=ref,
                    )
                )

        # Value mismatch
        if sch_comp.value and pcb_comp.value and sch_comp.value != pcb_comp.value:
                violations.append(
                    DRCViolation(
                        rule="consistency_value_mismatch",
                        message=(
                            f"{ref} value mismatch: schematic has "
                            f"'{sch_comp.value}', PCB has '{pcb_comp.value}'"
                        ),
                        severity=Severity.WARNING,
                        ref=ref,
                    )
                )

    sch_refs = tuple(sorted(sch_by_ref))
    pcb_refs = tuple(sorted(pcb_by_ref))

    return ConsistencyReport(
        violations=tuple(violations),
        schematic_refs=sch_refs,
        pcb_refs=pcb_refs,
    )


# ---------------------------------------------------------------------------
# Requirements hash
# ---------------------------------------------------------------------------


def compute_requirements_hash(req_path: Path) -> str:
    """Compute a SHA-256 hash of the requirements file content.

    The content is normalized (sorted keys, no whitespace) before hashing
    to ensure deterministic results regardless of formatting.

    Args:
        req_path: Path to ``requirements.json``.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = req_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    normalized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def check_requirements_hash(
    stored_hash: str,
    req_path: Path,
) -> DRCViolation | None:
    """Check if requirements have changed since the stored hash was computed.

    Args:
        stored_hash: The previously stored hash value.
        req_path: Path to the current ``requirements.json``.

    Returns:
        A WARNING violation if the hash differs, ``None`` if unchanged.
    """
    current_hash = compute_requirements_hash(req_path)
    if current_hash != stored_hash:
        return DRCViolation(
            rule="requirements_changed",
            message=(
                "requirements.json has changed since the schematic was generated "
                "(hash mismatch) — schematic and PCB may be out of sync"
            ),
            severity=Severity.WARNING,
        )
    return None


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def consistency_report_to_text(report: ConsistencyReport) -> str:
    """Format a :class:`ConsistencyReport` as human-readable text.

    Args:
        report: The consistency report to format.

    Returns:
        Multi-line text report string.
    """
    lines: list[str] = []
    lines.append("=== Schematic-PCB Consistency Report ===")
    lines.append("")
    lines.append(
        f"Schematic components: {len(report.schematic_refs)}"
    )
    lines.append(f"PCB components: {len(report.pcb_refs)}")
    lines.append(f"Errors: {len(report.errors)}")
    lines.append(f"Warnings: {len(report.warnings)}")
    lines.append(f"Result: {'PASS' if report.passed else 'FAIL'}")
    lines.append("")

    if report.violations:
        lines.append("--- Violations ---")
        for v in report.violations:
            severity = v.severity.value.upper()
            lines.append(f"[{severity}] {v.rule}: {v.message}")
        lines.append("")

    return "\n".join(lines)

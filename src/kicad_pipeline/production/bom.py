"""BOM (Bill of Materials) generator."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.requirements.component_db import ComponentDB, JLCPCBPart


@dataclass(frozen=True)
class BOMRow:
    """A single BOM line item (may cover multiple identical components)."""

    comment: str
    designator: str  # "R1 R2 R5" (space-separated)
    footprint: str
    lcsc: str
    quantity: int
    unit_price_usd: float = 0.0


def _extract_package(footprint: str) -> str:
    """Extract package size from footprint string (e.g. "R_0805_2012Metric" -> "0805")."""
    match = re.search(r"(\d{4})", footprint)
    return match.group(1) if match else "0805"


def _ref_prefix(ref: str) -> str:
    """Extract alphabetic prefix from reference designator ("R1" -> "R")."""
    match = re.match(r"[A-Za-z]+", ref)
    return match.group(0).upper() if match else ""


def _lookup_by_value(
    db: ComponentDB,
    value: str,
    package: str,
    prefix: str,
    parse_resistance: Callable[[str], float | None],
    parse_capacitance: Callable[[str], float | None],
) -> JLCPCBPart | None:
    """Look up a part by parsed value + package in ComponentDB."""
    if prefix == "R":
        ohms = parse_resistance(value)
        if ohms is not None:
            return db.find_resistor(ohms, package)
    elif prefix == "C":
        uf = parse_capacitance(value)
        if uf is not None:
            return db.find_capacitor(uf, package)
    return None


def generate_bom(
    pcb: PCBDesign,
    requirements: ProjectRequirements | None = None,
) -> tuple[BOMRow, ...]:
    """Group footprints by (value, lib_id, lcsc) and produce BOM rows."""
    # Group: key -> list of refs
    groups: dict[tuple[str, str, str], list[str]] = {}
    group_fp: dict[tuple[str, str, str], str] = {}

    for fp in pcb.footprints:
        lcsc_val = fp.lcsc if fp.lcsc is not None else ""
        key = (fp.value, fp.lib_id, lcsc_val)
        if key not in groups:
            groups[key] = []
            # footprint display: part after ":" if present
            if ":" in fp.lib_id:
                group_fp[key] = fp.lib_id.split(":")[-1]
            else:
                group_fp[key] = fp.lib_id
        groups[key].append(fp.ref)

    # Optionally enrich from ComponentDB
    db: ComponentDB | None = None
    if requirements is not None:
        try:
            from kicad_pipeline.requirements import component_db as _cdb_mod
            from kicad_pipeline.requirements.component_db import (
                _parse_capacitance_uf,
                _parse_resistance_ohms,
            )

            db = _cdb_mod.ComponentDB()
        except Exception:
            db = None

    rows: list[BOMRow] = []
    for key, refs in groups.items():
        value, _lib_id, lcsc_val = key
        sorted_refs = sorted(refs)
        designator = " ".join(sorted_refs)
        price: float = 0.0

        # Enrich missing LCSC / price from ComponentDB
        if db is not None:
            if lcsc_val:
                local = db.find_by_lcsc(lcsc_val)
                if local is not None:
                    price = local.price_usd
            else:
                # Try to find by value and package
                package = _extract_package(group_fp[key])
                prefix = _ref_prefix(sorted_refs[0]) if sorted_refs else ""
                found_part = _lookup_by_value(
                    db, value, package, prefix,
                    _parse_resistance_ohms, _parse_capacitance_uf,
                )
                if found_part is not None:
                    lcsc_val = found_part.lcsc
                    price = found_part.price_usd

        rows.append(
            BOMRow(
                comment=value,
                designator=designator,
                footprint=group_fp[key],
                lcsc=lcsc_val,
                quantity=len(refs),
                unit_price_usd=price,
            )
        )

    return tuple(rows)


def bom_to_csv(rows: tuple[BOMRow, ...]) -> str:
    """Return JLCPCB-format CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Comment", "Designator", "Footprint", "LCSC Part Number"])
    for row in rows:
        writer.writerow([row.comment, row.designator, row.footprint, row.lcsc])
    return output.getvalue()


def write_bom(rows: tuple[BOMRow, ...], path: str | Path) -> None:
    """Write CSV to file."""
    Path(path).write_text(bom_to_csv(rows), encoding="utf-8")

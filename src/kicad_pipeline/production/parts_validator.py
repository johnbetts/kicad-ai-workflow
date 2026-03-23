"""Parts validation orchestrator with 4-tier checking."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from kicad_pipeline.constants import JLCPCB_MIN_STOCK_QTY, JLCPCB_PARTS_SEARCH_URL
from kicad_pipeline.production.lcsc_client import LCSCStockInfo, fetch_lcsc_stock

if TYPE_CHECKING:
    from kicad_pipeline.production.bom import BOMRow
    from kicad_pipeline.requirements.component_db import ComponentDB

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartStatus:
    """Validation result for a single BOM line item."""

    lcsc: str
    ref_designators: tuple[str, ...]
    comment: str
    footprint: str
    tier: int
    status: str  # "ok" | "replaced" | "unavailable" | "missing_lcsc"
    in_stock: bool
    stock_qty: int | None
    unit_price_usd: float | None
    replacement_lcsc: str | None
    replacement_reason: str | None
    manual_url: str | None


@dataclass(frozen=True)
class PartsValidationReport:
    """Complete parts validation report."""

    project_name: str
    timestamp: str
    parts: tuple[PartStatus, ...]
    total_bom_cost_usd: float | None
    all_parts_available: bool
    unresolved_count: int
    summary_text: str
    low_stock_count: int = 0


def _extract_package(footprint: str) -> str:
    """Extract package size from footprint string.

    Examples: "R_0805_2012Metric" -> "0805", "C_0402" -> "0402"
    """
    match = re.search(r"(\d{4})", footprint)
    return match.group(1) if match else ""


def _ref_category(ref_designators: tuple[str, ...]) -> str:
    """Determine component category from reference designator prefix."""
    if not ref_designators:
        return ""
    prefix = re.match(r"[A-Za-z]+", ref_designators[0])
    if not prefix:
        return ""
    p = prefix.group(0).upper()
    categories: dict[str, str] = {
        "R": "resistor",
        "C": "capacitor",
        "D": "diode",
        "U": "ic",
        "L": "inductor",
        "J": "connector",
        "SW": "switch",
    }
    return categories.get(p, "")


def _find_replacement(
    comment: str,
    footprint: str,
    ref_designators: tuple[str, ...],
    db: ComponentDB,
) -> tuple[str | None, str | None]:
    """Try to find a replacement part from ComponentDB (Tier 3).

    Returns (lcsc, reason) or (None, None).
    """
    from kicad_pipeline.requirements.component_db import (
        _parse_capacitance_uf,
        _parse_resistance_ohms,
    )

    category = _ref_category(ref_designators)
    package = _extract_package(footprint)
    if not package:
        package = "0805"

    if category == "resistor":
        parsed = _parse_resistance_ohms(comment)
        if parsed is None:
            return None, None
        part = db.find_resistor(parsed, package)
        if part is not None:
            return part.lcsc, f"auto-replacement: {part.value} {part.package}"
    elif category == "capacitor":
        parsed = _parse_capacitance_uf(comment)
        if parsed is None:
            return None, None
        part = db.find_capacitor(parsed, package)
        if part is not None:
            return part.lcsc, f"auto-replacement: {part.value} {part.package}"
    elif category == "diode":
        for color in ("green", "red", "blue", "yellow"):
            part = db.find_led(color, package)
            if part is not None:
                return part.lcsc, f"auto-replacement: {part.mfr} LED {color}"

    return None, None


def _make_search_url(comment: str, footprint: str) -> str:
    """Build JLCPCB parts search URL for manual resolution."""
    query = f"{comment} {footprint}".strip()
    encoded = query.replace(" ", "+")
    return f"{JLCPCB_PARTS_SEARCH_URL}{encoded}"


class _ValidationAccumulator:
    """Mutable accumulator for parts validation results."""

    __slots__ = ("all_available", "low_stock", "parts", "total_cost", "unresolved")

    def __init__(self) -> None:
        self.parts: list[PartStatus] = []
        self.total_cost: float = 0.0
        self.all_available: bool = True
        self.unresolved: int = 0
        self.low_stock: int = 0


def _try_tier1_local(
    lcsc: str | None,
    row: BOMRow,
    refs: tuple[str, ...],
    db: ComponentDB | None,
    acc: _ValidationAccumulator,
) -> bool:
    """Tier 1: Local ComponentDB lookup. Returns True if resolved."""
    if db is None or not lcsc:
        return False
    local_part = db.find_by_lcsc(lcsc)
    if local_part is None or not local_part.in_stock:
        return False
    price = local_part.price_usd
    acc.total_cost += price * row.quantity
    acc.parts.append(PartStatus(
        lcsc=lcsc, ref_designators=refs, comment=row.comment,
        footprint=row.footprint, tier=1, status="ok", in_stock=True,
        stock_qty=None, unit_price_usd=price,
        replacement_lcsc=None, replacement_reason=None, manual_url=None,
    ))
    return True


def _try_tier2_web(
    lcsc: str | None,
    row: BOMRow,
    refs: tuple[str, ...],
    web_stock: dict[str, LCSCStockInfo],
    acc: _ValidationAccumulator,
) -> bool:
    """Tier 2: Web stock check. Returns True if resolved."""
    if not lcsc or lcsc not in web_stock:
        return False
    info = web_stock[lcsc]
    if not info.in_stock:
        return False
    qty = info.stock_qty or 0
    if qty < JLCPCB_MIN_STOCK_QTY:
        acc.all_available = False
        acc.unresolved += 1
        acc.low_stock += 1
        acc.parts.append(PartStatus(
            lcsc=lcsc, ref_designators=refs, comment=row.comment,
            footprint=row.footprint, tier=2, status="low_stock", in_stock=True,
            stock_qty=info.stock_qty, unit_price_usd=info.unit_price_usd,
            replacement_lcsc=None, replacement_reason=None,
            manual_url=_make_search_url(row.comment, row.footprint),
        ))
        return True
    web_price = info.unit_price_usd
    if web_price is not None:
        acc.total_cost += web_price * row.quantity
    acc.parts.append(PartStatus(
        lcsc=lcsc, ref_designators=refs, comment=row.comment,
        footprint=row.footprint, tier=2, status="ok", in_stock=True,
        stock_qty=info.stock_qty, unit_price_usd=info.unit_price_usd,
        replacement_lcsc=None, replacement_reason=None, manual_url=None,
    ))
    return True


def _try_tier3_replacement(
    lcsc: str | None,
    row: BOMRow,
    refs: tuple[str, ...],
    db: ComponentDB | None,
    acc: _ValidationAccumulator,
) -> bool:
    """Tier 3: Auto-replacement from ComponentDB. Returns True if resolved."""
    if db is None:
        return False
    repl_lcsc, repl_reason = _find_replacement(
        row.comment, row.footprint, refs, db,
    )
    if repl_lcsc is None:
        return False
    repl_part = db.find_by_lcsc(repl_lcsc)
    repl_price = repl_part.price_usd if repl_part else 0.0
    acc.total_cost += repl_price * row.quantity
    acc.parts.append(PartStatus(
        lcsc=lcsc or "", ref_designators=refs, comment=row.comment,
        footprint=row.footprint, tier=3, status="replaced", in_stock=True,
        stock_qty=None, unit_price_usd=repl_price,
        replacement_lcsc=repl_lcsc, replacement_reason=repl_reason,
        manual_url=None,
    ))
    return True


def _add_tier4_manual(
    lcsc: str | None,
    row: BOMRow,
    refs: tuple[str, ...],
    web_stock: dict[str, LCSCStockInfo],
    acc: _ValidationAccumulator,
) -> None:
    """Tier 4: Manual resolution (fallback)."""
    acc.all_available = False
    acc.unresolved += 1
    status = "missing_lcsc" if not lcsc else "unavailable"
    acc.parts.append(PartStatus(
        lcsc=lcsc or "", ref_designators=refs, comment=row.comment,
        footprint=row.footprint, tier=4, status=status, in_stock=False,
        stock_qty=web_stock[lcsc].stock_qty if lcsc and lcsc in web_stock else None,
        unit_price_usd=None,
        replacement_lcsc=None, replacement_reason=None,
        manual_url=_make_search_url(row.comment, row.footprint),
    ))


def validate_bom_parts(
    bom_rows: tuple[BOMRow, ...],
    db: ComponentDB | None = None,
    check_web_stock: bool = True,
    timeout: float = 10.0,
    project_name: str = "project",
) -> PartsValidationReport:
    """Run 4-tier parts validation on BOM rows.

    Tier 1: Local ComponentDB lookup
    Tier 2: Web fetch LCSC stock API
    Tier 3: Auto-suggest replacement from ComponentDB
    Tier 4: Manual review with JLCPCB search URL
    """
    acc = _ValidationAccumulator()

    web_stock: dict[str, LCSCStockInfo] = {}
    if check_web_stock:
        lcsc_numbers = tuple(
            r.lcsc for r in bom_rows if r.lcsc and r.lcsc.startswith("C")
        )
        for lcsc in lcsc_numbers:
            if lcsc not in web_stock:
                info = fetch_lcsc_stock(lcsc, timeout=timeout)
                if info is not None:
                    web_stock[lcsc] = info

    for row in bom_rows:
        refs = tuple(row.designator.split())
        lcsc = row.lcsc
        if _try_tier1_local(lcsc, row, refs, db, acc):
            continue
        if _try_tier2_web(lcsc, row, refs, web_stock, acc):
            continue
        if _try_tier3_replacement(lcsc, row, refs, db, acc):
            continue
        _add_tier4_manual(lcsc, row, refs, web_stock, acc)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = _build_summary(
        acc.parts, acc.total_cost, acc.all_available,
        acc.unresolved, acc.low_stock,
    )

    return PartsValidationReport(
        project_name=project_name,
        timestamp=timestamp,
        parts=tuple(acc.parts),
        total_bom_cost_usd=acc.total_cost if acc.total_cost > 0.0 else None,
        all_parts_available=acc.all_available,
        unresolved_count=acc.unresolved,
        summary_text=summary,
        low_stock_count=acc.low_stock,
    )


def _build_summary(
    parts: list[PartStatus],
    total_cost: float,
    all_available: bool,
    unresolved: int,
    low_stock: int = 0,
) -> str:
    """Build human-readable summary text."""
    lines: list[str] = []
    lines.append(f"Total BOM lines: {len(parts)}")
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in parts:
        tier_counts[p.tier] += 1
    lines.append(f"  Tier 1 (local DB):     {tier_counts[1]}")
    lines.append(f"  Tier 2 (web stock):    {tier_counts[2]}")
    lines.append(f"  Tier 3 (auto-replace): {tier_counts[3]}")
    lines.append(f"  Tier 4 (manual):       {tier_counts[4]}")
    if low_stock > 0:
        lines.append(
            f"  Low stock (<{JLCPCB_MIN_STOCK_QTY}):  {low_stock}"
        )
    if total_cost > 0.0:
        lines.append(f"Estimated BOM cost: ${total_cost:.2f} USD")
    if all_available:
        lines.append("All parts available.")
    else:
        lines.append(f"ATTENTION: {unresolved} part(s) need manual resolution.")
        if low_stock > 0:
            lines.append(
                f"  {low_stock} part(s) have stock below "
                f"{JLCPCB_MIN_STOCK_QTY} — approval required."
            )
    return "\n".join(lines)


def report_to_text(report: PartsValidationReport) -> str:
    """Format report as human-readable text."""
    lines: list[str] = []
    lines.append(f"Parts Validation Report: {report.project_name}")
    lines.append(f"Generated: {report.timestamp}")
    lines.append("=" * 60)
    lines.append("")

    for p in report.parts:
        refs = ", ".join(p.ref_designators)
        status_label = p.status.upper()
        if p.status == "low_stock":
            status_label = f"LOW STOCK({p.stock_qty})"
        lines.append(f"  [{status_label:12s}] {refs:20s}  {p.comment:12s}  {p.footprint}")
        lines.append(f"               LCSC: {p.lcsc or '(none)'}  Tier: {p.tier}")
        if p.in_stock and p.unit_price_usd is not None:
            stock_str = str(p.stock_qty) if p.stock_qty else "n/a"
            lines.append(f"               Price: ${p.unit_price_usd:.4f}  Stock: {stock_str}")
        if p.replacement_lcsc:
            lines.append(
                f"               Replacement: {p.replacement_lcsc}"
                f" ({p.replacement_reason})"
            )
        if p.manual_url:
            lines.append(f"               Search: {p.manual_url}")
        lines.append("")

    lines.append("=" * 60)
    lines.append(report.summary_text)
    return "\n".join(lines)


def report_to_json(report: PartsValidationReport) -> str:
    """Serialize report to JSON string."""
    parts_list: list[dict[str, object]] = []
    for p in report.parts:
        parts_list.append({
            "lcsc": p.lcsc,
            "ref_designators": list(p.ref_designators),
            "comment": p.comment,
            "footprint": p.footprint,
            "tier": p.tier,
            "status": p.status,
            "in_stock": p.in_stock,
            "stock_qty": p.stock_qty,
            "unit_price_usd": p.unit_price_usd,
            "replacement_lcsc": p.replacement_lcsc,
            "replacement_reason": p.replacement_reason,
            "manual_url": p.manual_url,
        })

    obj: dict[str, object] = {
        "project_name": report.project_name,
        "timestamp": report.timestamp,
        "parts": parts_list,
        "total_bom_cost_usd": report.total_bom_cost_usd,
        "all_parts_available": report.all_parts_available,
        "unresolved_count": report.unresolved_count,
        "low_stock_count": report.low_stock_count,
        "summary_text": report.summary_text,
    }
    return json.dumps(obj, indent=2) + "\n"

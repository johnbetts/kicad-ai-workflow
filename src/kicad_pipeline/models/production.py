"""Data models for PCB production artifacts (Gerbers, BOM, CPL, cost)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GerberLayer:
    """A single Gerber file for one layer."""

    layer_name: str  # "F.Cu", "B.Cu", etc.
    filename: str  # "project-F_Cu.gbr"
    content: str  # full Gerber file text


@dataclass(frozen=True)
class DrillFile:
    """An Excellon drill file (PTH or NPTH)."""

    filename: str
    content: str
    is_npth: bool = False  # Non-plated through holes


@dataclass(frozen=True)
class BOMEntry:
    """A single BOM line item (may cover multiple identical components)."""

    comment: str  # "10k"
    designators: tuple[str, ...]  # ("R1", "R2", "R5")
    footprint: str  # "R_0805"
    lcsc: str | None = None
    quantity: int = 1
    unit_price_usd: float | None = None
    in_stock: bool = True


@dataclass(frozen=True)
class BOM:
    """Complete bill of materials."""

    entries: tuple[BOMEntry, ...]
    project_name: str
    revision: str = "v0.1"

    @property
    def total_cost_usd(self) -> float | None:
        """Total estimated BOM cost, or None if any price unknown."""
        total = 0.0
        for e in self.entries:
            if e.unit_price_usd is None:
                return None
            total += e.unit_price_usd * e.quantity
        return total


@dataclass(frozen=True)
class CPLEntry:
    """A pick-and-place entry for a single component."""

    designator: str  # "R1"
    value: str  # "10k"
    package: str  # "0805"
    mid_x: float  # mm
    mid_y: float  # mm
    rotation: float  # degrees (JLCPCB-corrected)
    layer: str  # "top" or "bottom"


@dataclass(frozen=True)
class CPL:
    """Complete pick-and-place / component placement list."""

    entries: tuple[CPLEntry, ...]
    project_name: str


@dataclass(frozen=True)
class CostEstimate:
    """Manufacturing cost estimate for different quantities."""

    bom_cost_usd: float
    pcb_cost_5_usd: float
    pcb_cost_10_usd: float
    pcb_cost_50_usd: float
    assembly_cost_5_usd: float | None = None
    assembly_cost_10_usd: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductionPackage:
    """Complete production artifact bundle."""

    project_name: str
    revision: str
    gerbers: tuple[GerberLayer, ...]
    drill_files: tuple[DrillFile, ...]
    bom: BOM
    cpl: CPL
    cost_estimate: CostEstimate | None = None
    order_guide: str = ""  # markdown order guide text

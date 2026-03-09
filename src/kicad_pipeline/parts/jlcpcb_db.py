"""Adapter for the Bouldini kicad-jlcpcb-tools FTS5 SQLite database.

Wraps the ~7M-part JLCPCB parts database for fast full-text search.
Auto-discovers the database at the standard KiCad 9 plugin path;
falls back to the ``JLCPCB_PARTS_DB`` environment variable.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from kicad_pipeline.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Standard install paths for Bouldini's plugin database (macOS / Linux / Windows).
_DB_SEARCH_PATHS: tuple[Path, ...] = (
    Path.home()
    / "Documents"
    / "KiCad"
    / "10.0"
    / "3rdparty"
    / "plugins"
    / "com_github_bouni_kicad-jlcpcb-tools"
    / "jlcpcb"
    / "parts-fts5.db",
    Path.home()
    / "Documents"
    / "KiCad"
    / "9.0"
    / "3rdparty"
    / "plugins"
    / "com_github_bouni_kicad-jlcpcb-tools"
    / "jlcpcb"
    / "parts-fts5.db",
    Path.home()
    / ".kicad"
    / "10.0"
    / "3rdparty"
    / "plugins"
    / "com_github_bouni_kicad-jlcpcb-tools"
    / "jlcpcb"
    / "parts-fts5.db",
    Path.home()
    / ".kicad"
    / "9.0"
    / "3rdparty"
    / "plugins"
    / "com_github_bouni_kicad-jlcpcb-tools"
    / "jlcpcb"
    / "parts-fts5.db",
    Path.home()
    / ".local"
    / "share"
    / "kicad"
    / "10.0"
    / "3rdparty"
    / "plugins"
    / "com_github_bouni_kicad-jlcpcb-tools"
    / "jlcpcb"
    / "parts-fts5.db",
    Path.home()
    / ".local"
    / "share"
    / "kicad"
    / "9.0"
    / "3rdparty"
    / "plugins"
    / "com_github_bouni_kicad-jlcpcb-tools"
    / "jlcpcb"
    / "parts-fts5.db",
)


@dataclass(frozen=True)
class JLCPCBPart:
    """A part from the JLCPCB parts database."""

    lcsc: str
    mfr: str
    mfr_part: str
    description: str
    package: str
    category: str
    subcategory: str
    solder_joints: int
    stock: int
    price: float | None
    basic: bool

    @property
    def is_in_stock(self) -> bool:
        """Return True if the part has stock available."""
        return self.stock > 0


def _parse_price(price_str: str) -> float | None:
    """Extract the unit price from JLCPCB's price-break string.

    Format: ``"1-9:0.0123,10-99:0.0098,100-:0.0078"``
    Returns the first (unit) price, or None if unparseable.
    """
    if not price_str or not price_str.strip():
        return None
    try:
        first_break = price_str.split(",")[0]
        return float(first_break.split(":")[1])
    except (IndexError, ValueError):
        return None


def _parse_stock(stock_str: str) -> int:
    """Parse stock value from string, returning 0 on failure."""
    try:
        return int(stock_str)
    except (ValueError, TypeError):
        return 0


def _parse_solder_joints(joints_str: str) -> int:
    """Parse solder joints from string, returning 0 on failure."""
    try:
        return int(joints_str)
    except (ValueError, TypeError):
        return 0


def _row_to_part(row: tuple[str, ...]) -> JLCPCBPart:
    """Convert a database row to a JLCPCBPart.

    Expected column order matches the SELECT in ``_SEARCH_SQL``.
    """
    return JLCPCBPart(
        lcsc=str(row[0]),
        mfr_part=str(row[1]),
        package=str(row[2]),
        solder_joints=_parse_solder_joints(str(row[3])),
        basic=str(row[4]) == "Basic",
        stock=_parse_stock(str(row[5])),
        mfr=str(row[6]),
        description=str(row[7]),
        price=_parse_price(str(row[8])),
        category=str(row[9]),
        subcategory=str(row[10]) if len(row) > 10 else "",
    )


# Column selection — order must match _row_to_part.
_SELECT_COLS = (
    '"LCSC Part", "MFR.Part", "Package", "Solder Joint", '
    '"Library Type", "Stock", "Manufacturer", "Description", '
    '"Price", "First Category", "Second Category"'
)


def discover_db_path() -> Path:
    """Find the JLCPCB parts FTS5 database.

    Checks ``JLCPCB_PARTS_DB`` env var first, then standard install paths.

    Raises:
        ConfigurationError: If no database file can be found.
    """
    env_path = os.environ.get("JLCPCB_PARTS_DB")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        msg = f"JLCPCB_PARTS_DB points to non-existent file: {env_path}"
        raise ConfigurationError(msg)

    for candidate in _DB_SEARCH_PATHS:
        if candidate.is_file():
            logger.debug("Found JLCPCB parts DB at %s", candidate)
            return candidate

    msg = (
        "Cannot find JLCPCB parts-fts5.db. Install the Bouldini "
        "kicad-jlcpcb-tools plugin or set JLCPCB_PARTS_DB env var."
    )
    raise ConfigurationError(msg)


class JLCPCBPartsDB:
    """Query interface for the JLCPCB parts FTS5 database."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Open the parts database.

        Args:
            db_path: Explicit path to parts-fts5.db.  If ``None``,
                auto-discovers via :func:`discover_db_path`.
        """
        resolved = Path(db_path) if db_path else discover_db_path()
        self._db_path = resolved
        self._conn = sqlite3.connect(
            f"file:{resolved}?mode=ro",
            uri=True,
            check_same_thread=False,
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> JLCPCBPartsDB:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------

    def search_parts(
        self,
        query: str,
        category: str | None = None,
        basic_only: bool = False,
        in_stock: bool = False,
        limit: int = 20,
    ) -> list[JLCPCBPart]:
        """Full-text search across the parts database.

        Args:
            query: FTS5 search string (e.g. ``"10k 0805 resistor"``).
            category: Restrict to a ``First Category`` value.
            basic_only: Only return JLCPCB basic parts.
            in_stock: Only return parts with stock > 0.
            limit: Maximum results to return.

        Returns:
            Matching parts sorted by relevance.
        """
        match_clauses: list[str] = []
        # Sanitise query: remove FTS5 special chars that break quoting.
        sanitised = re.sub(r'["\'\*\(\)]', " ", query).strip()
        if not sanitised:
            return []
        # Add prefix wildcard to each token for partial matching.
        # Quote tokens to handle special chars like hyphens.
        tokens = sanitised.split()
        term_expr = " AND ".join(f'"{t}"*' for t in tokens)
        match_clauses.append(term_expr)

        if category:
            safe_cat = category.replace('"', "")
            match_clauses.append(f'"First Category":"{safe_cat}"')

        if basic_only:
            match_clauses.append('"Library Type":"Basic"')

        match_expr = " AND ".join(match_clauses)
        sql = f"SELECT {_SELECT_COLS} FROM parts WHERE parts MATCH ?"
        params: list[str | int] = [match_expr]

        if in_stock:
            sql += ' AND CAST("Stock" AS INTEGER) > 0'

        sql += " LIMIT ?"
        params.append(limit)

        try:
            cursor = self._conn.execute(sql, params)
            return [_row_to_part(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            logger.warning("FTS5 query failed for: %s", match_expr, exc_info=True)
            return []

    def get_part(self, lcsc: str) -> JLCPCBPart | None:
        """Look up a specific part by LCSC number.

        Args:
            lcsc: LCSC part number (e.g. ``"C25804"``).

        Returns:
            The matching part, or ``None`` if not found.
        """
        if not lcsc.startswith("C"):
            lcsc = f"C{lcsc}"
        sql = (
            f"SELECT {_SELECT_COLS} FROM parts "
            'WHERE parts MATCH \'"LCSC Part":"\' || ? || \'"\' LIMIT 1'
        )
        try:
            cursor = self._conn.execute(sql, [lcsc])
            row = cursor.fetchone()
            return _row_to_part(row) if row else None
        except sqlite3.OperationalError:
            logger.warning("Part lookup failed for %s", lcsc, exc_info=True)
            return None

    def find_resistor(
        self,
        value: str,
        package: str = "0805",
        basic_only: bool = True,
    ) -> list[JLCPCBPart]:
        """Search for resistors by value and package.

        Args:
            value: Resistance value (e.g. ``"10k"``, ``"4.7k"``, ``"100"``).
            package: Package size (e.g. ``"0805"``, ``"0603"``).
            basic_only: Only return JLCPCB basic parts.

        Returns:
            Matching resistors.
        """
        return self.search_parts(
            f"{value} {package}",
            category="Resistors",
            basic_only=basic_only,
            in_stock=True,
        )

    def find_capacitor(
        self,
        value: str,
        package: str = "0805",
        basic_only: bool = True,
    ) -> list[JLCPCBPart]:
        """Search for capacitors by value and package.

        Args:
            value: Capacitance value (e.g. ``"100nF"``, ``"10uF"``).
            package: Package size (e.g. ``"0805"``, ``"0603"``).
            basic_only: Only return JLCPCB basic parts.

        Returns:
            Matching capacitors.
        """
        return self.search_parts(
            f"{value} {package}",
            category="Capacitors",
            basic_only=basic_only,
            in_stock=True,
        )

    def find_ic(
        self,
        mfr_part: str,
        basic_only: bool = False,
    ) -> list[JLCPCBPart]:
        """Search for an IC by manufacturer part number.

        Args:
            mfr_part: Full or partial manufacturer part number.
            basic_only: Only return JLCPCB basic parts.

        Returns:
            Matching ICs/components.
        """
        return self.search_parts(mfr_part, basic_only=basic_only, in_stock=True)

    def find_by_category(
        self,
        category: str,
        subcategory: str | None = None,
        basic_only: bool = False,
        limit: int = 50,
    ) -> list[JLCPCBPart]:
        """Browse parts by category.

        Args:
            category: First-level category (e.g. ``"Resistors"``).
            subcategory: Optional second-level category filter.
            basic_only: Only return JLCPCB basic parts.
            limit: Maximum results.

        Returns:
            Parts in the given category.
        """
        query = category
        if subcategory:
            query += f" {subcategory}"
        return self.search_parts(
            query, category=category, basic_only=basic_only, limit=limit
        )

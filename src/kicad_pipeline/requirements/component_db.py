"""JLCPCB basic-parts component database: load and query utilities.

The database is backed by ``data/jlcpcb_basic_parts.json`` which ships with
the project.  An optional ``data/e_series.json`` file provides standard
E-series resistor / capacitor values used by :func:`nearest_e_series_value`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Default path to the bundled data file
_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
JLCPCB_PARTS_FILE = _DATA_DIR / "jlcpcb_basic_parts.json"
E_SERIES_FILE = _DATA_DIR / "e_series.json"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JLCPCBPart:
    """A single JLCPCB basic parts entry."""

    lcsc: str
    mfr: str
    value: str
    package: str
    category: str
    price_usd: float
    # Optional fields (some parts have them):
    voltage: str | None = None  # "50V"
    dielectric: str | None = None  # "X5R"
    vceo: str | None = None  # transistor collector-emitter voltage
    ic: str | None = None  # transistor collector current
    vr: str | None = None  # diode reverse voltage
    if_: str | None = None  # diode forward current (if_ to avoid keyword clash)
    vout: float | None = None  # LDO output voltage
    iout_ma: float | None = None  # LDO output current in mA
    vin_max: float | None = None  # LDO max input voltage
    vf: float | None = None  # LED forward voltage
    in_stock: bool = True


@dataclass(frozen=True)
class ESeries:
    """E-series resistor/capacitor standard values."""

    E6: tuple[float, ...]
    E12: tuple[float, ...]
    E24: tuple[float, ...]
    E96: tuple[float, ...]


# ---------------------------------------------------------------------------
# Value-string parsing helpers
# ---------------------------------------------------------------------------


def _parse_resistance_ohms(value_str: str) -> float | None:
    """Parse resistor value string to ohms.

    Examples:
        ``'10k'`` → ``10000.0``,
        ``'100R'`` → ``100.0``,
        ``'4.7k'`` → ``4700.0``,
        ``'2.2k'`` → ``2200.0``.

    Args:
        value_str: Raw value string from the parts database.

    Returns:
        Resistance in ohms, or ``None`` if the string cannot be parsed.
    """
    s = value_str.strip()
    # Match optional decimal number followed by an optional multiplier suffix
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kKmMrR]?)", s)
    if not m:
        return None
    mantissa = float(m.group(1))
    suffix = m.group(2).lower()
    multipliers: dict[str, float] = {"k": 1_000.0, "m": 1_000_000.0, "r": 1.0, "": 1.0}
    return mantissa * multipliers[suffix]


def _parse_capacitance_uf(value_str: str) -> float | None:
    """Parse capacitor value string to microfarads.

    Examples:
        ``'100nF'`` → ``0.1``,
        ``'10uF'`` → ``10.0``,
        ``'1uF'`` → ``1.0``,
        ``'22pF'`` → ``0.000022``.

    Args:
        value_str: Raw value string from the parts database.

    Returns:
        Capacitance in microfarads, or ``None`` if unparsable.
    """
    s = value_str.strip()
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([pPnNuUmM])[fF]?", s)
    if not m:
        return None
    mantissa = float(m.group(1))
    suffix = m.group(2).lower()
    # Convert everything to µF
    to_uf: dict[str, float] = {
        "p": 1e-6,   # pF → µF
        "n": 1e-3,   # nF → µF
        "u": 1.0,    # µF → µF
        "m": 1e6,    # mF → µF  (rare but defined)
    }
    return mantissa * to_uf[suffix]


def _parse_voltage_v(voltage_str: str) -> float | None:
    """Parse voltage string to volts.

    Examples:
        ``'50V'`` → ``50.0``,
        ``'10V'`` → ``10.0``.

    Args:
        voltage_str: Raw voltage string from the parts database.

    Returns:
        Voltage in volts, or ``None`` if unparsable.
    """
    s = voltage_str.strip()
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*[vV]", s)
    if not m:
        return None
    return float(m.group(1))


# ---------------------------------------------------------------------------
# E-series helpers
# ---------------------------------------------------------------------------


def load_e_series(file: Path = E_SERIES_FILE) -> ESeries:
    """Load E-series values from a JSON file.

    Args:
        file: Path to the JSON file (defaults to :data:`E_SERIES_FILE`).

    Returns:
        An :class:`ESeries` dataclass with E6, E12, E24, and E96 values.

    Raises:
        FileNotFoundError: If *file* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    log.debug("Loading E-series data from %s", file)
    with file.open(encoding="utf-8") as fh:
        raw: dict[str, list[float]] = json.load(fh)
    return ESeries(
        E6=tuple(raw["E6"]),
        E12=tuple(raw["E12"]),
        E24=tuple(raw["E24"]),
        E96=tuple(raw["E96"]),
    )


def nearest_e_series_value(
    target: float,
    series: str = "E24",
    e: ESeries | None = None,
) -> float:
    """Return the nearest E-series standard value to *target*.

    Searches across all decades from 1 Ω to 10 MΩ (factors of 10).

    Args:
        target: The target value (e.g. in ohms or farads — units are agnostic).
        series: Which E-series to use: ``'E6'``, ``'E12'``, ``'E24'``,
                or ``'E96'`` (default ``'E24'``).
        e: Pre-loaded :class:`ESeries` instance.  Loaded from disk if omitted.

    Returns:
        The nearest standard value in the same units as *target*.

    Raises:
        ValueError: If *series* is not a recognised E-series name.
    """
    if e is None:
        e = load_e_series()

    series_map: dict[str, tuple[float, ...]] = {
        "E6": e.E6,
        "E12": e.E12,
        "E24": e.E24,
        "E96": e.E96,
    }
    if series not in series_map:
        raise ValueError(f"Unknown E-series '{series}'. Choose from {list(series_map)}")

    base_values = series_map[series]

    # Generate all candidate values across decades 1e0 … 1e7
    candidates: list[float] = []
    for decade_exp in range(8):  # 1, 10, 100, … 10_000_000
        decade = 10.0 ** decade_exp
        candidates.extend(v * decade for v in base_values)

    best = min(
        candidates,
        key=lambda c: abs(math.log(c / target)) if target > 0 else abs(c - target),
    )
    return best


# ---------------------------------------------------------------------------
# Component database
# ---------------------------------------------------------------------------


class ComponentDB:
    """Query interface for the JLCPCB basic parts library.

    On construction the entire JSON parts file is loaded into memory and
    indexed by LCSC number and category for fast lookups.
    """

    def __init__(self, parts_file: Path = JLCPCB_PARTS_FILE) -> None:
        """Load the parts database from a JSON file.

        Args:
            parts_file: Path to the JLCPCB parts JSON (defaults to
                :data:`JLCPCB_PARTS_FILE`).

        Raises:
            FileNotFoundError: If *parts_file* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        log.debug("Loading JLCPCB parts database from %s", parts_file)
        with parts_file.open(encoding="utf-8") as fh:
            raw: dict[str, object] = json.load(fh)

        parts_raw: list[dict[str, object]] = raw.get("parts", [])  # type: ignore[assignment]
        self._parts: list[JLCPCBPart] = [self._parse_part(p) for p in parts_raw]
        self._by_lcsc: dict[str, JLCPCBPart] = {p.lcsc: p for p in self._parts}
        self._by_category: dict[str, list[JLCPCBPart]] = {}
        for part in self._parts:
            self._by_category.setdefault(part.category, []).append(part)
        log.debug("Loaded %d parts from database", len(self._parts))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_part(raw: dict[str, object]) -> JLCPCBPart:
        """Convert a raw JSON dict to a :class:`JLCPCBPart` instance."""
        # The JSON uses "if" (Python keyword) — remap to if_
        if_ = raw.get("if")
        return JLCPCBPart(
            lcsc=str(raw["lcsc"]),
            mfr=str(raw["mfr"]),
            value=str(raw["value"]),
            package=str(raw["package"]),
            category=str(raw["category"]),
            price_usd=float(raw["price_usd"]),  # type: ignore[arg-type]
            voltage=str(raw["voltage"]) if raw.get("voltage") is not None else None,
            dielectric=str(raw["dielectric"]) if raw.get("dielectric") is not None else None,
            vceo=str(raw["vceo"]) if raw.get("vceo") is not None else None,
            ic=str(raw["ic"]) if raw.get("ic") is not None else None,
            vr=str(raw["vr"]) if raw.get("vr") is not None else None,
            if_=str(if_) if if_ is not None else None,
            vout=float(raw["vout"]) if raw.get("vout") is not None else None,  # type: ignore[arg-type]
            iout_ma=float(raw["iout_ma"]) if raw.get("iout_ma") is not None else None,  # type: ignore[arg-type]
            vin_max=float(raw["vin_max"]) if raw.get("vin_max") is not None else None,  # type: ignore[arg-type]
            vf=float(raw["vf"]) if raw.get("vf") is not None else None,  # type: ignore[arg-type]
            in_stock=bool(raw.get("in_stock", True)),
        )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def find_by_lcsc(self, lcsc: str) -> JLCPCBPart | None:
        """Return a part by its LCSC number, or ``None`` if not found.

        Args:
            lcsc: LCSC part number (e.g. ``'C17414'``).
        """
        return self._by_lcsc.get(lcsc)

    def find_by_category(self, category: str) -> list[JLCPCBPart]:
        """Return all parts in a category.

        Args:
            category: Category string (e.g. ``'resistor'``, ``'capacitor'``,
                      ``'ldo'``).

        Returns:
            Possibly empty list of :class:`JLCPCBPart` instances.
        """
        return list(self._by_category.get(category, []))

    def find_resistor(
        self,
        value_ohms: float,
        package: str = "0805",
    ) -> JLCPCBPart | None:
        """Find the resistor closest to *value_ohms* in the given *package*.

        Matches by converting the value string to ohms (e.g. ``'10k'`` →
        ``10000.0``) then choosing the part with the smallest relative error.

        Args:
            value_ohms: Target resistance in ohms.
            package: Target package size (default ``'0805'``).

        Returns:
            Closest-matching :class:`JLCPCBPart`, or ``None`` if no match.
        """
        candidates: list[tuple[float, JLCPCBPart]] = []
        for part in self._by_category.get("resistor", []):
            if part.package != package:
                continue
            parsed = _parse_resistance_ohms(part.value)
            if parsed is None:
                continue
            rel_err = abs(parsed - value_ohms) / max(value_ohms, 1e-9)
            candidates.append((rel_err, part))

        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        best_err, best_part = candidates[0]
        log.debug(
            "find_resistor(%g, %s) → %s (rel_err=%.3f)",
            value_ohms,
            package,
            best_part.lcsc,
            best_err,
        )
        return best_part

    def find_capacitor(
        self,
        value_uf: float,
        package: str = "0805",
        min_voltage: float = 0.0,
    ) -> JLCPCBPart | None:
        """Find the capacitor closest to *value_uf* in the given *package*.

        Args:
            value_uf: Target capacitance in microfarads.
            package: Package size (default ``'0805'``).
            min_voltage: Minimum voltage rating in volts (``0`` = no filter).

        Returns:
            Closest-matching :class:`JLCPCBPart`, or ``None`` if no match.
        """
        candidates: list[tuple[float, JLCPCBPart]] = []
        for part in self._by_category.get("capacitor", []):
            if part.package != package:
                continue
            # Voltage filter
            if min_voltage > 0.0 and part.voltage is not None:
                v = _parse_voltage_v(part.voltage)
                if v is not None and v < min_voltage:
                    continue
            parsed = _parse_capacitance_uf(part.value)
            if parsed is None:
                continue
            rel_err = abs(parsed - value_uf) / max(value_uf, 1e-15)
            candidates.append((rel_err, part))

        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        best_err, best_part = candidates[0]
        log.debug(
            "find_capacitor(%g µF, %s, min_v=%g) → %s (rel_err=%.3f)",
            value_uf,
            package,
            min_voltage,
            best_part.lcsc,
            best_err,
        )
        return best_part

    def find_ldo(
        self,
        vout: float,
        min_iout_ma: float = 0.0,
    ) -> JLCPCBPart | None:
        """Find an LDO regulator with the given output voltage.

        Args:
            vout: Required output voltage (volts).
            min_iout_ma: Minimum output current capability (mA, default 0).

        Returns:
            Best-matching :class:`JLCPCBPart`, or ``None`` if no match.
        """
        candidates: list[tuple[float, JLCPCBPart]] = []
        for part in self._by_category.get("ldo", []):
            if part.vout is None:
                continue
            if min_iout_ma > 0.0 and (part.iout_ma is None or part.iout_ma < min_iout_ma):
                continue
            err = abs(part.vout - vout)
            candidates.append((err, part))

        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        _, best_part = candidates[0]
        log.debug("find_ldo(%g V, min_iout=%g mA) → %s", vout, min_iout_ma, best_part.lcsc)
        return best_part

    def find_led(
        self,
        color: str = "green",
        package: str = "0805",
    ) -> JLCPCBPart | None:
        """Find an LED by colour hint in the value string.

        Args:
            color: Colour keyword to search for (case-insensitive,
                   default ``'green'``).
            package: Package size (default ``'0805'``).

        Returns:
            First matching :class:`JLCPCBPart`, or ``None`` if no match.
        """
        colour_lower = color.lower()
        for part in self._by_category.get("led", []):
            if part.package != package:
                continue
            if colour_lower in part.value.lower():
                log.debug("find_led(%s, %s) → %s", color, package, part.lcsc)
                return part
        return None

    def all_parts(self) -> list[JLCPCBPart]:
        """Return all parts currently loaded in the database.

        Returns:
            List of every :class:`JLCPCBPart` in load order.
        """
        return list(self._parts)

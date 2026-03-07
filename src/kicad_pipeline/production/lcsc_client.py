"""LCSC stock lookup client using stdlib urllib."""

from __future__ import annotations

import contextlib
import json
import logging
import urllib.request
from dataclasses import dataclass

from kicad_pipeline.constants import LCSC_API_BASE_URL, LCSC_STOCK_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LCSCStockInfo:
    """Stock information for a single LCSC part."""

    lcsc: str
    in_stock: bool
    stock_qty: int | None
    unit_price_usd: float | None
    description: str
    package: str


def fetch_lcsc_stock(
    lcsc: str,
    timeout: float = LCSC_STOCK_TIMEOUT_SECONDS,
) -> LCSCStockInfo | None:
    """Fetch stock info for a single LCSC part number.

    Returns None on any network/parse error (logs warning).
    """
    if not lcsc or not lcsc.startswith("C"):
        return None

    url = f"{LCSC_API_BASE_URL}?productCode={lcsc}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "kicad-ai-pipeline/1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        logger.warning("LCSC lookup failed for %s", lcsc)
        return None

    return _parse_lcsc_response(lcsc, data)


def _parse_lcsc_response(
    lcsc: str, data: dict[str, object]
) -> LCSCStockInfo | None:
    """Parse LCSC API JSON response into LCSCStockInfo."""
    result = data.get("result")
    if not isinstance(result, dict):
        return None

    stock_qty_raw = result.get("stockNumber")
    stock_qty: int | None = int(stock_qty_raw) if stock_qty_raw is not None else None
    in_stock = stock_qty is not None and stock_qty > 0

    # Extract cheapest price from price list
    unit_price: float | None = None
    price_list = result.get("productPriceList")
    if isinstance(price_list, list) and price_list:
        for entry in price_list:
            if isinstance(entry, dict):
                price_val = entry.get("productPrice")
                if price_val is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        unit_price = float(price_val)
                    break

    description = str(result.get("productDescEn", ""))
    package = str(result.get("encapStandard", ""))

    return LCSCStockInfo(
        lcsc=lcsc,
        in_stock=in_stock,
        stock_qty=stock_qty,
        unit_price_usd=unit_price,
        description=description,
        package=package,
    )


def fetch_lcsc_stock_batch(
    lcsc_numbers: tuple[str, ...],
    timeout: float = LCSC_STOCK_TIMEOUT_SECONDS,
) -> dict[str, LCSCStockInfo]:
    """Fetch stock info for multiple LCSC part numbers.

    Returns dict mapping LCSC number to stock info. Missing/failed lookups
    are omitted from the result.
    """
    results: dict[str, LCSCStockInfo] = {}
    for lcsc in lcsc_numbers:
        info = fetch_lcsc_stock(lcsc, timeout=timeout)
        if info is not None:
            results[lcsc] = info
    return results

"""Datasheet downloader for project components.

Downloads PDF datasheets for components listed in project requirements.
Uses ``component.datasheet`` URL when available, with fallback to LCSC
product pages.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.requirements import ProjectRequirements

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasheetResult:
    """Outcome of a single datasheet download attempt."""

    ref: str
    url: str
    local_path: Path | None
    status: str  # "downloaded" | "cached" | "failed" | "no_url"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNSAFE_CHARS = re.compile(r"[^\w\-.]")


def _sanitize_filename(value: str) -> str:
    """Replace unsafe filesystem characters with underscores."""
    return _UNSAFE_CHARS.sub("_", value)


def _lcsc_url(lcsc: str) -> str:
    """Build an LCSC product page URL from a part number."""
    return f"https://www.lcsc.com/product-detail/{lcsc}.html"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_datasheets(
    requirements: ProjectRequirements,
    output_dir: Path,
    timeout: float = 30.0,
) -> tuple[DatasheetResult, ...]:
    """Download datasheets for all components in *requirements*.

    For each component:
    1. If a local PDF already exists, mark as ``"cached"``.
    2. If ``component.datasheet`` has a URL, fetch it.
    3. Else if ``component.lcsc`` is set, try the LCSC product page.
    4. Otherwise mark ``"no_url"``.

    Args:
        requirements: Project requirements containing component list.
        output_dir: Directory to save downloaded PDFs into.
        timeout: HTTP request timeout in seconds.

    Returns:
        Tuple of :class:`DatasheetResult` for every component.
    """
    from pathlib import Path as _Path  # avoid shadowing TYPE_CHECKING import

    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[DatasheetResult] = []

    for comp in requirements.components:
        safe_value = _sanitize_filename(comp.value)
        filename = f"{comp.ref}_{safe_value}.pdf"
        local_path = output_dir / filename

        # Already cached?
        if local_path.exists():
            url = comp.datasheet or ""
            results.append(
                DatasheetResult(
                    ref=comp.ref,
                    url=url,
                    local_path=local_path,
                    status="cached",
                )
            )
            log.debug("Datasheet cached: %s → %s", comp.ref, local_path)
            continue

        # Determine URL
        url = comp.datasheet or ""
        if not url and comp.lcsc:
            url = _lcsc_url(comp.lcsc)

        if not url:
            results.append(
                DatasheetResult(
                    ref=comp.ref, url="", local_path=None, status="no_url",
                )
            )
            log.info("No datasheet URL for %s (%s)", comp.ref, comp.value)
            continue

        # Download
        result = _fetch_datasheet(comp.ref, url, local_path, timeout)
        results.append(result)

    downloaded = sum(1 for r in results if r.status == "downloaded")
    cached = sum(1 for r in results if r.status == "cached")
    failed = sum(1 for r in results if r.status == "failed")
    log.info(
        "Datasheets: %d downloaded, %d cached, %d failed, %d no URL",
        downloaded,
        cached,
        failed,
        len(results) - downloaded - cached - failed,
    )

    return tuple(results)


def _fetch_datasheet(
    ref: str,
    url: str,
    local_path: Path,
    timeout: float,
) -> DatasheetResult:
    """Fetch a single datasheet PDF from *url*.

    Args:
        ref: Component reference designator.
        url: URL to download from.
        local_path: Local file path to save to.
        timeout: HTTP timeout in seconds.

    Returns:
        A :class:`DatasheetResult` describing the outcome.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "kicad-ai-pipeline/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        local_path.write_bytes(data)
        log.info("Downloaded datasheet: %s → %s", ref, local_path.name)
        return DatasheetResult(
            ref=ref, url=url, local_path=local_path, status="downloaded",
        )
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log.warning("Failed to download datasheet for %s from %s: %s", ref, url, exc)
        return DatasheetResult(
            ref=ref, url=url, local_path=None, status="failed",
        )

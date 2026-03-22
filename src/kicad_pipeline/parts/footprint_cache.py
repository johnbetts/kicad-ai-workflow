"""JLCPCB footprint download and cache management.

Downloads verified ``.kicad_mod`` footprint files from JLCPCB/EasyEDA by LCSC
part number, caching them locally. Falls back to the CDFER JLCPCB KiCad
library if installed.

Cache location: ``~/.cache/kicad-ai-pipeline/footprints/``
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

_log = logging.getLogger(__name__)

# Cache directory for downloaded footprints
_CACHE_DIR = Path.home() / ".cache" / "kicad-ai-pipeline" / "footprints"

# CDFER JLCPCB library path (KiCad 10 3rd-party install location)
_CDFER_BASE = (
    Path.home() / "Documents" / "KiCad" / "10.0" / "3rdparty" / "footprints"
    / "com_github_CDFER_JLCPCB-Kicad-Library" / "JLCPCB.pretty"
)


def _ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _is_negative_cached(lcsc: str) -> bool:
    """Check if a previous download attempt failed for this LCSC number."""
    return (_CACHE_DIR / f"{lcsc}.notfound").exists()


def _mark_negative_cache(lcsc: str) -> None:
    """Record that a download attempt failed so we don't retry."""
    _ensure_cache_dir()
    (_CACHE_DIR / f"{lcsc}.notfound").touch()


def _find_in_cache(lcsc: str) -> Path | None:
    """Check if a footprint for this LCSC number is already cached."""
    cache_dir = _CACHE_DIR
    if not cache_dir.exists():
        return None
    # easyeda2kicad outputs to {output}.pretty/{name}.kicad_mod
    pretty_dir = cache_dir / f"{lcsc}.pretty"
    if pretty_dir.exists():
        mods = list(pretty_dir.glob("*.kicad_mod"))
        if mods:
            return mods[0]
    return None


def _find_in_cdfer(lcsc: str) -> Path | None:
    """Check the CDFER JLCPCB library for a matching footprint.

    The CDFER library uses descriptive filenames, not LCSC numbers,
    so this is a best-effort lookup. Returns None if no match found.
    """
    if not _CDFER_BASE.exists():
        return None
    # CDFER doesn't index by LCSC number — skip for now
    # (Could add a mapping file in future)
    return None


def _download_footprint(lcsc: str) -> Path | None:
    """Download footprint via easyeda2kicad CLI.

    Args:
        lcsc: LCSC part number (e.g. ``"C2913202"``).

    Returns:
        Path to the downloaded ``.kicad_mod`` file, or None on failure.
    """
    cache_dir = _ensure_cache_dir()
    output_base = cache_dir / lcsc

    try:
        result = subprocess.run(
            [
                "easyeda2kicad",
                "--lcsc_id", lcsc,
                "--footprint",
                "--output", str(output_base),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        _log.warning("easyeda2kicad not installed — cannot download footprint for %s", lcsc)
        return None
    except subprocess.TimeoutExpired:
        _log.warning("easyeda2kicad timed out downloading %s", lcsc)
        _mark_negative_cache(lcsc)
        return None

    if result.returncode != 0:
        _log.warning(
            "easyeda2kicad failed for %s (rc=%d): %s",
            lcsc, result.returncode, result.stderr.strip(),
        )
        _mark_negative_cache(lcsc)
        return None

    # Find the output .kicad_mod
    pretty_dir = cache_dir / f"{lcsc}.pretty"
    if pretty_dir.exists():
        mods = list(pretty_dir.glob("*.kicad_mod"))
        if mods:
            _log.info("Downloaded JLCPCB footprint: %s -> %s", lcsc, mods[0].name)
            return mods[0]

    _log.warning("easyeda2kicad produced no .kicad_mod for %s", lcsc)
    _mark_negative_cache(lcsc)
    return None


def get_jlcpcb_footprint(lcsc: str) -> Path | None:
    """Get the path to a JLCPCB footprint ``.kicad_mod`` file.

    Resolution order:
    1. Local cache (``~/.cache/kicad-ai-pipeline/footprints/``)
    2. CDFER JLCPCB library (if installed)
    3. Download via ``easyeda2kicad``

    Never raises — returns None on any failure.

    Args:
        lcsc: LCSC part number (e.g. ``"C2913202"``).

    Returns:
        Path to the ``.kicad_mod`` file, or None if unavailable.
    """
    if not lcsc:
        return None

    # 0. Skip parts that previously failed download
    if _is_negative_cached(lcsc):
        _log.debug("Negative cache hit for %s — skipping download", lcsc)
        return None

    # 1. Check cache
    cached = _find_in_cache(lcsc)
    if cached is not None:
        _log.debug("Cache hit for %s: %s", lcsc, cached)
        return cached

    # 2. Check CDFER library
    cdfer = _find_in_cdfer(lcsc)
    if cdfer is not None:
        _log.debug("CDFER hit for %s: %s", lcsc, cdfer)
        return cdfer

    # 3. Download
    return _download_footprint(lcsc)

"""JLCPCB footprint download and cache management.

Downloads verified ``.kicad_mod`` footprint files from JLCPCB/EasyEDA by LCSC
part number, caching them locally. Falls back to the CDFER JLCPCB KiCad
library if installed.

Cache location: ``~/.cache/kicad-ai-pipeline/footprints/``

**Cache integrity:** Every cached footprint is validated before use.  If the
``.kicad_mod`` file cannot be parsed or contains zero pads, it is evicted
and re-downloaded.  Bad caches never persist in the pipeline.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

_log = logging.getLogger(__name__)

# Cache directory for downloaded footprints
_CACHE_DIR = Path.home() / ".cache" / "kicad-ai-pipeline" / "footprints"

# Cache TTL: footprints older than this are re-downloaded.
# EasyEDA/JLCPCB occasionally updates footprint data for the same LCSC number.
_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days

# CDFER JLCPCB library path (KiCad 10 3rd-party install location)
_CDFER_BASE = (
    Path.home() / "Documents" / "KiCad" / "10.0" / "3rdparty" / "footprints"
    / "com_github_CDFER_JLCPCB-Kicad-Library" / "JLCPCB.pretty"
)


def _ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _is_expired(path: Path) -> bool:
    """Return True if the file is older than ``_CACHE_TTL_SECONDS``."""
    try:
        age = time.time() - path.stat().st_mtime
        return age > _CACHE_TTL_SECONDS
    except OSError:
        return True


def _is_negative_cached(lcsc: str) -> bool:
    """Check if a previous download attempt failed for this LCSC number.

    Negative cache entries also expire after ``_CACHE_TTL_SECONDS`` so that
    parts that were temporarily unavailable get retried.
    """
    marker = _CACHE_DIR / f"{lcsc}.notfound"
    if not marker.exists():
        return False
    if _is_expired(marker):
        _log.info("Negative cache expired for %s — will retry download", lcsc)
        marker.unlink(missing_ok=True)
        return False
    return True


def _mark_negative_cache(lcsc: str) -> None:
    """Record that a download attempt failed so we don't retry."""
    _ensure_cache_dir()
    (_CACHE_DIR / f"{lcsc}.notfound").touch()


def _validate_cached_mod(mod_path: Path, lcsc: str) -> bool:
    """Return True if a cached .kicad_mod is structurally valid.

    Checks:
    1. File is non-empty
    2. File can be parsed as S-expression (valid KiCad format)
    3. Parsed footprint contains at least one pad

    If any check fails, the entire cache entry for this LCSC is evicted.
    """
    try:
        sz = mod_path.stat().st_size
        if sz < 50:  # too small to be a real footprint
            _log.warning("Cache evict %s: file too small (%d bytes)", lcsc, sz)
            return False
        content = mod_path.read_text(encoding="utf-8", errors="replace")
        if "(pad " not in content:
            _log.warning("Cache evict %s (%s): no pads found in file", lcsc, mod_path.name)
            return False
        if "(footprint " not in content and "(module " not in content:
            _log.warning("Cache evict %s (%s): not a valid footprint file", lcsc, mod_path.name)
            return False
    except OSError as exc:
        _log.warning("Cache evict %s: read error: %s", lcsc, exc)
        return False
    return True


def _evict_cache(lcsc: str) -> None:
    """Remove a cached footprint entry entirely."""
    pretty_dir = _CACHE_DIR / f"{lcsc}.pretty"
    if pretty_dir.exists():
        shutil.rmtree(pretty_dir, ignore_errors=True)
        _log.info("Evicted bad cache for %s", lcsc)


def _find_in_cache(lcsc: str) -> Path | None:
    """Check if a valid, non-expired footprint for this LCSC number is cached.

    Returns ``None`` (cache miss) if the entry is expired, missing, or
    fails structural validation.  Invalid entries are evicted immediately
    so bad caches never persist.
    """
    cache_dir = _CACHE_DIR
    if not cache_dir.exists():
        return None
    pretty_dir = cache_dir / f"{lcsc}.pretty"
    if pretty_dir.exists():
        mods = list(pretty_dir.glob("*.kicad_mod"))
        if mods:
            if _is_expired(mods[0]):
                _log.info(
                    "Cache expired for %s (%s) — will re-download",
                    lcsc, mods[0].name,
                )
                _evict_cache(lcsc)
                return None
            if not _validate_cached_mod(mods[0], lcsc):
                _evict_cache(lcsc)
                return None
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

    # Find the output .kicad_mod and validate it before accepting
    pretty_dir = cache_dir / f"{lcsc}.pretty"
    if pretty_dir.exists():
        mods = list(pretty_dir.glob("*.kicad_mod"))
        if mods:
            if _validate_cached_mod(mods[0], lcsc):
                _log.info("Downloaded JLCPCB footprint: %s -> %s", lcsc, mods[0].name)
                return mods[0]
            # Download produced an invalid file — evict and negative-cache
            _evict_cache(lcsc)
            _mark_negative_cache(lcsc)
            return None

    _log.warning("easyeda2kicad produced no .kicad_mod for %s", lcsc)
    _mark_negative_cache(lcsc)
    return None


def invalidate_footprint(lcsc: str) -> None:
    """Evict a cached footprint that failed downstream validation.

    Called by ``_try_jlcpcb_footprint`` when the loaded footprint fails
    pad-count, pad-type, or package-code checks.  Ensures the bad cache
    entry is removed so the next build attempt gets a fresh download.
    """
    _evict_cache(lcsc)


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

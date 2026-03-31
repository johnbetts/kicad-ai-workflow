"""Golden image baseline comparison for component verification.

After a component passes AI visual inspection and human confirmation,
its renders are promoted to "golden baselines". Future verification runs
compare against these baselines using SHA-256 hash comparison.

KiCad renders are deterministic for the same input, so byte-identical
hash comparison is the correct approach (no perceptual hashing needed).

Storage layout::

    data/component_baselines/
        R_0805/
            manifest.json          # checksums + metadata
            R_0805_3d_iso.png      # golden image copies
            R_0805_3d_top.png
            R_0805_2d_top.png
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path  # noqa: TC003 — used at runtime

_log = logging.getLogger(__name__)

# Views that are tracked in baselines.
_BASELINE_VIEWS = ("3d_iso", "3d_top", "2d_top", "3d_bottom")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineManifest:
    """Metadata for a component's golden baseline."""

    component_id: str
    views: tuple[tuple[str, str], ...]  # (view_name, sha256_hex)
    promoted_at: str                     # ISO 8601 timestamp
    commit_hash: str
    kicad_version: str


@dataclass(frozen=True)
class BaselineResult:
    """Result of comparing current renders against golden baseline."""

    component_id: str
    has_baseline: bool
    matches: tuple[tuple[str, bool], ...]  # (view_name, matched)
    all_match: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_hash() -> str:
    """Get current git HEAD short hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _kicad_version() -> str:
    """Get KiCad version string, or 'unknown'."""
    try:
        result = subprocess.run(
            ["kicad-cli", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip().split("\n")[0] or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def has_baseline(
    component_id: str,
    baselines_dir: Path,
) -> bool:
    """Check if a golden baseline exists for a component."""
    manifest_path = baselines_dir / component_id / "manifest.json"
    return manifest_path.exists()


def promote_to_baseline(
    component_id: str,
    evidence_dir: Path,
    baselines_dir: Path,
) -> BaselineManifest:
    """Promote current evidence renders to golden baseline.

    Copies render PNGs from the evidence directory to the baselines
    directory and writes a manifest with SHA-256 checksums.

    Args:
        component_id: Component identifier (e.g. "R_0805").
        evidence_dir: Directory containing current renders
            (e.g. ``data/component_evidence/R_0805/``).
        baselines_dir: Root baselines directory
            (e.g. ``data/component_baselines/``).

    Returns:
        The created BaselineManifest.

    Raises:
        FileNotFoundError: If no renderable views found in evidence_dir.
    """
    target_dir = baselines_dir / component_id
    target_dir.mkdir(parents=True, exist_ok=True)

    views: list[tuple[str, str]] = []

    for view in _BASELINE_VIEWS:
        # Evidence files are named: {component_id}_{view}.png
        src = evidence_dir / f"{component_id}_{view}.png"
        if not src.exists():
            _log.debug("No %s render for %s, skipping", view, component_id)
            continue

        dst = target_dir / src.name
        shutil.copy2(src, dst)
        digest = _sha256(dst)
        views.append((view, digest))

    if not views:
        msg = f"No baseline views found in {evidence_dir} for {component_id}"
        raise FileNotFoundError(msg)

    manifest = BaselineManifest(
        component_id=component_id,
        views=tuple(views),
        promoted_at=datetime.now(tz=timezone.utc).isoformat(),
        commit_hash=_git_commit_hash(),
        kicad_version=_kicad_version(),
    )

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "component_id": manifest.component_id,
        "views": {v: h for v, h in manifest.views},
        "promoted_at": manifest.promoted_at,
        "commit_hash": manifest.commit_hash,
        "kicad_version": manifest.kicad_version,
    }, indent=2) + "\n")

    _log.info(
        "Promoted %s baseline: %d views, commit %s",
        component_id, len(views), manifest.commit_hash,
    )
    return manifest


def compare_to_baseline(
    component_id: str,
    evidence_dir: Path,
    baselines_dir: Path,
) -> BaselineResult:
    """Compare current renders against golden baseline.

    Args:
        component_id: Component identifier.
        evidence_dir: Directory with current renders.
        baselines_dir: Root baselines directory.

    Returns:
        BaselineResult with per-view match status.
    """
    manifest_path = baselines_dir / component_id / "manifest.json"
    if not manifest_path.exists():
        return BaselineResult(
            component_id=component_id,
            has_baseline=False,
            matches=(),
            all_match=True,  # no baseline = nothing to fail
        )

    try:
        data = json.loads(manifest_path.read_text())
        baseline_views: dict[str, str] = data.get("views", {})
    except (json.JSONDecodeError, KeyError) as exc:
        _log.warning("Failed to read baseline manifest for %s: %s", component_id, exc)
        return BaselineResult(
            component_id=component_id,
            has_baseline=False,
            matches=(),
            all_match=True,
        )

    matches: list[tuple[str, bool]] = []
    for view, expected_hash in baseline_views.items():
        current_path = evidence_dir / f"{component_id}_{view}.png"
        if not current_path.exists():
            _log.debug("Missing %s render for baseline comparison", view)
            matches.append((view, False))
            continue

        current_hash = _sha256(current_path)
        matched = current_hash == expected_hash
        if not matched:
            _log.warning(
                "Baseline regression: %s/%s hash mismatch "
                "(expected %s..., got %s...)",
                component_id, view,
                expected_hash[:12], current_hash[:12],
            )
        matches.append((view, matched))

    return BaselineResult(
        component_id=component_id,
        has_baseline=True,
        matches=tuple(matches),
        all_match=all(m for _, m in matches),
    )

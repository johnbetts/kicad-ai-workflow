"""Shared utilities for training board builds and eval runs.

Provides archive-before-build logic so every iteration starts from a
clean state, with the previous output preserved for comparison.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_log = logging.getLogger(__name__)


def archive_and_clean(output_dir: Path) -> Path | None:
    """Archive the current output directory and return the archive path.

    Copies ``output_dir`` to ``output/_archive/<name>-<timestamp>/``,
    then deletes the original so the next build starts clean.

    Returns the archive path, or ``None`` if nothing to archive.

    Example::

        archive_path = archive_and_clean(Path("output/train_mcu_core"))
        # -> output/_archive/train_mcu_core-2026-03-28_143012/
        # output/train_mcu_core/ is now empty (recreated)
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return None

    # Check if there's anything worth archiving (at least a .kicad_pcb or .png)
    has_artifacts = any(
        output_dir.glob("*.kicad_pcb")
    ) or any(
        output_dir.glob("*.png")
    )
    if not has_artifacts:
        return None

    # Build archive path
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    name = output_dir.name
    archive_root = output_dir.parent / "_archive"
    archive_dir = archive_root / f"{name}-{ts}"

    # Copy to archive (preserve .pcb-review/ session data)
    archive_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, archive_dir, dirs_exist_ok=False)
    _log.info("Archived %s -> %s", output_dir, archive_dir)

    # Clean output dir — remove artifacts but preserve .pcb-review/
    pcb_review = output_dir / ".pcb-review"
    review_backup = None
    if pcb_review.exists():
        review_backup = archive_root / f".pcb-review-{name}-tmp"
        if review_backup.exists():
            shutil.rmtree(review_backup)
        shutil.copytree(pcb_review, review_backup)

    # Remove and recreate
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Restore .pcb-review/ (session knowledge persists across builds)
    if review_backup is not None and review_backup.exists():
        shutil.copytree(review_backup, pcb_review)
        shutil.rmtree(review_backup)

    _log.info("Cleaned %s for fresh build", output_dir)
    return archive_dir


def compare_summary(
    current_dir: Path,
    archive_dir: Path,
) -> str:
    """Generate a short comparison between current and archived output.

    Counts files, checks for new/removed components, compares PCB file sizes.
    """
    lines: list[str] = []

    cur_pcb = list(current_dir.glob("*.kicad_pcb"))
    old_pcb = list(archive_dir.glob("*.kicad_pcb"))

    if cur_pcb and old_pcb:
        cur_size = cur_pcb[0].stat().st_size
        old_size = old_pcb[0].stat().st_size
        delta = cur_size - old_size
        sign = "+" if delta >= 0 else ""
        lines.append(f"PCB size: {old_size} -> {cur_size} ({sign}{delta} bytes)")

    cur_pngs = set(p.name for p in current_dir.glob("*.png"))
    old_pngs = set(p.name for p in archive_dir.glob("*.png"))
    new_pngs = cur_pngs - old_pngs
    removed_pngs = old_pngs - cur_pngs
    if new_pngs:
        lines.append(f"New renders: {', '.join(sorted(new_pngs))}")
    if removed_pngs:
        lines.append(f"Removed renders: {', '.join(sorted(removed_pngs))}")

    lines.append(f"Current: {len(list(current_dir.iterdir()))} files")
    lines.append(f"Archive: {len(list(archive_dir.iterdir()))} files")
    lines.append(f"Archive at: {archive_dir}")

    return "\n".join(lines)

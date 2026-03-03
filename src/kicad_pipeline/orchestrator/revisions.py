"""Revision management for production snapshots.

Each variant can accumulate numbered revisions.  A revision is a frozen
snapshot of the ``production/`` directory at a point in time, stored under
``variants/{name}/revisions/revN/production/``.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import OrchestrationError
from kicad_pipeline.orchestrator.models import ProjectManifest, RevisionRecord

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class RevisionManager:
    """Create and query production revision snapshots for variants."""

    def __init__(self, project_root: Path) -> None:
        self._root = project_root

    def _variant_dir(self, variant_name: str) -> Path:
        """Return the filesystem path for a variant."""
        return self._root / "variants" / variant_name

    def create_revision(
        self,
        variant_name: str,
        manifest: ProjectManifest,
        notes: str = "",
        commit_hash: str = "",
    ) -> RevisionRecord:
        """Create a new production revision snapshot.

        Copies the contents of ``variants/{name}/production/`` into a new
        numbered revision directory and returns the corresponding
        :class:`RevisionRecord`.

        Args:
            variant_name: Variant to snapshot.
            manifest: Current project manifest (used to determine the next
                revision number).
            notes: Optional human-readable notes for this revision.
            commit_hash: Git commit hash to record against the revision.

        Returns:
            A new :class:`RevisionRecord`.

        Raises:
            OrchestrationError: If the source production directory does not
                exist.
        """
        source = self._variant_dir(variant_name) / "production"
        if not source.is_dir():
            raise OrchestrationError(
                f"Production directory does not exist: {source}"
            )

        # Determine next revision number from existing revisions in manifest.
        existing = self.list_revisions(variant_name, manifest)
        next_number = max((r.number for r in existing), default=0) + 1

        dest = self.get_revision_path(variant_name, next_number) / "production"
        shutil.copytree(source, dest)
        log.info(
            "Created revision %d for variant %s at %s",
            next_number,
            variant_name,
            dest,
        )

        git_tag = f"{variant_name}/rev{next_number}"
        return RevisionRecord(
            number=next_number,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            git_tag=git_tag,
            commit_hash=commit_hash,
            notes=notes,
        )

    def get_revision_path(self, variant_name: str, revision_number: int) -> Path:
        """Return the filesystem path for a specific revision.

        Args:
            variant_name: Name of the variant.
            revision_number: 1-based revision number.

        Returns:
            Path to the revision directory.
        """
        return self._variant_dir(variant_name) / "revisions" / f"rev{revision_number}"

    def list_revisions(
        self, variant_name: str, manifest: ProjectManifest
    ) -> tuple[RevisionRecord, ...]:
        """List all revisions for a variant from the manifest.

        Args:
            variant_name: Variant to query.
            manifest: Current project manifest.

        Returns:
            Tuple of :class:`RevisionRecord` for the variant, or an empty
            tuple if the variant is not found.
        """
        for variant in manifest.variants:
            if variant.name == variant_name:
                return variant.revisions
        return ()

    def mark_sent_to_fab(
        self,
        variant_name: str,
        revision_number: int,
        manifest: ProjectManifest,
        order_id: str | None = None,
    ) -> tuple[ProjectManifest, RevisionRecord]:
        """Mark a revision as sent to fabrication.

        Args:
            variant_name: Variant containing the revision.
            revision_number: Revision number to update.
            manifest: Current project manifest.
            order_id: Optional fabrication order identifier.

        Returns:
            A tuple of the updated :class:`ProjectManifest` and the updated
            :class:`RevisionRecord`.

        Raises:
            OrchestrationError: If the variant or revision is not found.
        """
        for v_idx, variant in enumerate(manifest.variants):
            if variant.name != variant_name:
                continue
            for r_idx, rev in enumerate(variant.revisions):
                if rev.number != revision_number:
                    continue
                updated_rev = replace(
                    rev, sent_to_fab=True, fab_order_id=order_id
                )
                new_revisions = (
                    *variant.revisions[:r_idx],
                    updated_rev,
                    *variant.revisions[r_idx + 1 :],
                )
                new_variant = replace(variant, revisions=new_revisions)
                new_variants = (
                    *manifest.variants[:v_idx],
                    new_variant,
                    *manifest.variants[v_idx + 1 :],
                )
                new_manifest = replace(manifest, variants=new_variants)
                return new_manifest, updated_rev

        raise OrchestrationError(
            f"Revision {revision_number} not found for variant {variant_name!r}"
        )

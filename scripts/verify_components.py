#!/usr/bin/env python3
"""Full component isolation verification with rendering and evidence.

Usage::

    python scripts/verify_components.py              # all components
    python scripts/verify_components.py R_0805        # one component
    python scripts/verify_components.py --failed-only # re-verify failures only

Builds single-component isolation boards, renders 2D + 3D views,
runs structural checks, and updates the registry with verification status.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure src/ is on the path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from kicad_pipeline.validation.component_registry import ComponentRegistry
from kicad_pipeline.validation.component_verifier import verify_component


def _git_short_hash() -> str | None:
    """Get current git short hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            cwd=_REPO_ROOT,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Component isolation verification")
    parser.add_argument(
        "components",
        nargs="*",
        help="Component IDs to verify (default: all)",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only re-verify components with status='failed'",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering (structural checks only)",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "component_evidence",
        help="Directory for evidence images",
    )
    args = parser.parse_args()

    registry = ComponentRegistry()
    commit = _git_short_hash()

    # Determine which components to verify
    if args.components:
        specs = []
        for cid in args.components:
            spec = registry.get(cid)
            if spec is None:
                print(f"ERROR: Component '{cid}' not found in registry")
                return 1
            specs.append(spec)
    elif args.failed_only:
        specs = [s for s in registry.all_components() if s.verification_status == "failed"]
        if not specs:
            print("No failed components to re-verify.")
            return 0
    else:
        specs = registry.all_components()

    print(f"Verifying {len(specs)} components...")
    print()

    passed = 0
    failed = 0
    errors: list[str] = []

    for spec in specs:
        evidence_dir = args.evidence_dir / spec.component_id.replace("/", "_")
        render = not args.no_render

        try:
            result = verify_component(spec, evidence_dir, render=render, registry=registry)
        except Exception as exc:
            print(f"  ERROR  {spec.component_id}: {exc}")
            errors.append(f"{spec.component_id}: {exc}")
            registry.update_status(spec.component_id, "failed", commit)
            failed += 1
            continue

        if result.passed:
            status_icon = "  PASS "
            registry.update_status(spec.component_id, "verified", commit)
            passed += 1
        else:
            status_icon = "  FAIL "
            registry.update_status(spec.component_id, "failed", commit)
            failed += 1

        # Print results
        failed_checks = [c for c in result.checks if not c.passed]
        print(f"{status_icon} {spec.component_id} ({spec.description})")
        for check in failed_checks:
            print(f"         [{check.severity}] {check.name}: {check.detail}")

        if result.render_paths:
            for view_name, path in result.render_paths:
                print(f"         render: {view_name} → {path}")

    # Save updated registry
    registry.save()

    # Summary
    print()
    print(f"{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(specs)} total")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors:
            print(f"  {e}")
    print(f"Registry updated at: {registry._path}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

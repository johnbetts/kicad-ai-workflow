#!/usr/bin/env python3
"""Verify ALL required renders exist for a board before review can proceed.

Exits 0 if all renders present, exits 1 if any missing.
Designed to be called as a gate before review agents are launched.

Usage:
  python scripts/verify-renders.py output/train_mcu_core/train_mcu_core.kicad_pcb
  python scripts/verify-renders.py mcu
  python scripts/verify-renders.py all
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TRAINING_BOARDS: dict[str, str] = {
    "mcu": "output/train_mcu_core/train_mcu_core.kicad_pcb",
    "relay": "output/train_relay/train_relay.kicad_pcb",
    "power": "output/train_power/train_power.kicad_pcb",
    "analog": "output/train_analog_input/train_analog_input.kicad_pcb",
    "ethernet": "output/train_ethernet/train_ethernet.kicad_pcb",
}

REQUIRED_BOARD_VIEWS = ["2d_top", "3d_top", "3d_iso", "3d_isoback", "3d_hires_top"]


def resolve_boards(selector: str) -> list[str]:
    if selector == "all":
        return [p for p in TRAINING_BOARDS.values() if Path(p).exists()]
    if "," in selector:
        result: list[str] = []
        for part in selector.split(","):
            result.extend(resolve_boards(part.strip()))
        return result
    if selector in TRAINING_BOARDS:
        return [TRAINING_BOARDS[selector]]
    if Path(selector).exists():
        return [selector]
    matches = [k for k in TRAINING_BOARDS if selector in k]
    if len(matches) == 1:
        return [TRAINING_BOARDS[matches[0]]]
    print(f"ERROR: Unknown board '{selector}'")
    sys.exit(1)


def extract_component_refs(pcb_path: str) -> list[str]:
    pcb_text = Path(pcb_path).read_text()
    refs: list[str] = []
    for m in re.finditer(
        r'\(footprint\s+"[^"]*".*?\(property\s+"Reference"\s+"([^"]+)"',
        pcb_text, re.DOTALL,
    ):
        refs.append(m.group(1))
    return refs


def verify_board(board_path: str) -> tuple[int, int, list[str]]:
    """Returns (ok_count, missing_count, missing_files)."""
    board = Path(board_path)
    out_dir = board.parent
    name = board.stem
    missing: list[str] = []
    ok = 0

    # Board-level renders
    for view in REQUIRED_BOARD_VIEWS:
        f = out_dir / f"{name}_{view}.png"
        if f.exists():
            ok += 1
        else:
            missing.append(f"{name}_{view}.png")

    # Per-component crops
    refs = extract_component_refs(board_path)
    for ref in refs:
        f = out_dir / f"{name}_3d_comp_{ref}.png"
        if f.exists():
            ok += 1
        else:
            missing.append(f"{name}_3d_comp_{ref}.png")

    return ok, len(missing), missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify all required renders exist before review"
    )
    parser.add_argument("board", nargs="?", default=None,
                        help="Board path or shorthand")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Only print failures")
    args = parser.parse_args()

    if not args.board:
        parser.error("board selector required")

    boards = resolve_boards(args.board)
    all_ok = True

    for board_path in boards:
        name = Path(board_path).stem
        ok, missing_count, missing = verify_board(board_path)

        if missing_count == 0:
            if not args.quiet:
                print(f"PASS: {name} — {ok} renders present")
        else:
            all_ok = False
            print(f"FAIL: {name} — {missing_count} renders MISSING (have {ok})")
            # Show first 10 missing
            for f in missing[:10]:
                print(f"  MISSING: {f}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            print(f"\n  FIX: python scripts/render-board.py {board_path}")

    if not all_ok:
        print("\nRenders incomplete. Run render-board.py before proceeding to review.")
        sys.exit(1)
    elif not args.quiet:
        print("\nAll renders verified.")


if __name__ == "__main__":
    main()

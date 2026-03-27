#!/usr/bin/env python3
"""Render ALL required images for a board — board-level AND per-component.

This is the ONLY render script. Do not call kicad-image-gen directly.
Call this script, then call verify-renders.py to confirm all files exist.

Usage:
  python scripts/render-board.py output/train_mcu_core/train_mcu_core.kicad_pcb
  python scripts/render-board.py mcu          # shorthand
  python scripts/render-board.py all          # all 5 training boards
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TRAINING_BOARDS: dict[str, str] = {
    "mcu": "output/train_mcu_core/train_mcu_core.kicad_pcb",
    "relay": "output/train_relay/train_relay.kicad_pcb",
    "power": "output/train_power/train_power.kicad_pcb",
    "analog": "output/train_analog_input/train_analog_input.kicad_pcb",
    "ethernet": "output/train_ethernet/train_ethernet.kicad_pcb",
}

BOARD_VIEWS = [
    ("2d", [], "2d_top"),
    ("3d", ["--view", "top"], "3d_top"),
    ("3d", ["--view", "iso"], "3d_iso"),
    ("3d", ["--view", "iso-back"], "3d_isoback"),
    ("3d", ["--view", "top", "--hires"], "3d_hires_top"),
]


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
    print(f"ERROR: Unknown board '{selector}'. Known: {', '.join(TRAINING_BOARDS)}, all")
    sys.exit(1)


def run_kicad_image_gen(mode: str, board: str, output: str,
                        extra_args: list[str] | None = None) -> bool:
    cmd = ["kicad-image-gen", mode, board, "-o", output]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.returncode == 0 and Path(output).exists()


def extract_component_positions(pcb_path: str) -> list[tuple[str, float, float]]:
    """Parse (ref, x, y) for every footprint from .kicad_pcb."""
    pcb_text = Path(pcb_path).read_text()
    components: list[tuple[str, float, float]] = []
    for m in re.finditer(
        r'\(footprint\s+"[^"]*".*?\(at\s+([\d.]+)\s+([\d.]+).*?\)'
        r'.*?\(property\s+"Reference"\s+"([^"]+)"',
        pcb_text, re.DOTALL,
    ):
        x, y, ref = float(m.group(1)), float(m.group(2)), m.group(3)
        components.append((ref, x, y))
    return components


def render_board(board_path: str, skip_components: bool = False) -> dict[str, bool]:
    """Render all images for one board. Returns {filename: success}."""
    board = Path(board_path)
    out_dir = board.parent
    name = board.stem
    results: dict[str, bool] = {}

    # Board-level renders
    for mode, extra_args, view_name in BOARD_VIEWS:
        filename = f"{name}_{view_name}.png"
        out_path = out_dir / filename
        print(f"  [{view_name}] ", end="", flush=True)
        ok = run_kicad_image_gen(mode, str(board), str(out_path), extra_args)
        results[filename] = ok
        print("OK" if ok else "FAILED")

    if skip_components:
        return results

    # Per-component 3D crops
    components = extract_component_positions(str(board))
    print(f"  [{len(components)} components] ", end="", flush=True)
    comp_ok = 0
    comp_fail = 0
    for ref, x, y in components:
        pad = 5.0  # mm padding around component
        crop = f"{x - pad},{y - pad},{pad * 2 + 2},{pad * 2 + 2}"
        filename = f"{name}_3d_comp_{ref}.png"
        out_path = out_dir / filename
        ok = run_kicad_image_gen(
            "3d", str(board), str(out_path),
            ["--view", "top", "--crop", crop],
        )
        results[filename] = ok
        if ok:
            comp_ok += 1
        else:
            comp_fail += 1
    print(f"{comp_ok} OK, {comp_fail} failed")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ALL required images for PCB review"
    )
    parser.add_argument(
        "board", nargs="?", default=None,
        help="Board path, shorthand (mcu/relay/power/analog/ethernet), or 'all'",
    )
    parser.add_argument("--skip-components", action="store_true",
                        help="Skip per-component 3D crops (board-level only)")
    parser.add_argument("--list", action="store_true",
                        help="List available boards")
    args = parser.parse_args()

    if args.list:
        for name, path in TRAINING_BOARDS.items():
            exists = "OK" if Path(path).exists() else "MISSING"
            print(f"  {name:10s} → {path}  [{exists}]")
        return

    if not args.board:
        parser.error("board selector required")

    boards = resolve_boards(args.board)

    total_ok = 0
    total_fail = 0
    for board_path in boards:
        name = Path(board_path).stem
        print(f"\n{'=' * 50}")
        print(f"Rendering: {name}")
        print(f"{'=' * 50}")
        results = render_board(board_path, skip_components=args.skip_components)
        ok = sum(1 for v in results.values() if v)
        fail = sum(1 for v in results.values() if not v)
        total_ok += ok
        total_fail += fail
        print(f"  Total: {ok} OK, {fail} failed")

    print(f"\nAll boards: {total_ok} renders OK, {total_fail} failed")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

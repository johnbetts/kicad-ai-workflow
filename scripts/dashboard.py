"""Launch the KiCad review dashboard."""

from __future__ import annotations

import argparse


def cli() -> None:
    """Parse arguments and launch the dashboard."""
    parser = argparse.ArgumentParser(description="KiCad Review Dashboard")
    parser.add_argument("--board", help="Initial board to display (name or path)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-dir", help="Override output directory path")
    args = parser.parse_args()

    from kicad_pipeline.dashboard.app import main

    main(board_path=args.board, port=args.port, output_dir=args.output_dir)


if __name__ in {"__main__", "__mp_main__"}:
    cli()

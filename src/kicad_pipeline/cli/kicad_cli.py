"""Shared utility for locating the ``kicad-cli`` binary.

Both ``validation/kicad_drc.py`` and ``visualization/kicad_export.py`` use
this to find the KiCad CLI tool on macOS (or Linux/Windows via PATH).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from kicad_pipeline.exceptions import KiCadPipelineError

logger = logging.getLogger(__name__)

_KICAD_CLI_SEARCH: tuple[str, ...] = (
    os.environ.get("KICAD_CLI", ""),
    "/Applications/KiCad 10/KiCad.app/Contents/MacOS/kicad-cli",
    "/Applications/KiCad 9/KiCad.app/Contents/MacOS/kicad-cli",
    "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
)


def find_kicad_cli() -> str:
    """Locate the ``kicad-cli`` binary.

    Search order:
    1. ``$KICAD_CLI`` environment variable
    2. KiCad 10 default macOS path
    3. KiCad 9 default macOS path
    4. Legacy KiCad default macOS path
    5. ``which kicad-cli`` (PATH lookup)

    Returns:
        Absolute path to ``kicad-cli``.

    Raises:
        KiCadPipelineError: If ``kicad-cli`` cannot be found.
    """
    for candidate in _KICAD_CLI_SEARCH:
        if candidate and Path(candidate).is_file():
            logger.debug("Found kicad-cli at %s", candidate)
            return candidate

    # Fall back to PATH lookup.
    try:
        result = subprocess.run(
            ["which", "kicad-cli"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            path = result.stdout.strip()
            logger.debug("Found kicad-cli on PATH: %s", path)
            return path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    msg = (
        "Cannot find kicad-cli. Install KiCad 9/10 or set the KICAD_CLI "
        "environment variable to the full path of the kicad-cli binary."
    )
    raise KiCadPipelineError(msg)

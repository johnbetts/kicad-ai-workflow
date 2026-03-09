"""FreeRouting integration: headless launcher and SES result parser.

FreeRouting is a Java-based autorouter.  This module attempts to invoke it
when available and falls back gracefully when Java or the FreeRouting JAR
cannot be found.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign, Track, Via

# ---------------------------------------------------------------------------
# Common search locations for the FreeRouting JAR
# ---------------------------------------------------------------------------

_DEFAULT_SEARCH_DIRS: tuple[str, ...] = (
    ".",
    os.path.expanduser("~/.local/share/freerouting/"),
    "/usr/local/share/freerouting/",
    "/opt/freerouting/",
    # KiCad plugin locations (macOS / Linux / Windows)
    os.path.expanduser(
        "~/Documents/KiCad/10.0/3rdparty/plugins/"
        "app_freerouting_kicad-plugin/jar/"
    ),
    os.path.expanduser(
        "~/Documents/KiCad/9.0/3rdparty/plugins/"
        "app_freerouting_kicad-plugin/jar/"
    ),
    os.path.expanduser(
        "~/Documents/KiCad/8.0/3rdparty/plugins/"
        "app_freerouting_kicad-plugin/jar/"
    ),
    os.path.expanduser(
        "~/.local/share/kicad/10.0/3rdparty/plugins/"
        "app_freerouting_kicad-plugin/jar/"
    ),
    os.path.expanduser(
        "~/.local/share/kicad/9.0/3rdparty/plugins/"
        "app_freerouting_kicad-plugin/jar/"
    ),
)

_JAR_NAMES: tuple[str, ...] = (
    "freerouting.jar",
    "freerouting-2.1.0.jar",
    "freerouting-2.0.1.jar",
)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeRoutingResult:
    """Result of a FreeRouting autorouting run."""

    success: bool
    ses_file: str | None  # path to output .ses file if successful
    stdout: str
    stderr: str
    error: str = ""


# ---------------------------------------------------------------------------
# JAR discovery
# ---------------------------------------------------------------------------


def find_freerouting_jar(search_dirs: list[str] | None = None) -> str | None:
    """Search for freerouting.jar in common locations.

    Args:
        search_dirs: Additional directories to search before the defaults.

    Returns:
        Absolute path to the JAR file if found, else ``None``.
    """
    dirs_to_check: list[str] = list(search_dirs) if search_dirs else []
    dirs_to_check.extend(_DEFAULT_SEARCH_DIRS)

    for directory in dirs_to_check:
        for jar_name in _JAR_NAMES:
            candidate = os.path.join(directory, jar_name)
            if os.path.isfile(candidate):
                return candidate
    return None


# ---------------------------------------------------------------------------
# FreeRouting launcher
# ---------------------------------------------------------------------------


def route_with_freerouting(
    dsn_path: str,
    jar_path: str | None = None,
    timeout_seconds: int = 300,
) -> FreeRoutingResult:
    """Run FreeRouting headlessly to route the given DSN file.

    Executes::

        java -jar freerouting.jar -de input.dsn -do output.ses -mp 20

    Args:
        dsn_path: Path to the input ``.dsn`` file.
        jar_path: Path to the FreeRouting JAR.  If ``None``, the common
            locations are searched automatically.
        timeout_seconds: Maximum run time in seconds before the process is
            killed.

    Returns:
        A :class:`FreeRoutingResult` describing the outcome.
    """
    resolved_jar = jar_path if jar_path is not None else find_freerouting_jar()

    if resolved_jar is None:
        return FreeRoutingResult(
            success=False,
            ses_file=None,
            stdout="",
            stderr="",
            error="FreeRouting jar not found",
        )

    # Derive output .ses path by replacing the extension
    ses_path = dsn_path[:-4] + ".ses" if dsn_path.endswith(".dsn") else dsn_path + ".ses"

    cmd: list[str] = [
        "java",
        "-Djava.awt.headless=true",
        "-Dapple.awt.UIElement=true",
        "-jar",
        resolved_jar,
        "--gui.enabled=false",
        "-de",
        dsn_path,
        "-do",
        ses_path,
        "-mp",
        "20",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return FreeRoutingResult(
            success=False,
            ses_file=None,
            stdout=stdout,
            stderr=stderr,
            error=f"FreeRouting timed out after {timeout_seconds}s",
        )
    except FileNotFoundError:
        return FreeRoutingResult(
            success=False,
            ses_file=None,
            stdout="",
            stderr="",
            error="java executable not found; please install a JRE",
        )

    success = proc.returncode == 0 and os.path.isfile(ses_path)
    return FreeRoutingResult(
        success=success,
        ses_file=ses_path if success else None,
        stdout=proc.stdout,
        stderr=proc.stderr,
        error="" if success else f"FreeRouting exited with code {proc.returncode}",
    )


# ---------------------------------------------------------------------------
# SES parser
# ---------------------------------------------------------------------------

# Regex to match a wire path statement (quoted or unquoted layer names):
#   (path "F.Cu" 250 x1 y1 x2 y2 ...)
#   (path F.Cu 250 x1 y1 x2 y2 ...)
_PATH_RE = re.compile(
    r'\(path\s+(?:"([^"]+)"|([A-Za-z_.]+))\s+([\d.]+)((?:\s+[-\d.]+){4,})\s*\)',
    re.DOTALL,
)

# Regex to extract the net name containing wire paths
_NET_OUT_RE = re.compile(r'\(net\s+(?:"([^"]+)"|(\S+))(.*?)\)', re.DOTALL)


def ses_to_tracks(ses_content: str, pcb: PCBDesign) -> tuple[Track, ...]:
    """Parse a Specctra SES session file into :class:`Track` objects.

    Handles wire paths of the form::

        (path "F.Cu" 0.25 x1 y1 x2 y2 ...)

    Each consecutive (x, y) pair in the path becomes one :class:`Track`
    segment.  The net name is looked up in *pcb.nets* to obtain the net
    number; unknown net names receive net number 0.

    Args:
        ses_content: Raw text content of the ``.ses`` file.
        pcb: PCB design used for net-name to net-number resolution.

    Returns:
        Tuple of :class:`Track` objects parsed from the session file.
    """
    from kicad_pipeline.models.pcb import Point, Track

    net_name_to_num: dict[str, int] = {n.name: n.number for n in pcb.nets}

    tracks: list[Track] = []

    # We need to find which net each path belongs to.
    # Walk through the network_out section block by block.
    # Strategy: find all (net "NAME" ...) blocks, then find (path ...) within each.

    # Extract the routes / network_out section
    network_out_match = re.search(r"\(network_out(.*)", ses_content, re.DOTALL)
    search_text = network_out_match.group(1) if network_out_match else ses_content

    # Find net blocks: we use a simple bracket-counting approach for each
    # occurrence of (net NAME ...) to capture its full content.
    # FreeRouting may use quoted or unquoted net names.
    net_block_re = re.compile(r'\(net\s+(?:"([^"]+)"|(\S+))')
    pos = 0
    while True:
        m = net_block_re.search(search_text, pos)
        if m is None:
            break
        net_name = m.group(1) or m.group(2)
        net_number = net_name_to_num.get(net_name, 0)

        # Find the matching closing parenthesis for this (net ... block
        block_start = m.start()
        depth = 0
        block_end = block_start
        for i in range(block_start, len(search_text)):
            if search_text[i] == "(":
                depth += 1
            elif search_text[i] == ")":
                depth -= 1
                if depth == 0:
                    block_end = i + 1
                    break

        net_block = search_text[block_start:block_end]

        # Find all (path ...) within this net block — handle both
        # quoted and unquoted layer names and coordinates in mm*1000
        for path_m in _PATH_RE.finditer(net_block):
            layer = path_m.group(1) or path_m.group(2)
            width_raw = float(path_m.group(3))
            coords_str = path_m.group(4).strip()
            coord_vals = [float(v) for v in coords_str.split()]

            # FreeRouting uses resolution mm 1000 — convert to mm
            is_scaled = any(abs(v) > 200.0 for v in coord_vals)
            scale = 0.001 if is_scaled else 1.0
            width = width_raw * scale

            # Each consecutive pair of (x, y) values forms a track segment
            if len(coord_vals) >= 4 and len(coord_vals) % 2 == 0:
                for j in range(0, len(coord_vals) - 2, 2):
                    x0 = coord_vals[j] * scale
                    y0 = coord_vals[j + 1] * scale
                    x1 = coord_vals[j + 2] * scale
                    y1 = coord_vals[j + 3] * scale
                    tracks.append(
                        Track(
                            start=Point(x=x0, y=y0),
                            end=Point(x=x1, y=y1),
                            width=width,
                            layer=layer,
                            net_number=net_number,
                        )
                    )

        pos = m.end()

    return tuple(tracks)


# Regex to match a via statement in SES:
#   (via "Via[0-1]_0.9:0.508_mm" x y)
_VIA_RE = re.compile(
    r'\(via\s+"[^"]*"\s+([-\d.]+)\s+([-\d.]+)\s*\)',
)


def ses_to_vias(
    ses_content: str,
    pcb: PCBDesign,
    via_size: float = 0.9,
    via_drill: float = 0.508,
) -> tuple[Via, ...]:
    """Parse via statements from a Specctra SES session file.

    Handles via placements of the form::

        (via "Via[0-1]_0.9:0.508_mm" x y)

    Each via becomes a :class:`Via` with ``F.Cu``/``B.Cu`` layers.

    Args:
        ses_content: Raw text content of the ``.ses`` file.
        pcb: PCB design used for net-name to net-number resolution.
        via_size: Via annular ring diameter in mm.
        via_drill: Via drill diameter in mm.

    Returns:
        Tuple of :class:`Via` objects parsed from the session file.
    """
    from kicad_pipeline.models.pcb import Point, Via

    net_name_to_num: dict[str, int] = {n.name: n.number for n in pcb.nets}

    vias: list[Via] = []

    # Extract the network_out section
    network_out_match = re.search(r"\(network_out(.*)", ses_content, re.DOTALL)
    search_text = network_out_match.group(1) if network_out_match else ses_content

    # Find net blocks and extract vias from each (quoted or unquoted names)
    net_block_re = re.compile(r'\(net\s+(?:"([^"]+)"|(\S+))')
    pos = 0
    while True:
        m = net_block_re.search(search_text, pos)
        if m is None:
            break
        net_name = m.group(1) or m.group(2)
        net_number = net_name_to_num.get(net_name, 0)

        # Find matching closing parenthesis
        block_start = m.start()
        depth = 0
        block_end = block_start
        for i in range(block_start, len(search_text)):
            if search_text[i] == "(":
                depth += 1
            elif search_text[i] == ")":
                depth -= 1
                if depth == 0:
                    block_end = i + 1
                    break

        net_block = search_text[block_start:block_end]

        # Find all (via ...) within this net block
        for via_m in _VIA_RE.finditer(net_block):
            x_raw = float(via_m.group(1))
            y_raw = float(via_m.group(2))
            # FreeRouting uses resolution mm 1000 — detect and scale
            is_scaled = abs(x_raw) > 200.0 or abs(y_raw) > 200.0
            scale = 0.001 if is_scaled else 1.0
            vias.append(
                Via(
                    position=Point(x=x_raw * scale, y=y_raw * scale),
                    size=via_size,
                    drill=via_drill,
                    layers=("F.Cu", "B.Cu"),
                    net_number=net_number,
                )
            )

        pos = m.end()

    return tuple(vias)

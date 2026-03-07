"""Export a PCBDesign to Specctra DSN format for FreeRouting integration.

Produces a text-based DSN file that FreeRouting can read to perform
autorouting.  Only the elements required by FreeRouting are generated:
parser header, structure (layers, boundary, via template, rules), placement,
library (images + padstacks), network (nets), and an empty wiring section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.pcb import Footprint, Pad, PCBDesign

# Specctra DSN version string
DSN_VERSION: str = "ELECTRA"

# Default via template dimensions (drill x annular diameter in mm)
_VIA_DRILL_MM: float = 0.508
_VIA_DIAMETER_MM: float = 0.9

# Default design-rule values used in the DSN structure section
_DEFAULT_TRACE_WIDTH_MM: float = 0.25
_DEFAULT_CLEARANCE_MM: float = 0.2


# ---------------------------------------------------------------------------
# DSN builder helpers
# ---------------------------------------------------------------------------


def _indent(lines: list[str], level: int = 1) -> list[str]:
    """Prefix every line in *lines* with *level* * 2 spaces."""
    prefix = "  " * level
    return [prefix + ln for ln in lines]


def _board_bounds(pcb: PCBDesign) -> tuple[float, float, float, float]:
    """Return (x0, y0, x1, y1) bounding box of the board outline."""
    if not pcb.outline.polygon:
        return 0.0, 0.0, 100.0, 100.0
    xs = [p.x for p in pcb.outline.polygon]
    ys = [p.y for p in pcb.outline.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _fmt(v: float) -> str:
    """Format a float for DSN output (strip trailing zeros)."""
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _parser_section() -> list[str]:
    return [
        "(parser",
        '  (string_quote ")',
        "  (space_in_quoted_tokens on)",
        '  (host_cad "kicad-ai-pipeline")',
        '  (host_version "9.0")',
        ")",
    ]


def _pad_is_tht(pad: Pad) -> bool:
    """Return True if pad has layers on both F.Cu and B.Cu (through-hole)."""
    layers = {ly.upper() for ly in pad.layers}
    if "*.CU" in layers:
        return True
    return "F.CU" in layers and "B.CU" in layers


def _padstack_key(pad: Pad) -> str:
    """Generate a unique padstack key for a pad geometry."""
    layer_tag = "all" if _pad_is_tht(pad) else (pad.layers[0] if pad.layers else "F.Cu")
    return f"{pad.shape}_{_fmt(pad.size_x)}_{_fmt(pad.size_y)}_{layer_tag}"


def _padstack_lines(key: str, pad: Pad) -> list[str]:
    """Generate padstack definition lines for a pad."""
    is_tht = _pad_is_tht(pad)

    if is_tht:
        # Through-hole: define shapes on both F.Cu and B.Cu
        if pad.shape == "circle":
            r = _fmt(pad.size_x / 2)
            return [
                f'(padstack "{key}"',
                f'  (shape (circle "F.Cu" {r}))',
                f'  (shape (circle "B.Cu" {r}))',
                "  (attach off))",
            ]
        else:
            hx = _fmt(pad.size_x / 2)
            hy = _fmt(pad.size_y / 2)
            return [
                f'(padstack "{key}"',
                f'  (shape (rect "F.Cu" -{hx} -{hy} {hx} {hy}))',
                f'  (shape (rect "B.Cu" -{hx} -{hy} {hx} {hy}))',
                "  (attach off))",
            ]
    else:
        layer = pad.layers[0] if pad.layers else "F.Cu"
        if pad.shape == "circle":
            r = _fmt(pad.size_x / 2)
            return [
                f'(padstack "{key}"',
                f'  (shape (circle "{layer}" {r}))',
                "  (attach off))",
            ]
        else:
            hx = _fmt(pad.size_x / 2)
            hy = _fmt(pad.size_y / 2)
            return [
                f'(padstack "{key}"',
                f'  (shape (rect "{layer}" -{hx} -{hy} {hx} {hy}))',
                "  (attach off))",
            ]


def _structure_section(pcb: PCBDesign) -> list[str]:
    x0, y0, x1, y1 = _board_bounds(pcb)
    # Use netclass defaults if available
    trace_w = _DEFAULT_TRACE_WIDTH_MM
    clearance = _DEFAULT_CLEARANCE_MM
    via_dia = _VIA_DIAMETER_MM
    via_drill = _VIA_DRILL_MM
    if pcb.netclasses:
        for nc in pcb.netclasses:
            if nc.name == "Default":
                trace_w = nc.trace_width_mm
                clearance = nc.clearance_mm
                via_dia = nc.via_diameter_mm
                via_drill = nc.via_drill_mm
                break

    via_name = f"Via[0-1]_{_fmt(via_dia)}:{_fmt(via_drill)}_mm"
    lines: list[str] = [
        "(structure",
        '  (layer "F.Cu"',
        "    (type signal)",
        "    (property",
        "      (index 0)))",
        '  (layer "B.Cu"',
        "    (type signal)",
        "    (property",
        "      (index 1)))",
        "  (boundary",
        f'    (rect pcb {_fmt(x0)} {_fmt(y0)} {_fmt(x1)} {_fmt(y1)}))',
        f'  (via "{via_name}")',
        "  (rule",
        f"    (width {_fmt(trace_w)})",
        f"    (clearance {_fmt(clearance)}))",
    ]

    # Emit per-netclass rules for FreeRouting
    if pcb.netclasses:
        for nc in pcb.netclasses:
            if nc.name == "Default":
                continue
            if not nc.nets:
                continue
            lines.append(f'  (class "{nc.name}"')
            lines.append("    (rule")
            lines.append(f"      (width {_fmt(nc.trace_width_mm)})")
            lines.append(f"      (clearance {_fmt(nc.clearance_mm)}))")
            for net_name in nc.nets:
                lines.append(f'    (net "{net_name}")')
            lines.append("  )")

    lines.append(")")
    return lines


def _placement_section(pcb: PCBDesign) -> list[str]:
    if not pcb.footprints:
        return ["(placement)"]

    # Group footprints by lib_id (component type)
    from collections import defaultdict

    by_lib: dict[str, list[Footprint]] = defaultdict(list)
    for fp in pcb.footprints:
        by_lib[fp.lib_id].append(fp)

    lines: list[str] = ["(placement"]
    for lib_id, fps in by_lib.items():
        # Use just the part after ":" as the component name, or the full string
        comp_name = lib_id.split(":")[-1] if ":" in lib_id else lib_id
        lines.append(f'  (component "{comp_name}"')
        for fp in fps:
            side = "front" if fp.layer == "F.Cu" else "back"
            rot = _fmt(fp.rotation)
            lines.append(
                f'    (place "{fp.ref}" {_fmt(fp.position.x)} '
                f"{_fmt(fp.position.y)} {side} {rot})"
            )
        lines.append("  )")
    lines.append(")")
    return lines


def _library_section(pcb: PCBDesign) -> list[str]:
    """Build the library section with images (component pin maps) and padstacks.

    FreeRouting requires:
    - ``image`` blocks defining pin positions and padstack references per
      component type
    - ``padstack`` blocks defining pad geometries on specific layers
    - A ``via`` padstack for the via template referenced in the structure
    """
    from collections import defaultdict

    # Collect unique padstacks
    padstack_defs: dict[str, list[str]] = {}
    for fp in pcb.footprints:
        for pad in fp.pads:
            key = _padstack_key(pad)
            if key not in padstack_defs:
                padstack_defs[key] = _padstack_lines(key, pad)

    # Build image definitions grouped by lib_id
    by_lib: dict[str, list[Footprint]] = defaultdict(list)
    for fp in pcb.footprints:
        by_lib[fp.lib_id].append(fp)

    lines: list[str] = ["(library"]

    # Image blocks — one per unique lib_id (component type)
    for lib_id, fps in by_lib.items():
        comp_name = lib_id.split(":")[-1] if ":" in lib_id else lib_id
        # Use the first footprint as the template (all share the same pin layout)
        fp = fps[0]
        lines.append(f'  (image "{comp_name}"')
        for pad in fp.pads:
            key = _padstack_key(pad)
            lines.append(
                f'    (pin "{key}" "{pad.number}" '
                f"{_fmt(pad.position.x)} {_fmt(pad.position.y)})"
            )
        lines.append("  )")

    # Padstack definitions
    for ps_lines in padstack_defs.values():
        for pl in ps_lines:
            lines.append(f"  {pl}")

    # Via padstack definition
    via_dia = _VIA_DIAMETER_MM
    via_drill = _VIA_DRILL_MM
    if pcb.netclasses:
        for nc in pcb.netclasses:
            if nc.name == "Default":
                via_dia = nc.via_diameter_mm
                via_drill = nc.via_drill_mm
                break
    via_name = f"Via[0-1]_{_fmt(via_dia)}:{_fmt(via_drill)}_mm"
    via_r = _fmt(via_dia / 2)
    lines.extend([
        f'  (padstack "{via_name}"',
        f'    (shape (circle "F.Cu" {via_r}))',
        f'    (shape (circle "B.Cu" {via_r}))',
        "    (attach off))",
    ])

    lines.append(")")
    return lines


def _network_section(pcb: PCBDesign) -> list[str]:
    """Build the network section listing all nets and their pad connections."""
    if not pcb.nets:
        return ["(network)"]

    # Build pad membership: net_number -> list of (ref, pad_number)
    from collections import defaultdict

    net_pads: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for fp in pcb.footprints:
        for pad in fp.pads:
            if pad.net_number is not None and pad.net_number != 0:
                net_pads[pad.net_number].append((fp.ref, pad.number))

    # Build net name lookup
    net_name_by_num: dict[int, str] = {n.number: n.name for n in pcb.nets}

    lines: list[str] = ["(network"]
    for net in pcb.nets:
        if net.number == 0:
            continue
        pads = net_pads.get(net.number, [])
        if not pads:
            continue
        lines.append(f'  (net "{net.name}"')
        pin_refs = " ".join(f'"{ref}"-"{pad_num}"' for ref, pad_num in pads)
        lines.append(f"    (pins {pin_refs}))")
    _ = net_name_by_num  # used implicitly via net.name above
    lines.append(")")
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pcb_to_dsn(pcb: PCBDesign) -> str:
    """Convert a PCBDesign to a Specctra DSN string.

    Generates the essential elements required by FreeRouting:
    parser header, structure (layers + boundary + via + rules), placement,
    library (images + padstacks + via), network (nets), and an empty wiring
    section.

    Args:
        pcb: The PCB design to export.

    Returns:
        Complete DSN file content as a string.
    """
    lines: list[str] = ['(pcb "project.dsn"']
    for ln in _parser_section():
        lines.append(f"  {ln}")
    lines.append("  (resolution mm 1000)")
    lines.append("  (unit mm)")
    for ln in _structure_section(pcb):
        lines.append(f"  {ln}")
    for ln in _placement_section(pcb):
        lines.append(f"  {ln}")
    for ln in _library_section(pcb):
        lines.append(f"  {ln}")
    for ln in _network_section(pcb):
        lines.append(f"  {ln}")
    lines.append("  (wiring)")
    lines.append(")")
    return "\n".join(lines) + "\n"


def write_dsn(pcb: PCBDesign, path: Path | str) -> None:
    """Write the DSN representation of *pcb* to *path*.

    Args:
        pcb: The PCB design to export.
        path: Destination file path.
    """
    from pathlib import Path as _Path

    content = pcb_to_dsn(pcb)
    _Path(path).write_text(content, encoding="utf-8")

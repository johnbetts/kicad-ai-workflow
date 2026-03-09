"""Extract footprint positions, routing, and board features from existing KiCad PCB files.

Enables layout preservation during regeneration: footprint positions, tracks,
vias, user zones, and board slots from an existing board are extracted and
injected into the rebuilt PCBDesign so manual work is not lost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import Point, Track, Via, ZonePolygon, ZoneFill
from kicad_pipeline.sexp.parser import parse_file

if TYPE_CHECKING:
    from kicad_pipeline.sexp.writer import SExpNode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreservedRouting:
    """All user-created board features that should survive a rebuild."""

    tracks: tuple[Track, ...]
    vias: tuple[Via, ...]
    zones: tuple[ZonePolygon, ...]
    edge_cuts: tuple[tuple[Point, Point, float], ...]  # (start, end, width) segments
    net_map: dict[int, str]  # old_net_number → net_name (for remapping)


def _extract_ref(fp_node: SExpNode) -> str | None:
    """Extract reference designator from a footprint node.

    Handles both KiCad 9 ``(property "Reference" "R1" ...)`` and legacy
    ``(fp_text reference "R1" ...)`` formats.
    """
    if not isinstance(fp_node, list):
        return None
    for child in fp_node:
        if not isinstance(child, list) or len(child) < 3:
            continue
        tag = child[0]
        if tag == "property" and child[1] == "Reference":
            return str(child[2])
        if tag == "fp_text" and child[1] == "reference":
            return str(child[2])
    return None


def _extract_position(fp_node: SExpNode) -> tuple[float, float, float] | None:
    """Extract (x, y, rotation) from a footprint's ``(at ...)`` node."""
    if not isinstance(fp_node, list):
        return None
    for child in fp_node:
        if not isinstance(child, list) or len(child) < 3:
            continue
        if child[0] == "at":
            x = float(str(child[1]))
            y = float(str(child[2]))
            rotation = float(str(child[3])) if len(child) > 3 else 0.0
            return (x, y, rotation)
    return None


def _extract_ref_text_position(
    fp_node: SExpNode,
) -> tuple[float, float, float] | None:
    """Extract reference text (x, y, rotation) from a footprint node.

    Searches for ``(property "Reference" ... (at x y angle))`` or legacy
    ``(fp_text reference ... (at x y angle))``.

    Returns:
        ``(x, y, rotation)`` relative to the footprint origin, or ``None``.
    """
    if not isinstance(fp_node, list):
        return None
    for child in fp_node:
        if not isinstance(child, list) or len(child) < 3:
            continue
        tag = child[0]
        is_ref_property = tag == "property" and child[1] == "Reference"
        is_ref_fp_text = tag == "fp_text" and child[1] == "reference"
        if is_ref_property or is_ref_fp_text:
            # Find the (at x y angle) within this property/fp_text node
            for sub in child:
                if isinstance(sub, list) and len(sub) >= 3 and sub[0] == "at":
                    rx = float(str(sub[1]))
                    ry = float(str(sub[2]))
                    rrot = float(str(sub[3])) if len(sub) > 3 else 0.0
                    return (rx, ry, rrot)
    return None


def positions_from_pcb_file(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """Parse a ``.kicad_pcb`` file and extract ref -> (x_mm, y_mm, rotation_deg).

    Args:
        path: Path to the ``.kicad_pcb`` file.

    Returns:
        Mapping of reference designator to position tuple.
    """
    tree = parse_file(path)
    result: dict[str, tuple[float, float, float]] = {}
    if not isinstance(tree, list):
        return result
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "footprint":
            continue
        ref = _extract_ref(node)
        pos = _extract_position(node)
        if ref is not None and pos is not None:
            result[ref] = pos
    return result


def ref_text_positions_from_pcb_file(
    path: str | Path,
) -> dict[str, tuple[float, float, float]]:
    """Parse a ``.kicad_pcb`` file and extract ref text positions.

    Args:
        path: Path to the ``.kicad_pcb`` file.

    Returns:
        Mapping of reference designator to text ``(x, y, rotation)``
        relative to the footprint origin.
    """
    tree = parse_file(path)
    result: dict[str, tuple[float, float, float]] = {}
    if not isinstance(tree, list):
        return result
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "footprint":
            continue
        ref = _extract_ref(node)
        text_pos = _extract_ref_text_position(node)
        if ref is not None and text_pos is not None:
            result[ref] = text_pos
    return result


def positions_from_source(
    source: str | Path | object,
) -> dict[str, tuple[float, float, float]]:
    """Unified position extraction from file path or IPC connection.

    Args:
        source: Either a file path (str/Path) to a ``.kicad_pcb`` file, or a
            :class:`~kicad_pipeline.ipc.connection.KiCadConnection` instance.

    Returns:
        Mapping of reference designator to ``(x_mm, y_mm, rotation_deg)``.
    """
    if isinstance(source, str | Path):
        path = Path(source)
        if path.is_file():
            log.info("Extracting positions from PCB file: %s", path)
            return positions_from_pcb_file(path)
        log.warning("PCB file not found, no positions to preserve: %s", path)
        return {}

    # Assume IPC connection object
    from kicad_pipeline.ipc.board_ops import pull_footprint_positions

    log.info("Pulling positions from live KiCad IPC connection")
    return pull_footprint_positions(source)  # type: ignore[arg-type]


def ref_text_positions_from_source(
    source: str | Path | object,
) -> dict[str, tuple[float, float, float]]:
    """Unified ref text position extraction from file or IPC.

    Args:
        source: File path to ``.kicad_pcb`` or IPC connection.

    Returns:
        Mapping of ref to ``(text_x, text_y, text_rotation)`` relative
        to footprint origin.
    """
    if isinstance(source, str | Path):
        path = Path(source)
        if path.is_file():
            return ref_text_positions_from_pcb_file(path)
        return {}

    # IPC path — extract from live board
    from kicad_pipeline.ipc.board_ops import pull_ref_text_positions

    return pull_ref_text_positions(source)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Routing / board feature preservation
# ---------------------------------------------------------------------------


def _find_child(node: SExpNode, tag: str) -> SExpNode | None:
    """Find first child list with given tag in a node."""
    if not isinstance(node, list):
        return None
    for child in node:
        if isinstance(child, list) and child and child[0] == tag:
            return child
    return None


def _float_val(node: SExpNode, tag: str) -> float:
    """Extract float value from ``(tag value)`` child."""
    child = _find_child(node, tag)
    if child and len(child) >= 2:
        return float(str(child[1]))
    return 0.0


def _int_val(node: SExpNode, tag: str) -> int:
    """Extract int value from ``(tag value)`` child."""
    child = _find_child(node, tag)
    if child and len(child) >= 2:
        return int(float(str(child[1])))
    return 0


def _str_val(node: SExpNode, tag: str) -> str:
    """Extract string value from ``(tag value)`` child."""
    child = _find_child(node, tag)
    if child and len(child) >= 2:
        return str(child[1])
    return ""


def _point_val(node: SExpNode, tag: str) -> tuple[float, float]:
    """Extract (x, y) from ``(tag x y)`` child."""
    child = _find_child(node, tag)
    if child and len(child) >= 3:
        return float(str(child[1])), float(str(child[2]))
    return 0.0, 0.0


def _extract_net_map(tree: SExpNode) -> dict[int, str]:
    """Build net_number → net_name map from the PCB file.

    Handles both formats:
    - KiCad 9: Top-level ``(net N "name")`` entries
    - KiCad 10: Nets embedded in footprint pads ``(net "name")`` and zone
      ``(net N)``/``(net_name "name")`` pairs.

    Falls back to scanning pad and zone nodes if no top-level net entries found.
    """
    result: dict[int, str] = {}
    if not isinstance(tree, list):
        return result

    # Try top-level net declarations first (KiCad 9 format)
    for node in tree:
        if isinstance(node, list) and len(node) >= 3 and node[0] == "net":
            try:
                num = int(float(str(node[1])))
                name = str(node[2])
                result[num] = name
            except (ValueError, IndexError):
                continue

    if result:
        return result

    # KiCad 10 format: extract from footprint pad (net "name") and
    # zone (net N) + (net_name "name") pairs
    name_to_num: dict[str, int] = {}

    # From zones: KiCad 9: (zone (net N) (net_name "name") ...)
    #             KiCad 10: (zone (net "name") ...)
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "zone":
            continue
        net_child = _find_child(node, "net")
        if net_child and len(net_child) >= 2:
            val = str(net_child[1])
            try:
                net_num = int(float(val))
                # KiCad 9: (net N) — get name from net_name
                net_name = _str_val(node, "net_name")
                if net_name and net_num > 0:
                    result[net_num] = net_name
                    name_to_num[net_name] = net_num
            except ValueError:
                # KiCad 10: (net "name") — name directly
                net_name = val
                if net_name and net_name not in name_to_num:
                    next_num = max(result.keys(), default=0) + 1
                    result[next_num] = net_name
                    name_to_num[net_name] = next_num

    # From footprint pads: (pad ... (net N "name") ...)
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "footprint":
            continue
        for child in node:
            if not isinstance(child, list) or not child or child[0] != "pad":
                continue
            for sub in child:
                if isinstance(sub, list) and sub and sub[0] == "net":
                    if len(sub) >= 3:
                        # (net N "name") format
                        try:
                            num = int(float(str(sub[1])))
                            name = str(sub[2])
                            result[num] = name
                        except (ValueError, IndexError):
                            pass
                    elif len(sub) == 2:
                        # (net "name") format (KiCad 10)
                        name = str(sub[1])
                        # Assign a number if we know it from zones
                        if name in name_to_num:
                            result[name_to_num[name]] = name
                        elif name not in result.values():
                            # Assign next available number
                            next_num = max(result.keys(), default=0) + 1
                            result[next_num] = name
                            name_to_num[name] = next_num

    return result


def _net_info(
    node: SExpNode,
    name_to_num: dict[str, int],
) -> tuple[int, str]:
    """Extract net number and name from a node, handling both KiCad formats.

    KiCad 9: ``(net N)`` with separate ``(net_name "name")``
    KiCad 10: ``(net "name")`` with no separate net_name

    Uses *name_to_num* to resolve names to numbers. Unknown names get
    auto-assigned numbers.
    """
    net_child = _find_child(node, "net")
    if not net_child or len(net_child) < 2:
        return 0, ""

    val = str(net_child[1])
    try:
        # KiCad 9: (net N) — numeric
        net_num = int(float(val))
        net_name = _str_val(node, "net_name") or name_to_num.get(net_num, "")  # type: ignore[arg-type]
        return net_num, net_name
    except ValueError:
        # KiCad 10: (net "name") — string
        net_name = val
        if net_name in name_to_num:
            return name_to_num[net_name], net_name
        # Auto-assign
        next_num = max(name_to_num.values(), default=0) + 1
        name_to_num[net_name] = next_num
        return next_num, net_name


def _extract_tracks(
    tree: SExpNode,
    name_to_num: dict[str, int],
) -> list[Track]:
    """Extract all ``(segment ...)`` nodes as Track objects."""
    tracks: list[Track] = []
    if not isinstance(tree, list):
        return tracks
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "segment":
            continue
        sx, sy = _point_val(node, "start")
        ex, ey = _point_val(node, "end")
        width = _float_val(node, "width")
        layer = _str_val(node, "layer")
        net_num, _ = _net_info(node, name_to_num)
        uuid = _str_val(node, "uuid")
        tracks.append(Track(
            start=Point(sx, sy), end=Point(ex, ey),
            width=width, layer=layer, net_number=net_num, uuid=uuid,
        ))
    return tracks


def _extract_vias(
    tree: SExpNode,
    name_to_num: dict[str, int],
) -> list[Via]:
    """Extract all ``(via ...)`` nodes as Via objects."""
    vias: list[Via] = []
    if not isinstance(tree, list):
        return vias
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "via":
            continue
        x, y = _point_val(node, "at")
        size = _float_val(node, "size")
        drill = _float_val(node, "drill")
        net_num, _ = _net_info(node, name_to_num)
        uuid = _str_val(node, "uuid")
        # Extract layers
        layers_child = _find_child(node, "layers")
        layers: tuple[str, ...] = ()
        if layers_child and len(layers_child) > 1:
            layers = tuple(str(l) for l in layers_child[1:])
        vias.append(Via(
            position=Point(x, y), drill=drill, size=size,
            layers=layers, net_number=net_num, uuid=uuid,
        ))
    return vias


def _is_auto_generated_zone(node: SExpNode) -> bool:
    """Detect zones auto-generated by the pipeline (GND pours, power planes).

    Detection methods:
    1. ``(name "...")`` matching known pipeline patterns (if present)
    2. ``(keepout ...)`` marker — keepouts are always regenerated
    3. Full-board GND pours and power planes identified by net+layer combo
       (KiCad may strip the name tag on save)
    """
    if not isinstance(node, list):
        return False
    # Check for keepout marker — keepouts are always regenerated
    for child in node:
        if isinstance(child, list) and child and child[0] == "keepout":
            return True
    # Check zone name against known auto-generated patterns
    name = _str_val(node, "name")
    auto_names = {
        "GND_pour_F.Cu", "GND_pour_B.Cu",
        "GND_plane_In1.Cu", "+5V_plane_In2.Cu",
    }
    if name in auto_names:
        return True
    # Detect by net+layer combo: pipeline always generates these full-board zones
    layer = _str_val(node, "layer")
    net_child = _find_child(node, "net")
    net_name_child = _str_val(node, "net_name")
    net_name = ""
    if net_child and len(net_child) >= 2:
        val = str(net_child[1])
        try:
            int(float(val))
            net_name = net_name_child
        except ValueError:
            net_name = val
    auto_layer_net = {
        ("F.Cu", "GND"), ("B.Cu", "GND"),
        ("In1.Cu", "GND"), ("In2.Cu", "+5V"),
    }
    if (layer, net_name) in auto_layer_net:
        return True
    return False


def _extract_user_zones(
    tree: SExpNode,
    name_to_num: dict[str, int],
) -> list[ZonePolygon]:
    """Extract user-created zones (not auto-generated pours/planes).

    User zones include custom copper fills, isolation zones, etc.
    """
    zones: list[ZonePolygon] = []
    if not isinstance(tree, list):
        return zones
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "zone":
            continue
        if _is_auto_generated_zone(node):
            continue
        net_num, net_name = _net_info(node, name_to_num)
        layer = _str_val(node, "layer")
        name = _str_val(node, "name")
        uuid = _str_val(node, "uuid")
        # Extract polygon points
        polygon_node = _find_child(node, "polygon")
        points: list[Point] = []
        if polygon_node:
            pts_node = _find_child(polygon_node, "pts")
            if pts_node and isinstance(pts_node, list):
                for child in pts_node[1:]:
                    if isinstance(child, list) and child and child[0] == "xy":
                        points.append(Point(
                            float(str(child[1])), float(str(child[2])),
                        ))
        # Extract fill settings
        fill_node = _find_child(node, "fill")
        thermal_gap = 0.3
        thermal_bridge = 0.5
        if fill_node:
            thermal_gap = _float_val(fill_node, "thermal_gap")
            thermal_bridge = _float_val(fill_node, "thermal_bridge_width")
        connect_pads = _find_child(node, "connect_pads")
        clearance = 0.3
        if connect_pads:
            clearance = _float_val(connect_pads, "clearance")
        min_thickness = _float_val(node, "min_thickness") or 0.25
        # Extract filled_polygon data
        filled_polys: list[tuple[Point, ...]] = []
        for child in node:
            if isinstance(child, list) and child and child[0] == "filled_polygon":
                fp_pts_node = _find_child(child, "pts")
                if fp_pts_node and isinstance(fp_pts_node, list):
                    fp_points: list[Point] = []
                    for sub in fp_pts_node[1:]:
                        if isinstance(sub, list) and sub and sub[0] == "xy":
                            fp_points.append(Point(
                                float(str(sub[1])), float(str(sub[2])),
                            ))
                    if fp_points:
                        filled_polys.append(tuple(fp_points))
        if points:
            zones.append(ZonePolygon(
                net_number=net_num, net_name=net_name, layer=layer,
                name=name, polygon=tuple(points),
                min_thickness=min_thickness, fill=ZoneFill.SOLID,
                thermal_relief_gap=thermal_gap,
                thermal_relief_bridge=thermal_bridge,
                clearance_mm=clearance,
                filled_polygons=tuple(filled_polys),
                uuid=uuid,
            ))
    return zones


def _extract_edge_cuts(tree: SExpNode) -> list[tuple[Point, Point, float]]:
    """Extract Edge.Cuts line segments that are NOT the board outline.

    Board outline segments form a closed rectangle; any extra segments
    (board slots, cutouts) are user-created and should be preserved.
    """
    segments: list[tuple[Point, Point, float]] = []
    if not isinstance(tree, list):
        return segments
    for node in tree:
        if not isinstance(node, list) or not node:
            continue
        # Match (gr_line (start ...) (end ...) (layer "Edge.Cuts") ...)
        if node[0] not in ("gr_line", "fp_line"):
            continue
        layer = _str_val(node, "layer")
        if layer != "Edge.Cuts":
            continue
        sx, sy = _point_val(node, "start")
        ex, ey = _point_val(node, "end")
        width = _float_val(node, "stroke_width") or _float_val(node, "width") or 0.05
        # Check stroke node for width
        stroke = _find_child(node, "stroke")
        if stroke:
            sw = _float_val(stroke, "width")
            if sw > 0:
                width = sw
        segments.append((Point(sx, sy), Point(ex, ey), width))
    return segments


def _is_outline_segment(
    seg: tuple[Point, Point, float],
    outline_pts: set[tuple[float, float]],
    tolerance: float = 0.1,
) -> bool:
    """Check if an Edge.Cuts segment is part of the rectangular board outline."""
    s, e = seg[0], seg[1]
    s_match = any(
        abs(s.x - ox) < tolerance and abs(s.y - oy) < tolerance
        for ox, oy in outline_pts
    )
    e_match = any(
        abs(e.x - ox) < tolerance and abs(e.y - oy) < tolerance
        for ox, oy in outline_pts
    )
    return s_match and e_match


def routing_from_pcb_file(path: str | Path) -> PreservedRouting:
    """Parse a ``.kicad_pcb`` file and extract all user routing and board features.

    Returns:
        A :class:`PreservedRouting` with tracks, vias, user zones, board slots,
        and a net name map for remapping net numbers in the new design.
    """
    path = Path(path)
    if not path.is_file():
        log.warning("PCB file not found, no routing to preserve: %s", path)
        return PreservedRouting(
            tracks=(), vias=(), zones=(), edge_cuts=(), net_map={},
        )

    tree = parse_file(path)
    net_map = _extract_net_map(tree)
    # Build reverse lookup for name → number (used by KiCad 10 format)
    name_to_num = {name: num for num, name in net_map.items()}
    tracks = _extract_tracks(tree, name_to_num)
    vias = _extract_vias(tree, name_to_num)
    user_zones = _extract_user_zones(tree, name_to_num)
    # Update net_map with any newly discovered nets
    for name, num in name_to_num.items():
        if num not in net_map:
            net_map[num] = name
    all_edge_cuts = _extract_edge_cuts(tree)

    # Filter out board outline segments — keep only slots/cutouts
    # Detect outline corners from the auto-generated rectangular outline
    # (4 corners = min_x,min_y / max_x,min_y / max_x,max_y / min_x,max_y)
    if all_edge_cuts:
        all_x = [s.x for s, e, w in all_edge_cuts] + [e.x for s, e, w in all_edge_cuts]
        all_y = [s.y for s, e, w in all_edge_cuts] + [e.y for s, e, w in all_edge_cuts]
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            outline_pts = {
                (min_x, min_y), (max_x, min_y),
                (max_x, max_y), (min_x, max_y),
            }
            user_edge_cuts = [
                seg for seg in all_edge_cuts
                if not _is_outline_segment(seg, outline_pts)
            ]
        else:
            user_edge_cuts = all_edge_cuts
    else:
        user_edge_cuts = []

    log.info(
        "Preserved routing: %d tracks, %d vias, %d user zones, %d edge cuts",
        len(tracks), len(vias), len(user_zones), len(user_edge_cuts),
    )
    return PreservedRouting(
        tracks=tuple(tracks),
        vias=tuple(vias),
        zones=tuple(user_zones),
        edge_cuts=tuple(user_edge_cuts),
        net_map=net_map,
    )


def routing_from_source(
    source: str | Path | object | None,
    pcb_file_path: str | Path | None = None,
) -> PreservedRouting | None:
    """Extract preserved routing from a PCB file.

    When *source* is an IPC connection, the on-disk file at *pcb_file_path*
    is used instead (the file has the user's manual work before rebuild).

    Args:
        source: File path to ``.kicad_pcb``, IPC connection, or ``None``.
        pcb_file_path: Explicit path to the PCB file on disk (used when
            *source* is an IPC connection).

    Returns:
        :class:`PreservedRouting` or ``None`` if no source available.
    """
    if source is None:
        return None

    if isinstance(source, str | Path):
        path = Path(source)
        if path.is_file():
            return routing_from_pcb_file(path)
        return None

    # IPC connection — read the on-disk file instead (has user's manual work)
    if pcb_file_path is not None:
        path = Path(pcb_file_path)
        if path.is_file():
            log.info("Extracting routing from on-disk PCB file: %s", path)
            return routing_from_pcb_file(path)

    return None


def remap_routing(
    routing: PreservedRouting,
    new_nets: dict[str, int],
) -> tuple[tuple[Track, ...], tuple[Via, ...], tuple[ZonePolygon, ...]]:
    """Remap net numbers in preserved routing to match the new PCBDesign.

    Net numbers can change between rebuilds (components added/removed,
    net ordering changed). This remaps old net numbers to new ones by
    matching on net name.

    Args:
        routing: Preserved routing from the previous build.
        new_nets: Mapping of net_name → new_net_number.

    Returns:
        Tuple of (remapped_tracks, remapped_vias, remapped_zones).
        Items whose net name cannot be found in *new_nets* are dropped
        with a warning.
    """
    old_to_new: dict[int, int] = {}
    dropped_nets: set[str] = set()
    for old_num, old_name in routing.net_map.items():
        if old_name in new_nets:
            old_to_new[old_num] = new_nets[old_name]
        elif old_num != 0:  # net 0 = unconnected, always valid
            dropped_nets.add(old_name)
    old_to_new[0] = 0  # unconnected net always maps to 0

    if dropped_nets:
        log.warning(
            "Routing references %d nets that no longer exist: %s",
            len(dropped_nets),
            ", ".join(sorted(dropped_nets)[:10]),
        )

    # Remap tracks
    tracks: list[Track] = []
    for t in routing.tracks:
        new_net = old_to_new.get(t.net_number)
        if new_net is not None:
            tracks.append(Track(
                start=t.start, end=t.end, width=t.width,
                layer=t.layer, net_number=new_net, uuid=t.uuid,
            ))

    # Remap vias
    vias: list[Via] = []
    for v in routing.vias:
        new_net = old_to_new.get(v.net_number)
        if new_net is not None:
            vias.append(Via(
                position=v.position, drill=v.drill, size=v.size,
                layers=v.layers, net_number=new_net, uuid=v.uuid,
            ))

    # Remap zones
    zones: list[ZonePolygon] = []
    for z in routing.zones:
        new_net = old_to_new.get(z.net_number)
        new_name = z.net_name
        if new_net is not None:
            # Update net name to match new design
            for name, num in new_nets.items():
                if num == new_net:
                    new_name = name
                    break
            zones.append(ZonePolygon(
                net_number=new_net, net_name=new_name, layer=z.layer,
                name=z.name, polygon=z.polygon,
                min_thickness=z.min_thickness, fill=z.fill,
                thermal_relief_gap=z.thermal_relief_gap,
                thermal_relief_bridge=z.thermal_relief_bridge,
                clearance_mm=z.clearance_mm,
                filled_polygons=z.filled_polygons,
                uuid=z.uuid,
            ))

    log.info(
        "Remapped routing: %d/%d tracks, %d/%d vias, %d/%d zones",
        len(tracks), len(routing.tracks),
        len(vias), len(routing.vias),
        len(zones), len(routing.zones),
    )
    return tuple(tracks), tuple(vias), tuple(zones)

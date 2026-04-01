"""Extract footprint positions, routing, and board features from existing KiCad PCB files.

Enables layout preservation during regeneration: footprint positions, tracks,
vias, user zones, and board slots from an existing board are extracted and
injected into the rebuilt PCBDesign so manual work is not lost.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import Keepout, Point, Track, Via, ZoneFill, ZonePolygon
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
    keepouts: tuple[Keepout, ...] = ()
    edge_cuts: tuple[tuple[Point, Point, float], ...] = ()  # (start, end, width) segments
    net_map: dict[int, str] | None = None  # old_net_number → net_name (for remapping)


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


@dataclass(frozen=True)
class BoardFootprintModel:
    """3D model info extracted from an actual .kicad_pcb footprint."""

    ref: str
    footprint_lib: str  # e.g. "R_0805_2012Metric"
    position: tuple[float, float, float]  # (x, y, rotation)
    model_path: str  # STEP file path
    model_offset: tuple[float, float, float]  # (ox, oy, oz)
    model_rotate: tuple[float, float, float]  # (rx, ry, rz)


_ModelInfo = tuple[str, tuple[float, float, float], tuple[float, float, float]]


def _extract_models(fp_node: SExpNode) -> list[_ModelInfo]:
    """Extract 3D model path, offset, and rotation from a footprint node."""
    models: list[_ModelInfo] = []
    if not isinstance(fp_node, list):
        return models
    for child in fp_node:
        if not isinstance(child, list) or len(child) < 2 or child[0] != "model":
            continue
        model_path = str(child[1])
        offset = (0.0, 0.0, 0.0)
        rotate = (0.0, 0.0, 0.0)
        for sub in child[2:]:
            if not isinstance(sub, list) or len(sub) < 2:
                continue
            if sub[0] == "offset" and isinstance(sub[1], list) and sub[1][0] == "xyz":
                offset = (
                    float(str(sub[1][1])) if len(sub[1]) > 1 else 0.0,
                    float(str(sub[1][2])) if len(sub[1]) > 2 else 0.0,
                    float(str(sub[1][3])) if len(sub[1]) > 3 else 0.0,
                )
            elif sub[0] == "rotate" and isinstance(sub[1], list) and sub[1][0] == "xyz":
                rotate = (
                    float(str(sub[1][1])) if len(sub[1]) > 1 else 0.0,
                    float(str(sub[1][2])) if len(sub[1]) > 2 else 0.0,
                    float(str(sub[1][3])) if len(sub[1]) > 3 else 0.0,
                )
        models.append((model_path, offset, rotate))
    return models


def models_from_pcb_file(
    path: str | Path,
) -> dict[str, BoardFootprintModel]:
    """Parse a ``.kicad_pcb`` and extract 3D model info for every footprint.

    Returns:
        Mapping of reference designator to :class:`BoardFootprintModel`.
        Only includes footprints that have at least one 3D model.
    """
    tree = parse_file(path)
    result: dict[str, BoardFootprintModel] = {}
    if not isinstance(tree, list):
        return result
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "footprint":
            continue
        ref = _extract_ref(node)
        pos = _extract_position(node)
        if ref is None or pos is None:
            continue
        models = _extract_models(node)
        if not models:
            continue
        # Use the first model (primary 3D model)
        model_path, offset, rotate = models[0]
        # Extract footprint library name (first arg after "footprint")
        lib_name = str(node[1]) if len(node) > 1 else ""
        result[ref] = BoardFootprintModel(
            ref=ref,
            footprint_lib=lib_name,
            position=pos,
            model_path=model_path,
            model_offset=offset,
            model_rotate=rotate,
        )
    return result


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


def _extract_toplevel_nets(tree: list[object]) -> dict[int, str]:
    """Extract net map from top-level (net N "name") entries (KiCad 9 format)."""
    result: dict[int, str] = {}
    for node in tree:
        if isinstance(node, list) and len(node) >= 3 and node[0] == "net":
            try:
                num = int(float(str(node[1])))
                name = str(node[2])
                result[num] = name
            except (ValueError, IndexError):
                continue
    return result


def _extract_nets_from_zones(
    tree: list[object],
    result: dict[int, str],
    name_to_num: dict[str, int],
) -> None:
    """Extract net mappings from zone nodes (KiCad 9 and 10 formats)."""
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "zone":
            continue
        net_child = _find_child(node, "net")
        if not (net_child and len(net_child) >= 2):
            continue
        val = str(net_child[1])
        try:
            net_num = int(float(val))
            net_name = _str_val(node, "net_name")
            if net_name and net_num > 0:
                result[net_num] = net_name
                name_to_num[net_name] = net_num
        except ValueError:
            net_name = val
            if net_name and net_name not in name_to_num:
                next_num = max(result.keys(), default=0) + 1
                result[next_num] = net_name
                name_to_num[net_name] = next_num


def _extract_net_map(tree: SExpNode) -> dict[int, str]:
    """Build net_number -> net_name map from the PCB file.

    Handles both formats:
    - KiCad 9: Top-level ``(net N "name")`` entries
    - KiCad 10: Nets embedded in footprint pads ``(net "name")`` and zone
      ``(net N)``/``(net_name "name")`` pairs.

    Falls back to scanning pad and zone nodes if no top-level net entries found.
    """
    if not isinstance(tree, list):
        return {}

    result = _extract_toplevel_nets(tree)
    if result:
        return result

    name_to_num: dict[str, int] = {}
    _extract_nets_from_zones(tree, result, name_to_num)
    _extract_nets_from_pads(tree, result, name_to_num)
    return result


def _record_pad_net(
    net_sub: list[SExpNode],
    result: dict[int, str],
    name_to_num: dict[str, int],
) -> None:
    """Record a single pad net entry into *result* and *name_to_num*."""
    if len(net_sub) >= 3:
        # (net N "name") format
        try:
            num = int(float(str(net_sub[1])))
            name = str(net_sub[2])
            result[num] = name
        except (ValueError, IndexError) as exc:
            log.debug("Net number/name parse failed for sub-node: %s", exc)
    elif len(net_sub) == 2:
        # (net "name") format (KiCad 10)
        name = str(net_sub[1])
        if name in name_to_num:
            result[name_to_num[name]] = name
        elif name not in result.values():
            next_num = max(result.keys(), default=0) + 1
            result[next_num] = name
            name_to_num[name] = next_num


def _extract_nets_from_pads(
    tree: SExpNode,
    result: dict[int, str],
    name_to_num: dict[str, int],
) -> None:
    """Extract net mappings from footprint pad nodes."""
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "footprint":
            continue
        for child in node:
            if not isinstance(child, list) or not child or child[0] != "pad":
                continue
            net_sub = _find_child(child, "net")
            if not isinstance(net_sub, list):
                continue
            _record_pad_net(net_sub, result, name_to_num)


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
            layers = tuple(str(layer) for layer in layers_child[1:])
        vias.append(Via(
            position=Point(x, y), drill=drill, size=size,
            layers=layers, net_number=net_num, uuid=uuid,
        ))
    return vias


def _is_auto_generated_zone(node: SExpNode) -> bool:
    """Detect zones auto-generated by the pipeline (GND pours, power planes).

    Detection methods:
    1. ``(name "...")`` matching known pipeline patterns (if present)
    2. Full-board GND pours and power planes identified by net+layer combo
       (KiCad may strip the name tag on save)

    Note: keepout zones are NOT filtered — user-created keepouts (exclusion
    zones, no-fill areas) must be preserved.  Only pipeline-generated keepouts
    (mounting holes, antenna) are identified by their ``tag`` attribute during
    generation, not here.
    """
    if not isinstance(node, list):
        return False
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
    return (layer, net_name) in auto_layer_net


def _extract_polygon_points(node: SExpNode) -> list[Point]:
    """Extract points from a (polygon (pts (xy x y) ...)) structure."""
    polygon_node = _find_child(node, "polygon")
    if not polygon_node:
        return []
    pts_node = _find_child(polygon_node, "pts")
    if not pts_node or not isinstance(pts_node, list):
        return []
    points: list[Point] = []
    for child in pts_node[1:]:
        if isinstance(child, list) and child and child[0] == "xy":
            points.append(Point(float(str(child[1])), float(str(child[2]))))
    return points


def _extract_zone_fill_settings(
    node: SExpNode,
) -> tuple[float, float, float]:
    """Extract thermal relief and clearance settings from a zone node.

    Returns:
        ``(thermal_gap, thermal_bridge, clearance)`` with defaults applied.
    """
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
    return thermal_gap, thermal_bridge, clearance


def _extract_filled_polygons(node: SExpNode) -> list[tuple[Point, ...]]:
    """Extract filled_polygon data from a zone node."""
    filled_polys: list[tuple[Point, ...]] = []
    if not isinstance(node, list):
        return filled_polys
    for child in node:
        if not isinstance(child, list) or not child or child[0] != "filled_polygon":
            continue
        fp_pts_node = _find_child(child, "pts")
        if not fp_pts_node or not isinstance(fp_pts_node, list):
            continue
        fp_points = [
            Point(float(str(sub[1])), float(str(sub[2])))
            for sub in fp_pts_node[1:]
            if isinstance(sub, list) and sub and sub[0] == "xy"
        ]
        if fp_points:
            filled_polys.append(tuple(fp_points))
    return filled_polys


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
        # Skip keepout zones — they're extracted separately with multi-layer support
        is_keepout = any(
            isinstance(c, list) and c and c[0] == "keepout" for c in node
        )
        if is_keepout:
            continue
        net_num, net_name = _net_info(node, name_to_num)
        layer = _str_val(node, "layer")
        name = _str_val(node, "name")
        uuid = _str_val(node, "uuid")
        # Extract polygon points
        points = _extract_polygon_points(node)
        thermal_gap, thermal_bridge, clearance = _extract_zone_fill_settings(node)
        min_thickness = _float_val(node, "min_thickness") or 0.25
        priority = int(_float_val(node, "priority") or 0)
        filled_polys = _extract_filled_polygons(node)
        if points:
            zones.append(ZonePolygon(
                net_number=net_num, net_name=net_name, layer=layer,
                name=name, polygon=tuple(points),
                min_thickness=min_thickness, fill=ZoneFill.SOLID,
                thermal_relief_gap=thermal_gap,
                thermal_relief_bridge=thermal_bridge,
                clearance_mm=clearance,
                priority=priority,
                filled_polygons=tuple(filled_polys),
                uuid=uuid,
            ))
    return zones


def _extract_zone_layers(node: list[object]) -> list[str]:
    """Extract layer names from a zone node (supports both 'layer' and 'layers')."""
    layers: list[str] = []
    for child in node:
        if not isinstance(child, list) or not child:
            continue
        if child[0] == "layers":
            layers.extend(str(v) for v in child[1:] if isinstance(v, str))
        elif child[0] == "layer":
            layers.append(str(child[1]))
    return layers


def _parse_keepout_rules(
    keepout_node: list[object],
) -> tuple[bool, bool, bool]:
    """Parse keepout rules from a keepout S-expression node.

    Returns (no_copper, no_tracks, no_vias).
    """
    no_copper = False
    no_tracks = False
    no_vias = False
    for child in keepout_node:
        if not isinstance(child, list) or len(child) != 2:
            continue
        if child[0] == "copperpour" and child[1] == "not_allowed":
            no_copper = True
        elif child[0] == "tracks" and child[1] == "not_allowed":
            no_tracks = True
        elif child[0] == "vias" and child[1] == "not_allowed":
            no_vias = True
    return no_copper, no_tracks, no_vias


def _extract_user_keepouts(tree: SExpNode) -> list[Keepout]:
    """Extract user-created keepout zones with full multi-layer support."""
    keepouts: list[Keepout] = []
    if not isinstance(tree, list):
        return keepouts
    for node in tree:
        if not isinstance(node, list) or not node or node[0] != "zone":
            continue
        # Only process keepout zones
        keepout_node = None
        for child in node:
            if isinstance(child, list) and child and child[0] == "keepout":
                keepout_node = child
                break
        if keepout_node is None:
            continue

        layers = _extract_zone_layers(node)
        uuid = _str_val(node, "uuid")
        points = _extract_polygon_points(node)
        no_copper, no_tracks, no_vias = _parse_keepout_rules(keepout_node)

        if points:
            keepouts.append(Keepout(
                polygon=tuple(points),
                layers=tuple(layers),
                no_copper=no_copper,
                no_tracks=no_tracks,
                no_vias=no_vias,
                uuid=uuid,
            ))
    return keepouts


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


def _touches_outline_edge(
    seg: tuple[Point, Point, float],
    min_x: float, max_x: float,
    min_y: float, max_y: float,
    tolerance: float = 0.1,
) -> bool:
    """Check if at least one endpoint of *seg* lies on the board outline edge.

    A segment that touches the outline indicates a notch or cutout.
    Isolated interior fragments (artefacts) return False.
    """
    for pt in (seg[0], seg[1]):
        on_left = abs(pt.x - min_x) < tolerance
        on_right = abs(pt.x - max_x) < tolerance
        on_top = abs(pt.y - min_y) < tolerance
        on_bottom = abs(pt.y - max_y) < tolerance
        if on_left or on_right or on_top or on_bottom:
            return True
    return False


_MIN_SEGMENT_LENGTH_MM = 0.5


def _filter_user_edge_cuts(
    all_edge_cuts: list[tuple[Point, Point, float]],
) -> list[tuple[Point, Point, float]]:
    """Filter board outline segments, keeping only user-created slots/cutouts.

    Removes segments that form the rectangular board outline and discards
    micro-fragments (< 0.5mm) that are routing artefacts.
    """
    if not all_edge_cuts:
        return []

    all_x = [s.x for s, e, w in all_edge_cuts] + [e.x for s, e, w in all_edge_cuts]
    all_y = [s.y for s, e, w in all_edge_cuts] + [e.y for s, e, w in all_edge_cuts]
    if not all_x or not all_y:
        return list(all_edge_cuts)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    outline_pts = {
        (min_x, min_y), (max_x, min_y),
        (max_x, max_y), (min_x, max_y),
    }
    non_outline = [
        seg for seg in all_edge_cuts
        if not _is_outline_segment(seg, outline_pts)
    ]

    user_edge_cuts: list[tuple[Point, Point, float]] = []
    filtered_count = 0
    for seg in non_outline:
        s, e, _w = seg
        if math.hypot(e.x - s.x, e.y - s.y) < _MIN_SEGMENT_LENGTH_MM:
            filtered_count += 1
        else:
            user_edge_cuts.append(seg)

    if filtered_count:
        log.info(
            "Filtered %d micro Edge.Cuts fragments (< %.1fmm)",
            filtered_count,
            _MIN_SEGMENT_LENGTH_MM,
        )
    return user_edge_cuts


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
    user_keepouts = _extract_user_keepouts(tree)
    all_edge_cuts = _extract_edge_cuts(tree)

    user_edge_cuts = _filter_user_edge_cuts(all_edge_cuts)

    log.info(
        "Preserved routing: %d tracks, %d vias, %d user zones, %d keepouts, %d edge cuts",
        len(tracks), len(vias), len(user_zones), len(user_keepouts),
        len(user_edge_cuts),
    )
    return PreservedRouting(
        tracks=tuple(tracks),
        vias=tuple(vias),
        zones=tuple(user_zones),
        edge_cuts=tuple(user_edge_cuts),
        net_map=net_map,
        keepouts=tuple(user_keepouts),
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

    tracks, vias, zones = _remap_routing_elements(routing, old_to_new, new_nets)
    log.info(
        "Remapped routing: %d/%d tracks, %d/%d vias, %d/%d zones",
        len(tracks), len(routing.tracks),
        len(vias), len(routing.vias),
        len(zones), len(routing.zones),
    )
    return tuple(tracks), tuple(vias), tuple(zones)


def _remap_routing_elements(
    routing: PreservedRouting,
    old_to_new: dict[int, int],
    new_nets: dict[str, int],
) -> tuple[list[Track], list[Via], list[ZonePolygon]]:
    """Remap tracks, vias, and zones using the provided net-number map."""
    tracks: list[Track] = []
    for t in routing.tracks:
        new_net = old_to_new.get(t.net_number)
        if new_net is not None:
            tracks.append(Track(
                start=t.start, end=t.end, width=t.width,
                layer=t.layer, net_number=new_net, uuid=t.uuid,
            ))

    vias: list[Via] = []
    for v in routing.vias:
        new_net = old_to_new.get(v.net_number)
        if new_net is not None:
            vias.append(Via(
                position=v.position, drill=v.drill, size=v.size,
                layers=v.layers, net_number=new_net, uuid=v.uuid,
            ))

    zones: list[ZonePolygon] = []
    for z in routing.zones:
        new_net = old_to_new.get(z.net_number)
        new_name = z.net_name
        if new_net is not None:
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
                priority=z.priority,
                filled_polygons=z.filled_polygons,
                uuid=z.uuid,
            ))

    return tracks, vias, zones

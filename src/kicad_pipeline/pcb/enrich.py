"""Post-processor for existing .kicad_pcb files.

Injects 3D model references and/or flips connector footprints to B.Cu
without modifying placement, routing, or any other PCB content.

Operates directly on the parsed S-expression tree — does **not** require
a :class:`~kicad_pipeline.models.pcb.PCBDesign` round-trip.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.constants import KICAD_3DMODEL_VAR
from kicad_pipeline.pcb.footprints import _LAYER_FLIP_MAP, _model_for_package
from kicad_pipeline.sexp.parser import parse_file
from kicad_pipeline.sexp.writer import write_file

if TYPE_CHECKING:
    from kicad_pipeline.sexp.writer import SExpNode

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S-expression tree helpers
# ---------------------------------------------------------------------------


def _find_footprint_nodes(tree: SExpNode) -> list[list[SExpNode]]:
    """Walk an S-expression tree and collect all ``(footprint ...)`` sublists."""
    result: list[list[SExpNode]] = []
    if not isinstance(tree, list):
        return result
    if len(tree) > 0 and tree[0] == "footprint":
        result.append(tree)
    for child in tree:
        if isinstance(child, list):
            result.extend(_find_footprint_nodes(child))
    return result


def _extract_ref(fp_node: list[SExpNode]) -> str:
    """Extract the reference designator from a footprint node.

    Looks for ``(property "Reference" "R1" ...)`` or legacy ``(fp_text reference "R1" ...)``.
    """
    for child in fp_node:
        if not isinstance(child, list) or len(child) < 3:
            continue
        if child[0] == "property" and child[1] == "Reference":
            return str(child[2])
        if child[0] == "fp_text" and child[1] == "reference":
            return str(child[2])
    return ""


def _extract_lib_id(fp_node: list[SExpNode]) -> str:
    """Extract the lib_id (second element) from a footprint node."""
    if len(fp_node) > 1 and isinstance(fp_node[1], str):
        return fp_node[1]
    return ""


def _has_model(fp_node: list[SExpNode]) -> bool:
    """Check if the footprint already has a ``(model ...)`` child."""
    for child in fp_node:
        if isinstance(child, list) and len(child) > 0 and child[0] == "model":
            return True
    return False


def _inject_model(fp_node: list[SExpNode], model_path: str) -> None:
    """Append a ``(model ...)`` sublist to a footprint node."""
    model_node: list[SExpNode] = [
        "model",
        model_path,
        ["offset", ["xyz", 0, 0, 0]],
        ["scale", ["xyz", 1, 1, 1]],
        ["rotate", ["xyz", 0, 0, 0]],
    ]
    fp_node.append(model_node)


def _flip_footprint_layer(fp_node: list[SExpNode]) -> None:
    """Flip all layer references in a footprint node from F↔B.

    Mutates layer strings in ``(layer ...)``, ``(property ... (layer ...))``,
    graphic layer refs, etc.  Also rewrites PinHeader→PinSocket in the lib_id.
    """
    # Flip lib_id: PinHeader → PinSocket
    if len(fp_node) > 1 and isinstance(fp_node[1], str):
        old_id = fp_node[1]
        if "PinHeader" in old_id:
            fp_node[1] = old_id.replace("PinHeader", "PinSocket").replace(
                "Connector_PinHeader", "Connector_PinSocket"
            )

    _flip_layers_recursive(fp_node)


def _flip_layers_recursive(node: list[SExpNode]) -> None:
    """Recursively flip F↔B layer strings in all ``(layer ...)`` sublists."""
    for child in node:
        if isinstance(child, list):
            if len(child) == 2 and child[0] == "layer" and isinstance(child[1], str):
                flipped = _LAYER_FLIP_MAP.get(child[1])
                if flipped is not None:
                    child[1] = flipped
            else:
                _flip_layers_recursive(child)


def _model_path_from_lib_id(lib_id: str) -> str | None:
    """Compute the 3D model path from a KiCad lib_id string.

    Reuses the same mapping logic as :func:`_model_for_package`.
    """
    model = _model_for_package(lib_id)
    if model is not None:
        return model.path
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_pcb_file(
    pcb_path: str | Path,
    output_path: str | Path | None = None,
    flip_refs: tuple[str, ...] = (),
    add_3d_models: bool = True,
    model_var: str = KICAD_3DMODEL_VAR,
) -> None:
    """Enrich an existing ``.kicad_pcb`` file with 3D models and/or B.Cu flips.

    This is a non-destructive post-processor: placement, routing, zones,
    and all other PCB content are preserved exactly as-is.

    Args:
        pcb_path: Path to the input ``.kicad_pcb`` file.
        output_path: Path for the output file.  ``None`` overwrites in-place.
        flip_refs: Reference designators to move from F.Cu to B.Cu.
        add_3d_models: If ``True``, inject ``(model ...)`` for footprints
            that lack one.
        model_var: KiCad environment variable prefix for model paths.
    """
    pcb_path = Path(pcb_path)
    output_path = pcb_path if output_path is None else Path(output_path)

    _log.info("enrich_pcb_file: reading %s", pcb_path)
    tree = parse_file(str(pcb_path))

    flip_set = set(flip_refs)
    footprints = _find_footprint_nodes(tree)
    models_added = 0
    layers_flipped = 0

    for fp_node in footprints:
        ref = _extract_ref(fp_node)
        lib_id = _extract_lib_id(fp_node)

        # Flip to B.Cu if requested
        if ref in flip_set:
            _flip_footprint_layer(fp_node)
            layers_flipped += 1
            _log.info("enrich: flipped %s to B.Cu", ref)
            # Update lib_id for model lookup after flip
            lib_id = _extract_lib_id(fp_node)

        # Inject 3D model if missing
        if add_3d_models and not _has_model(fp_node):
            model_path = _model_path_from_lib_id(lib_id)
            if model_path is not None:
                # Replace default var with user-specified var if different
                if model_var != KICAD_3DMODEL_VAR:
                    model_path = model_path.replace(KICAD_3DMODEL_VAR, model_var)
                _inject_model(fp_node, model_path)
                models_added += 1

    _log.info(
        "enrich_pcb_file: %d models added, %d footprints flipped → %s",
        models_added, layers_flipped, output_path,
    )
    write_file(tree, str(output_path))


# ---------------------------------------------------------------------------
# DRC auto-fix helpers (S-expression tree mutations)
# ---------------------------------------------------------------------------


def _find_property_nodes(
    fp_node: list[SExpNode], prop_name: str
) -> list[list[SExpNode]]:
    """Find all ``(property <prop_name> ...)`` sublists within a footprint."""
    results: list[list[SExpNode]] = []
    for child in fp_node:
        if (
            isinstance(child, list)
            and len(child) >= 3
            and child[0] == "property"
            and child[1] == prop_name
        ):
            results.append(child)
    return results


def _find_at_node(node: list[SExpNode]) -> list[SExpNode] | None:
    """Find the ``(at x y [angle])`` sublist in a node."""
    for child in node:
        if isinstance(child, list) and len(child) >= 3 and child[0] == "at":
            return child
    return None


def _find_layer_node(node: list[SExpNode]) -> list[SExpNode] | None:
    """Find the ``(layer ...)`` sublist in a node."""
    for child in node:
        if isinstance(child, list) and len(child) >= 2 and child[0] == "layer":
            return child
    return None


def _get_silk_properties(
    fp_node: list[SExpNode],
) -> list[list[SExpNode]]:
    """Get all silkscreen-layer property nodes from a footprint."""
    silk_layers = {"F.SilkS", "B.SilkS", "F.Silkscreen", "B.Silkscreen"}
    silk_props: list[list[SExpNode]] = []

    for child in fp_node:
        if not isinstance(child, list) or len(child) < 3:
            continue
        if child[0] not in ("property", "fp_text"):
            continue
        # Check if this property has a silkscreen layer.
        layer_node = _find_layer_node(child)
        if layer_node and isinstance(layer_node[1], str) and layer_node[1] in silk_layers:
            silk_props.append(child)

    return silk_props


def fix_silk_overlap(pcb_path: str | Path, output_path: str | Path | None = None) -> int:
    """Move silkscreen text away from pads to fix silk overlap DRC violations.

    Shifts silk text outward from the footprint origin by a small offset.

    Args:
        pcb_path: Path to the ``.kicad_pcb`` file.
        output_path: Output path.  ``None`` overwrites in-place.

    Returns:
        Number of silk labels adjusted.
    """
    pcb_path = Path(pcb_path)
    output_path = pcb_path if output_path is None else Path(output_path)

    tree = parse_file(str(pcb_path))
    footprints = _find_footprint_nodes(tree)
    fixes = 0

    for fp_node in footprints:
        silk_props = _get_silk_properties(fp_node)
        fp_at = _find_at_node(fp_node)
        if not fp_at:
            continue

        for prop in silk_props:
            prop_at = _find_at_node(prop)
            if not prop_at:
                continue

            # Shift silk text 0.5mm upward (negative Y in KiCad coords).
            try:
                current_y = float(str(prop_at[2]))
                prop_at[2] = current_y - 0.5
                fixes += 1
            except (ValueError, IndexError):
                continue

    if fixes:
        _log.info("fix_silk_overlap: adjusted %d silk labels", fixes)
        write_file(tree, str(output_path))

    return fixes


def fix_courtyard_spacing(
    pcb_path: str | Path,
    output_path: str | Path | None = None,
) -> int:
    """Adjust courtyard outlines to fix courtyard overlap violations.

    Reduces courtyard size slightly for overlapping footprints.

    Args:
        pcb_path: Path to the ``.kicad_pcb`` file.
        output_path: Output path.  ``None`` overwrites in-place.

    Returns:
        Number of courtyard adjustments made.
    """
    pcb_path = Path(pcb_path)
    output_path = pcb_path if output_path is None else Path(output_path)

    tree = parse_file(str(pcb_path))
    footprints = _find_footprint_nodes(tree)
    fixes = 0

    courtyard_layers = {"F.CrtYd", "B.CrtYd", "F.Courtyard", "B.Courtyard"}

    for fp_node in footprints:
        for child in fp_node:
            if not isinstance(child, list) or len(child) < 2:
                continue
            if child[0] != "fp_rect" and child[0] != "fp_poly":
                continue

            # Check if this graphic is on a courtyard layer.
            layer_node = _find_layer_node(child)
            if not layer_node or not isinstance(layer_node[1], str):
                continue
            if layer_node[1] not in courtyard_layers:
                continue

            # Shrink courtyard by adjusting rect coordinates.
            if child[0] == "fp_rect":
                _shrink_rect(child, 0.05)
                fixes += 1

    if fixes:
        _log.info("fix_courtyard_spacing: adjusted %d courtyards", fixes)
        write_file(tree, str(output_path))

    return fixes


def _shrink_rect(rect_node: list[SExpNode], amount_mm: float) -> None:
    """Shrink an ``(fp_rect (start x y) (end x y) ...)`` by *amount_mm* on each side."""
    start_node: list[SExpNode] | None = None
    end_node: list[SExpNode] | None = None

    for child in rect_node:
        if isinstance(child, list) and len(child) >= 3:
            if child[0] == "start":
                start_node = child
            elif child[0] == "end":
                end_node = child

    if not start_node or not end_node:
        return

    try:
        sx, sy = float(str(start_node[1])), float(str(start_node[2]))
        ex, ey = float(str(end_node[1])), float(str(end_node[2]))

        # Shrink inward.
        if sx < ex:
            start_node[1] = sx + amount_mm
            end_node[1] = ex - amount_mm
        else:
            start_node[1] = sx - amount_mm
            end_node[1] = ex + amount_mm

        if sy < ey:
            start_node[2] = sy + amount_mm
            end_node[2] = ey - amount_mm
        else:
            start_node[2] = sy - amount_mm
            end_node[2] = ey + amount_mm
    except (ValueError, IndexError):
        pass

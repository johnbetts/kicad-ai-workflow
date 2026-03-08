"""Hierarchical schematic support for multi-sheet designs.

Splits a flat :class:`~kicad_pipeline.models.requirements.ProjectRequirements`
into per-feature sub-sheets connected by hierarchical labels/sheet pins.
Power nets use global labels (auto-visible across sheets); inter-feature
signal nets use hierarchical labels + sheet pins on the root.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    HIERARCHICAL_MIN_COMPONENTS,
    HIERARCHICAL_MIN_FEATURES,
    SHEET_SYMBOL_GRID_MARGIN_MM,
    SHEET_SYMBOL_MIN_HEIGHT_MM,
    SHEET_SYMBOL_MIN_WIDTH_MM,
    SHEET_SYMBOL_PIN_SPACING_MM,
)
from kicad_pipeline.models.requirements import (
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.models.schematic import (
    FontEffect,
    HierarchicalLabel,
    Point,
    Schematic,
    Sheet,
    SheetPin,
)

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import (
        Component,
        FeatureBlock,
        Net,
    )

log = logging.getLogger(__name__)

# Power net names that should use global labels instead of hierarchical labels.
# Imported at runtime to avoid circular imports.
_POWER_NET_NAMES: frozenset[str] = frozenset({
    "GND", "GNDD", "VBUS", "+5V", "+3V3", "+3.3V", "VCC", "+1V8",
})


def _new_uuid() -> str:
    """Generate a fresh UUID string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def should_use_hierarchy(requirements: ProjectRequirements) -> bool:
    """Decide whether hierarchical schematics should be used.

    Returns ``True`` when the project has more than one feature block AND
    the total component count exceeds the minimum threshold.

    Args:
        requirements: Project requirements to evaluate.

    Returns:
        ``True`` if hierarchical layout is recommended.
    """
    return (
        len(requirements.features) >= HIERARCHICAL_MIN_FEATURES
        and len(requirements.components) >= HIERARCHICAL_MIN_COMPONENTS
    )


# ---------------------------------------------------------------------------
# Net classification
# ---------------------------------------------------------------------------


def classify_nets(
    requirements: ProjectRequirements,
) -> tuple[frozenset[str], dict[str, frozenset[str]], frozenset[str]]:
    """Classify all nets into power, intra-feature, and inter-feature.

    Args:
        requirements: Project requirements with features and nets.

    Returns:
        A 3-tuple of:
        - **power_nets**: Net names that are power (GND, VCC, etc.)
        - **intra_nets**: Mapping from feature name to set of net names
          that are local to that feature.
        - **inter_feature_nets**: Net names whose connections span 2+ features.
    """
    # Build ref-to-feature mapping
    ref_to_feature: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            ref_to_feature[ref] = fb.name

    power_nets: set[str] = set()
    inter_feature_nets: set[str] = set()
    intra_nets: dict[str, set[str]] = {fb.name: set() for fb in requirements.features}

    for net in requirements.nets:
        if net.name in _POWER_NET_NAMES:
            power_nets.add(net.name)
            continue

        # Find which features this net touches
        features_touched: set[str] = set()
        for conn in net.connections:
            feat = ref_to_feature.get(conn.ref)
            if feat is not None:
                features_touched.add(feat)

        if len(features_touched) > 1:
            inter_feature_nets.add(net.name)
        elif len(features_touched) == 1:
            feat_name = next(iter(features_touched))
            intra_nets.setdefault(feat_name, set()).add(net.name)
        # Nets with 0 features touched (orphan refs) are treated as intra

    frozen_intra: dict[str, frozenset[str]] = {
        k: frozenset(v) for k, v in intra_nets.items()
    }
    return frozenset(power_nets), frozen_intra, frozenset(inter_feature_nets)


# ---------------------------------------------------------------------------
# Requirement partitioning
# ---------------------------------------------------------------------------


def partition_requirements(
    requirements: ProjectRequirements,
) -> dict[str, ProjectRequirements]:
    """Split a project into per-feature sub-requirements.

    Each sub-requirement contains only the components and nets belonging to
    one feature block. Inter-feature nets are included in both features so
    that hierarchical labels can be placed.

    Args:
        requirements: Full project requirements.

    Returns:
        Mapping from feature name to its sub-requirements.
    """
    ref_to_feature: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            ref_to_feature[ref] = fb.name

    _, _, inter_nets = classify_nets(requirements)

    result: dict[str, ProjectRequirements] = {}
    for fb in requirements.features:
        fb_refs = set(fb.components)

        # Components in this feature
        fb_components: list[Component] = [
            c for c in requirements.components if c.ref in fb_refs
        ]

        # Nets: include if any connection is in this feature
        fb_nets: list[Net] = []
        for net in requirements.nets:
            refs_in_net = {conn.ref for conn in net.connections}
            if refs_in_net & fb_refs:
                # For inter-feature nets, keep only connections in this feature
                if net.name in inter_nets:
                    filtered_conns = tuple(
                        c for c in net.connections if c.ref in fb_refs
                    )
                    if filtered_conns:
                        fb_nets.append(replace(net, connections=filtered_conns))
                else:
                    fb_nets.append(net)

        sub_req = ProjectRequirements(
            project=ProjectInfo(
                name=f"{requirements.project.name} - {fb.name}",
                author=requirements.project.author,
                revision=requirements.project.revision,
                description=fb.description,
            ),
            features=(fb,),
            components=tuple(fb_components),
            nets=tuple(fb_nets),
            pin_map=requirements.pin_map,
            power_budget=requirements.power_budget,
            mechanical=requirements.mechanical,
        )
        result[fb.name] = sub_req

    return result


# ---------------------------------------------------------------------------
# Sub-sheet builder
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Convert a feature name to a safe filename stem.

    Lowercases, replaces spaces/special chars with underscores.
    """
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _infer_pin_direction(
    net_name: str,
    feature_connections: int,
    total_connections: int,
) -> str:
    """Infer hierarchical label direction for an inter-feature net.

    Simple heuristic: if the feature has fewer connections than total,
    it is likely an output from the source feature.
    Falls back to 'bidirectional' for ambiguous cases.
    """
    name_lower = net_name.lower()
    if any(k in name_lower for k in ("clk", "sck", "mosi", "tx", "scl")):
        return "output"
    if any(k in name_lower for k in ("miso", "rx", "sda")):
        return "bidirectional"
    if feature_connections < total_connections:
        return "input"
    return "bidirectional"


def build_sub_sheet(
    feature: FeatureBlock,
    sub_req: ProjectRequirements,
    inter_nets: frozenset[str],
    power_nets: frozenset[str],
) -> Schematic:
    """Build a sub-sheet schematic for a single feature.

    Uses :func:`build_schematic` with ``compact=True`` for tight component
    placement, then adds hierarchical labels for inter-feature nets.
    Labels are vertically centred on the component bounding box.

    Args:
        feature: The feature block this sub-sheet represents.
        sub_req: Sub-requirements containing only this feature's components.
        inter_nets: Set of net names that cross feature boundaries.
        power_nets: Set of power net names (handled by global labels).

    Returns:
        A :class:`Schematic` with hierarchical labels added.
    """
    from kicad_pipeline.schematic.builder import build_schematic

    sch = build_schematic(sub_req, compact=True)

    # Collect which inter-feature nets need hierarchical labels
    fb_refs = set(feature.components)
    label_entries: list[tuple[str, str]] = []  # (net_name, direction)
    for net in sub_req.nets:
        if net.name not in inter_nets:
            continue
        if net.name in power_nets:
            continue
        feat_conns = sum(1 for c in net.connections if c.ref in fb_refs)
        total_conns = len(net.connections)
        direction = _infer_pin_direction(net.name, feat_conns, total_conns)
        label_entries.append((net.name, direction))

    if not label_entries:
        return sch

    # Compute component bounding box for vertical centering of labels
    all_positions = [inst.position for inst in sch.symbols]
    if all_positions:
        min_y = min(p.y for p in all_positions)
        max_y = max(p.y for p in all_positions)
    else:
        min_y = 25.0
        max_y = 50.0

    center_y = (min_y + max_y) / 2.0
    n_labels = len(label_entries)
    total_label_span = (n_labels - 1) * SHEET_SYMBOL_PIN_SPACING_MM
    label_start_y = center_y - total_label_span / 2.0
    # Ensure labels don't go above top margin
    label_start_y = max(label_start_y, 10.0)

    h_labels: list[HierarchicalLabel] = []
    for idx, (net_name, direction) in enumerate(label_entries):
        label_y = label_start_y + idx * SHEET_SYMBOL_PIN_SPACING_MM
        h_labels.append(HierarchicalLabel(
            text=net_name,
            shape=direction,
            position=Point(x=5.0, y=label_y),
            rotation=180.0,  # left-side label
            effects=FontEffect(),
            uuid=_new_uuid(),
        ))

    return replace(sch, hierarchical_labels=tuple(h_labels))


# ---------------------------------------------------------------------------
# Root sheet builder
# ---------------------------------------------------------------------------


def build_root_sheet(
    sub_schematics: dict[str, Schematic],
    inter_nets: frozenset[str],
    project_name: str,
    requirements: ProjectRequirements,
) -> Schematic:
    """Build the root schematic containing only sheet symbols.

    Args:
        sub_schematics: Mapping from feature name to its built sub-schematic.
        inter_nets: Inter-feature net names.
        project_name: Project name for the title block.
        requirements: Original requirements for metadata.

    Returns:
        Root :class:`Schematic` with :class:`Sheet` entries and no components.
    """
    import datetime

    from kicad_pipeline.constants import KICAD_GENERATOR, KICAD_SCH_VERSION

    sheets: list[Sheet] = []
    feature_names = list(sub_schematics.keys())

    # 2-column grid layout
    cols = 2
    col_width = SHEET_SYMBOL_MIN_WIDTH_MM + SHEET_SYMBOL_GRID_MARGIN_MM
    row_height_base = SHEET_SYMBOL_MIN_HEIGHT_MM + SHEET_SYMBOL_GRID_MARGIN_MM

    for idx, feat_name in enumerate(feature_names):
        sub_sch = sub_schematics[feat_name]
        col = idx % cols
        row = idx // cols

        # Determine sheet pin count from hierarchical labels
        h_labels = sub_sch.hierarchical_labels
        pin_count = len(h_labels)

        # Size the sheet symbol
        width = SHEET_SYMBOL_MIN_WIDTH_MM
        if h_labels:
            max_name_len = max(len(hl.text) for hl in h_labels)
            width = max(width, max_name_len * 1.5 + 10.0)

        height = max(
            SHEET_SYMBOL_MIN_HEIGHT_MM,
            pin_count * SHEET_SYMBOL_PIN_SPACING_MM + 5.0,
        )

        x = 20.0 + col * col_width
        y = 30.0 + row * row_height_base

        # Build sheet pins matching hierarchical labels
        pins: list[SheetPin] = []
        for pin_idx, hl in enumerate(h_labels):
            pin_y = y + 5.0 + pin_idx * SHEET_SYMBOL_PIN_SPACING_MM
            pins.append(SheetPin(
                name=hl.text,
                pin_type=hl.shape,
                position=Point(x=x, y=pin_y),
                rotation=180.0,  # pins on the left side of the sheet
                effects=FontEffect(),
                uuid=_new_uuid(),
            ))

        sheet_file = f"{_sanitize_filename(feat_name)}.kicad_sch"
        sheets.append(Sheet(
            position=Point(x=x, y=y),
            size_x=width,
            size_y=height,
            sheet_name=feat_name,
            sheet_file=sheet_file,
            pins=tuple(pins),
            uuid=_new_uuid(),
        ))

    # Determine page size
    total_rows = (len(feature_names) + cols - 1) // cols
    paper = "A3" if total_rows > 3 or len(feature_names) > 6 else "A4"

    return Schematic(
        lib_symbols=(),
        symbols=(),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
        sheets=tuple(sheets),
        hierarchical_labels=(),
        version=KICAD_SCH_VERSION,
        generator=KICAD_GENERATOR,
        paper=paper,
        title=requirements.project.name,
        date=datetime.date.today().isoformat(),
        revision=requirements.project.revision,
        company=requirements.project.author or "",
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def build_hierarchical_schematic(
    requirements: ProjectRequirements,
) -> dict[str, Schematic]:
    """Build a complete hierarchical schematic from requirements.

    Each feature block becomes a sub-sheet. The root schematic contains
    only sheet symbols with pins for inter-feature signal nets.

    Args:
        requirements: Full project requirements.

    Returns:
        Mapping from filename stem to :class:`Schematic`:
        - The root schematic key is the project name
        - Sub-sheet keys are sanitized feature names
    """
    log.info(
        "build_hierarchical_schematic: %d features, %d components",
        len(requirements.features),
        len(requirements.components),
    )

    power_nets, _intra_nets, inter_nets = classify_nets(requirements)
    sub_reqs = partition_requirements(requirements)

    # Build each sub-sheet
    sub_schematics: dict[str, Schematic] = {}
    for feat_name, sub_req in sub_reqs.items():
        feature = next(fb for fb in requirements.features if fb.name == feat_name)
        sub_sch = build_sub_sheet(feature, sub_req, inter_nets, power_nets)
        sub_schematics[feat_name] = sub_sch

    # Build root sheet
    project_name = _sanitize_filename(requirements.project.name)
    root_sch = build_root_sheet(sub_schematics, inter_nets, project_name, requirements)

    # Assemble result: root + sub-sheets
    result: dict[str, Schematic] = {}
    result[project_name] = root_sch
    for feat_name, sub_sch in sub_schematics.items():
        result[_sanitize_filename(feat_name)] = sub_sch

    log.info(
        "build_hierarchical_schematic: %d sheets (%d sub-sheets + root)",
        len(result),
        len(sub_schematics),
    )

    return result

"""Constraint resolver: translate Component placement fields and subcircuit topology
into a unified PlacementConstraintSet.

This module bridges two sources of placement knowledge:

1. **Explicit**: ``placement_group``, ``placement_near``, ``placement_order``, and
   ``placement_near_max_mm`` fields on :class:`~kicad_pipeline.models.requirements.Component`.
2. **Inferred**: :class:`~kicad_pipeline.optimization.functional_grouper.DetectedSubCircuit`
   topology (layout hints, component roles, power-flow ordering).

Explicit constraints always win over inferred ones.  The resolver never raises
exceptions — validation issues are logged as warnings and the best-effort
:class:`~kicad_pipeline.models.pcb.PlacementConstraintSet` is returned.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import (
    OrderingChain,
    PlacementConstraintSet,
    ProximityConstraint,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kicad_pipeline.models.requirements import Component, ProjectRequirements
    from kicad_pipeline.optimization.functional_grouper import DetectedSubCircuit

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_DISTANCE_MM = 5.0
_BUCK_ROLE_ORDER = ("input", "input_cap", "ic", "inductor", "output_cap", "output")
_DIVIDER_ROLE_ORDER = ("top", "bottom")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_near_target(raw: str) -> tuple[str, str | None]:
    """Parse a ``placement_near`` value into ``(target_ref, target_pin | None)``.

    Accepts either ``"U1:VIN"`` (returns ``("U1", "VIN")``) or
    ``"U1"`` (returns ``("U1", None)``).
    """
    if ":" in raw:
        ref_part, pin_part = raw.split(":", 1)
        return ref_part.strip(), pin_part.strip() or None
    return raw.strip(), None


def _deduplicated_components(requirements: ProjectRequirements) -> list[Component]:
    """Return all components from top-level requirements, deduplicated by ref."""
    seen: set[str] = set()
    result: list[Component] = []
    for comp in requirements.components:
        if comp.ref not in seen:
            seen.add(comp.ref)
            result.append(comp)
    return result


# ---------------------------------------------------------------------------
# Subcircuit inference helpers
# ---------------------------------------------------------------------------


def _infer_ordering_from_subcircuit(
    sc: DetectedSubCircuit,
    explicit_refs_in_group: set[str],
) -> OrderingChain | None:
    """Infer an :class:`OrderingChain` from a subcircuit's topology.

    Returns ``None`` if no meaningful ordering can be derived or all refs are
    already explicitly ordered.
    """
    from kicad_pipeline.optimization.functional_grouper import SubCircuitType

    refs_to_order: list[str] = [r for r in sc.refs if r not in explicit_refs_in_group]
    if not refs_to_order:
        return None

    group_name = f"{sc.circuit_type.value}_{sc.anchor_ref}"

    if sc.circuit_type == SubCircuitType.BUCK_CONVERTER and sc.hierarchy is not None:
        return _order_by_role(refs_to_order, sc, group_name, _BUCK_ROLE_ORDER)

    if sc.circuit_type == SubCircuitType.VOLTAGE_DIVIDER and sc.hierarchy is not None:
        return _order_by_role(refs_to_order, sc, group_name, _DIVIDER_ROLE_ORDER)

    if sc.layout_hint == "linear":
        ordered_linear = [r for r in sc.refs if r in set(refs_to_order)]
        if ordered_linear:
            return OrderingChain(group=group_name, refs=tuple(ordered_linear))

    return None


def _order_by_role(
    refs_to_order: list[str],
    sc: DetectedSubCircuit,
    group_name: str,
    role_sequence: tuple[str, ...],
) -> OrderingChain:
    """Sort *refs_to_order* by their role in the subcircuit hierarchy."""
    role_index: dict[str, int] = {r: i for i, r in enumerate(role_sequence)}
    role_map: dict[str, str] = {}

    if sc.hierarchy is not None:
        node = sc.hierarchy
        role_map[node.anchor_ref] = node.role
        for child in node.children:
            for r in child.refs:
                role_map[r] = child.role
            role_map[child.anchor_ref] = child.role

    def _key(ref: str) -> int:
        return role_index.get(role_map.get(ref, ""), 999)

    ordered = sorted(refs_to_order, key=_key)
    return OrderingChain(group=group_name, refs=tuple(ordered))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_constraints(
    requirements: ProjectRequirements,
    subcircuits: Sequence[DetectedSubCircuit] | None = None,
) -> PlacementConstraintSet:
    """Resolve placement constraints into a unified :class:`PlacementConstraintSet`.

    Scans every :class:`~kicad_pipeline.models.requirements.Component` for explicit
    ``placement_*`` fields, then overlays inferred ordering from ``subcircuits``.
    Explicit fields always win.

    Args:
        requirements: Fully parsed project requirements.
        subcircuits: Optional list of sub-circuits detected by
            :func:`~kicad_pipeline.optimization.functional_grouper.detect_subcircuits`.

    Returns:
        A resolved :class:`PlacementConstraintSet`.  Never raises.
    """
    components = _deduplicated_components(requirements)
    all_refs: set[str] = {c.ref for c in components}

    # -- 1. Groups -----------------------------------------------------------
    # group_name → ordered list of (placement_order, ref)
    group_buckets: dict[str, list[tuple[int, str]]] = defaultdict(list)
    explicit_group_refs: set[str] = set()
    ref_to_groups: dict[str, list[str]] = defaultdict(list)

    for comp in components:
        if comp.placement_group is None:
            continue
        order = comp.placement_order if comp.placement_order is not None else 0
        group_buckets[comp.placement_group].append((order, comp.ref))
        explicit_group_refs.add(comp.ref)
        ref_to_groups[comp.ref].append(comp.placement_group)

    for ref, grps in ref_to_groups.items():
        if len(grps) > 1:
            _log.warning(
                "Component %s is assigned to multiple placement groups: %s — "
                "it will appear in all of them",
                ref,
                grps,
            )

    groups_list: list[tuple[str, tuple[str, ...]]] = []
    for group_name, entries in sorted(group_buckets.items()):
        sorted_entries = sorted(entries, key=lambda t: t[0])
        refs_in_order = tuple(ref for _, ref in sorted_entries)
        groups_list.append((group_name, refs_in_order))

    # -- 2. Proximity constraints -------------------------------------------
    proximity_list: list[ProximityConstraint] = []

    for comp in components:
        if comp.placement_near is None:
            continue
        target_ref, target_pin = _parse_near_target(comp.placement_near)
        if target_ref not in all_refs:
            _log.warning(
                "Component %s has placement_near=%r but target ref %r does not exist",
                comp.ref,
                comp.placement_near,
                target_ref,
            )
        max_dist = (
            comp.placement_near_max_mm
            if comp.placement_near_max_mm is not None
            else _DEFAULT_MAX_DISTANCE_MM
        )
        proximity_list.append(
            ProximityConstraint(
                ref=comp.ref,
                target_ref=target_ref,
                target_pin=target_pin,
                max_distance_mm=max_dist,
            )
        )

    # -- 3. Explicit ordering chains ----------------------------------------
    ordering_explicit: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for comp in components:
        if comp.placement_group is None or comp.placement_order is None:
            continue
        ordering_explicit[comp.placement_group].append((comp.placement_order, comp.ref))

    ordering_list: list[OrderingChain] = []
    explicit_ordering_groups: set[str] = set()

    for group_name, entries in sorted(ordering_explicit.items()):
        if len(entries) < 2:
            continue  # need at least two refs to form a chain
        sorted_entries = sorted(entries, key=lambda t: t[0])
        refs_in_order = tuple(ref for _, ref in sorted_entries)
        ordering_list.append(OrderingChain(group=group_name, refs=refs_in_order))
        explicit_ordering_groups.add(group_name)

    # -- 4. Inferred ordering from subcircuits ------------------------------
    if subcircuits:
        for sc in subcircuits:
            inferred = _infer_ordering_from_subcircuit(sc, explicit_group_refs)
            if inferred is None:
                continue
            if inferred.group in explicit_ordering_groups:
                _log.debug(
                    "Skipping inferred ordering for %s — explicit ordering present",
                    inferred.group,
                )
                continue
            ordering_list.append(inferred)

    return PlacementConstraintSet(
        groups=tuple(groups_list),
        proximity=tuple(proximity_list),
        ordering=tuple(ordering_list),
        trace_length_match=(),
    )

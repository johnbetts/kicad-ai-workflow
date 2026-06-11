"""Gate A sign-off verifier — re-derives every constraint from the artifact.

The verifier consumes the SAME :class:`~kicad_pipeline.placement_v2.ir.ConstraintSet`
objects the solvers used and re-checks them against the written board —
ideally the ``.kicad_pcb`` file re-parsed from disk
(:func:`verify_board_file`, which also catches writer bugs), falling back
to an in-memory :class:`~kicad_pipeline.models.pcb.PCBDesign`
(:func:`verify_board`). It never trusts solver state.

File-parsing subset
-------------------
No full ``.kicad_pcb`` → ``PCBDesign`` loader exists in the pipeline, so
this module implements the minimal subset Gate A needs on top of
:func:`kicad_pipeline.sexp.parser.parse_file`:

* footprints: ref, position, rotation, layer, pads (number/type/shape/
  position/size/layers/drill)
* footprint courtyard graphics: ``fp_line`` / ``fp_rect`` on ``*.CrtYd``
  layers — REQUIRED so THT bodies larger than their pad field (relays,
  connectors) are checked at full size instead of degrading to the
  pad-bbox fallback (a Gate B vision finding converted to a
  deterministic rule, per the standing rule in architecture section 7)
* board outline: ``gr_line`` / ``gr_rect`` segments on ``Edge.Cuts``,
  chained into an ordered polygon
* keepout zones: ``(zone ... (keepout ...))`` polygons and layers

Nets, tracks, vias, copper zones, non-courtyard graphics, texts, and 3D
models are NOT parsed — no Gate A check needs them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import ValidationError
from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    FootprintLine,
    Keepout,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.placement_v2.ir import Severity
from kicad_pipeline.placement_v2.verifier_rules import (
    check_attach_bundles,
    check_contain,
    check_courtyards,
    check_edge_pins,
    check_isolation,
    check_keepouts,
    check_pin_attach,
    check_sequences,
)
from kicad_pipeline.sexp.parser import parse_file

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.placement_v2.ir import ConstraintSet, Violation
    from kicad_pipeline.sexp.writer import SExpNode

logger = logging.getLogger(__name__)

_EDGE_CUTS = "Edge.Cuts"
_CHAIN_PRECISION = 3  # mm decimals used to match outline segment endpoints

Segment = tuple[tuple[float, float], tuple[float, float]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_board(
    pcb: PCBDesign,
    constraints: ConstraintSet,
    *,
    domains: tuple[tuple[str, str], ...] = (),
) -> tuple[Violation, ...]:
    """Re-derive every constraint in *constraints* from *pcb*.

    Args:
        pcb: The board artifact (re-parsed from disk where possible).
        constraints: The SAME constraint set the solvers consumed.
        domains: ``(ref, domain)`` pairs assigning footprints to voltage
            domains, used by the isolation check.

    Returns:
        All violations found, in check order. Empty means Gate A passes.
    """
    violations: list[Violation] = []
    violations.extend(check_pin_attach(pcb, constraints.pin_attach))
    violations.extend(check_sequences(pcb, constraints.sequences))
    violations.extend(check_edge_pins(pcb, constraints.edge_pins))
    violations.extend(check_contain(pcb, constraints.contain))
    violations.extend(check_courtyards(pcb))
    violations.extend(check_keepouts(pcb, constraints.keepouts))
    violations.extend(check_isolation(pcb, constraints.isolation, domains))
    violations.extend(check_attach_bundles(pcb, constraints.bundles))
    return tuple(violations)


def verify_board_file(
    path: Path,
    constraints: ConstraintSet,
    *,
    domains: tuple[tuple[str, str], ...] = (),
) -> tuple[Violation, ...]:
    """Re-parse the ``.kicad_pcb`` at *path* from disk, then verify it."""
    return verify_board(load_pcb_subset(path), constraints, domains=domains)


@dataclass(frozen=True)
class GateAReport:
    """Result of a Gate A run: every check executed and every violation.

    ``checks_run`` names each check with the number of constraint
    instances it covered (e.g. ``"pin_attach x12"``), including
    explicitly skipped checks, so coverage is honest, not just a verdict.
    """

    violations: tuple[Violation, ...]
    checks_run: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True when no CRITICAL or MAJOR violation blocks sign-off."""
        blocking = (Severity.CRITICAL, Severity.MAJOR)
        return all(v.severity not in blocking for v in self.violations)


def run_gate_a(
    path: Path,
    constraints: ConstraintSet,
    *,
    domains: tuple[tuple[str, str], ...] = (),
) -> GateAReport:
    """Run Gate A against the written board file and report coverage."""
    pcb = load_pcb_subset(path)
    violations = verify_board(pcb, constraints, domains=domains)
    report = GateAReport(
        violations=violations,
        checks_run=_checks_summary(pcb, constraints, domains),
    )
    logger.info(
        "Gate A on %s: %d checks, %d violations, passed=%s",
        path, len(report.checks_run), len(violations), report.passed,
    )
    return report


def _checks_summary(
    pcb: PCBDesign,
    constraints: ConstraintSet,
    domains: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """Name every check executed, with instance counts."""
    same_layer_pairs = 0
    fps = [fp for fp in pcb.footprints if fp.pads]
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if fps[i].layer == fps[j].layer:
                same_layer_pairs += 1
    domain_of = dict(domains)
    isolation_pairs = 0
    for gap in constraints.isolation:
        n_a = sum(1 for d in domain_of.values() if d == gap.domain_a)
        n_b = sum(1 for d in domain_of.values() if d == gap.domain_b)
        isolation_pairs += n_a * n_b
    return (
        f"pin_attach x{len(constraints.pin_attach)}",
        f"sequence x{len(constraints.sequences)}",
        f"edge_pin_distance x{len(constraints.edge_pins)}",
        f"edge_pin_face_out x{len(constraints.edge_pins)}",
        f"contain_pad x{len(pcb.footprints)}",
        f"contain_courtyard x{len(pcb.footprints)}",
        f"courtyard_pair x{same_layer_pairs}",
        f"keepout x{len(constraints.keepouts)}",
        f"isolation_pair x{isolation_pairs}",
        f"attach_bundle x{len(constraints.bundles)}",
    )


# ---------------------------------------------------------------------------
# Minimal .kicad_pcb subset loader
# ---------------------------------------------------------------------------


def load_pcb_subset(path: Path) -> PCBDesign:
    """Parse the Gate A subset of a ``.kicad_pcb`` file into a PCBDesign.

    Raises:
        ValidationError: If the file is not a kicad_pcb document or has
            no Edge.Cuts outline.
    """
    root = parse_file(path)
    if not isinstance(root, list) or not root or root[0] != "kicad_pcb":
        raise ValidationError(f"{path}: not a kicad_pcb document")
    footprints: list[Footprint] = []
    segments: list[Segment] = []
    keepouts: list[Keepout] = []
    for node in root:
        if not isinstance(node, list) or not node:
            continue
        tag = node[0]
        if tag == "footprint":
            footprints.append(_parse_footprint(node))
        elif tag in ("gr_line", "gr_rect") and _str_child(node, "layer") == _EDGE_CUTS:
            segments.extend(_parse_edge_segments(node))
        elif tag == "zone":
            keepout = _parse_keepout_zone(node)
            if keepout is not None:
                keepouts.append(keepout)
    if not segments:
        raise ValidationError(f"{path}: no Edge.Cuts outline found")
    return PCBDesign(
        outline=BoardOutline(polygon=_chain_outline(segments)),
        design_rules=DesignRules(),
        nets=(),
        footprints=tuple(footprints),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=tuple(keepouts),
    )


def _num(value: SExpNode) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"expected a number, got boolean {value!r}")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValidationError(f"expected a number, got {value!r}")


def _child(node: list[SExpNode], tag: str) -> list[SExpNode] | None:
    for item in node:
        if isinstance(item, list) and item and item[0] == tag:
            return item
    return None


def _str_child(node: list[SExpNode], tag: str) -> str | None:
    child = _child(node, tag)
    if child is None or len(child) < 2:
        return None
    return str(child[1])


def _parse_edge_segments(node: list[SExpNode]) -> tuple[Segment, ...]:
    start = _child(node, "start")
    end = _child(node, "end")
    if start is None or end is None or len(start) < 3 or len(end) < 3:
        return ()
    x1, y1 = _num(start[1]), _num(start[2])
    x2, y2 = _num(end[1]), _num(end[2])
    if node[0] == "gr_rect":
        return (
            (((x1, y1), (x2, y1))),
            (((x2, y1), (x2, y2))),
            (((x2, y2), (x1, y2))),
            (((x1, y2), (x1, y1))),
        )
    return ((((x1, y1), (x2, y2))),)


def _chain_outline(segments: list[Segment]) -> tuple[Point, ...]:
    """Order Edge.Cuts segments into a polygon by chaining endpoints."""
    def key(p: tuple[float, float]) -> tuple[float, float]:
        return (round(p[0], _CHAIN_PRECISION), round(p[1], _CHAIN_PRECISION))

    remaining = list(segments)
    first = remaining.pop(0)
    points: list[tuple[float, float]] = [first[0], first[1]]
    while remaining:
        cursor = key(points[-1])
        for i, (a, b) in enumerate(remaining):
            if key(a) == cursor:
                points.append(b)
                remaining.pop(i)
                break
            if key(b) == cursor:
                points.append(a)
                remaining.pop(i)
                break
        else:
            logger.warning(
                "Edge.Cuts outline is not a single closed chain; "
                "%d segments left unchained", len(remaining),
            )
            for a, b in remaining:
                points.extend((a, b))
            break
    if len(points) > 2 and key(points[0]) == key(points[-1]):
        points.pop()
    return tuple(Point(x, y) for x, y in points)


def _parse_pad(node: list[SExpNode]) -> Pad | None:
    if len(node) < 4:
        return None
    at = _child(node, "at")
    size = _child(node, "size")
    if at is None or size is None or len(at) < 3 or len(size) < 3:
        return None
    layers_node = _child(node, "layers")
    layers = tuple(str(layer) for layer in layers_node[1:]) if layers_node else ()
    drill = _child(node, "drill")
    return Pad(
        number=str(node[1]),
        pad_type=str(node[2]),
        shape=str(node[3]),
        position=Point(_num(at[1]), _num(at[2])),
        size_x=_num(size[1]),
        size_y=_num(size[2]),
        layers=layers,
        drill_diameter=_num(drill[1]) if drill is not None and len(drill) > 1 else None,
    )


def _parse_ref(node: list[SExpNode]) -> str:
    """KiCad 9 ``(property "Reference" "R1")`` or legacy ``fp_text reference``."""
    for child in node:
        if not isinstance(child, list) or len(child) < 3:
            continue
        if child[0] == "property" and child[1] == "Reference":
            return str(child[2])
        if child[0] == "fp_text" and child[1] == "reference":
            return str(child[2])
    return ""


def _parse_footprint(node: list[SExpNode]) -> Footprint:
    lib_id = str(node[1]) if len(node) > 1 else ""
    at = _child(node, "at")
    if at is None or len(at) < 3:
        raise ValidationError(f"footprint {lib_id!r} has no (at ...) position")
    pads = [
        pad
        for child in node
        if isinstance(child, list) and child and child[0] == "pad"
        and (pad := _parse_pad(child)) is not None
    ]
    return Footprint(
        lib_id=lib_id,
        ref=_parse_ref(node),
        value="",
        position=Point(_num(at[1]), _num(at[2])),
        rotation=_num(at[3]) if len(at) > 3 else 0.0,
        layer=_str_child(node, "layer") or "F.Cu",
        pads=tuple(pads),
        graphics=tuple(_parse_courtyard_lines(node)),
    )


def _parse_courtyard_lines(node: list[SExpNode]) -> list[FootprintLine]:
    """Courtyard graphics (``fp_line``/``fp_rect`` on ``*.CrtYd``).

    Without these, ``courtyard_polygon`` degrades to the pad bbox and
    Gate A under-checks THT bodies (relay bodies overlap passives while
    the check passes) — found by a Gate B vision review and converted
    to this deterministic parse per the standing rule.
    """
    lines: list[FootprintLine] = []
    for child in node:
        if not (isinstance(child, list) and child
                and child[0] in ("fp_line", "fp_rect")):
            continue
        layer = _str_child(child, "layer") or ""
        if not layer.endswith(".CrtYd"):
            continue
        start = _child(child, "start")
        end = _child(child, "end")
        if start is None or end is None or len(start) < 3 or len(end) < 3:
            continue
        lines.append(FootprintLine(
            start=Point(_num(start[1]), _num(start[2])),
            end=Point(_num(end[1]), _num(end[2])),
            layer=layer,
        ))
    return lines


def _parse_keepout_zone(node: list[SExpNode]) -> Keepout | None:
    rules = _child(node, "keepout")
    if rules is None:
        return None
    polygon_node = _child(node, "polygon")
    pts_node = _child(polygon_node, "pts") if polygon_node is not None else None
    if pts_node is None:
        return None
    points = tuple(
        Point(_num(xy[1]), _num(xy[2]))
        for xy in pts_node
        if isinstance(xy, list) and len(xy) >= 3 and xy[0] == "xy"
    )
    layers_node = _child(node, "layers")
    layers = tuple(str(layer) for layer in layers_node[1:]) if layers_node else ()
    return Keepout(
        polygon=points,
        layers=layers,
        no_copper=_str_child(rules, "copperpour") == "not_allowed",
        no_tracks=_str_child(rules, "tracks") == "not_allowed",
        no_vias=_str_child(rules, "vias") == "not_allowed",
    )

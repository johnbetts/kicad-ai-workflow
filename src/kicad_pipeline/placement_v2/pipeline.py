"""The v2 placement pipeline: certify -> cells -> floorplan -> ledger.

Pure-function chain over frozen artifacts. Every stage appends a
:class:`~kicad_pipeline.placement_v2.ledger.StageRecord`; a failing
stage HALTS the build with quantified violations — there are no fixup
phases and no partially-correct output.

Coordinate conventions at the boundary:

* v2 cells work in CENTROID space using the KICAD rotation convention
  natively (positive = screen-CCW on the Y-down board) — solver-frame
  geometry is identical to the artifact, no conversion seam.
* :func:`emit_layout` only converts centroid -> ORIGIN positions via
  the blessed ``pin_map`` helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kicad_pipeline.models.pcb import Point
from kicad_pipeline.pcb.pin_map import centroid_to_origin
from kicad_pipeline.placement_v2.arrays import instantiate_array
from kicad_pipeline.placement_v2.cells import (
    Cell,
    CellProof,
    PlacedCell,
    PlacedMember,
    Port,
)
from kicad_pipeline.placement_v2.certify import (
    CertificateStore,
    build_certificate,
    certify_board_footprints,
    compute_footprint_sha256,
)
from kicad_pipeline.placement_v2.compile import compile_constraints
from kicad_pipeline.placement_v2.floorplan import Floorplan, pack_board, pack_group
from kicad_pipeline.placement_v2.footprint_geom import (
    courtyard_polygon,
    pad_position_in_frame,
)
from kicad_pipeline.placement_v2.generators import CellGenerationError, generate_cell
from kicad_pipeline.placement_v2.ir import PadRef, Severity, Violation
from kicad_pipeline.placement_v2.ledger import (
    BuildLedger,
    StageRecord,
    sha256_text,
)
from kicad_pipeline.placement_v2.legalize import LegalizationError, legalize

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from kicad_pipeline.models.pcb import Footprint
    from kicad_pipeline.models.requirements import ProjectRequirements
    from kicad_pipeline.optimization.functional_grouper import DetectedSubCircuit
    from kicad_pipeline.placement_v2.ir import ConstraintSet

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlacementV2Result:
    """Final placement in KiCad conventions + the evidence trail."""

    positions: tuple[tuple[str, float, float], ...]  # ref, origin x, y
    rotations: tuple[tuple[str, float], ...]  # ref, KiCad rotation deg
    board_width: float
    board_height: float
    floorplan: Floorplan
    constraints: ConstraintSet
    violations: tuple[Violation, ...] = ()
    halted_stage: str | None = None

    @property
    def ok(self) -> bool:
        """True when no stage halted the build."""
        return self.halted_stage is None

    def positions_dict(self) -> dict[str, Point]:
        """Builder-compatible ref -> origin Point mapping."""
        return {ref: Point(x, y) for ref, x, y in self.positions}

    def rotations_dict(self) -> dict[str, float]:
        """Builder-compatible ref -> KiCad rotation mapping."""
        return dict(self.rotations)


@dataclass
class _StageLog:
    """Internal collector that mirrors every record into the ledger."""

    ledger: BuildLedger | None
    timestamp: str
    records: list[StageRecord] = field(default_factory=list)

    def add(
        self,
        stage: str,
        passed: bool,
        checks: tuple[str, ...],
        violations: tuple[Violation, ...],
        in_hash: str,
        out_hash: str,
        detail: str = "",
    ) -> None:
        rec = StageRecord(
            stage=stage,
            input_sha256=in_hash,
            output_sha256=out_hash,
            checks_run=checks,
            violations=violations,
            passed=passed,
            detail=detail,
            timestamp=self.timestamp,
        )
        self.records.append(rec)
        if self.ledger is not None:
            self.ledger.append(rec)


def _singleton_cell(ref: str, fp: Footprint, nets: Mapping[str, tuple[PadRef, ...]]) -> Cell:
    """A one-component cell for parts outside any detected subcircuit."""
    ports = []
    for net in sorted(nets):
        for pr in nets[net]:
            if pr.ref != ref:
                continue
            try:
                x, y = pad_position_in_frame(fp, pr.pin, 0.0, 0.0, 0.0)
            except KeyError:
                continue
            ports.append(Port(net=net, x=x, y=y))
            break
    return Cell(
        name=f"single:{ref}",
        kind="singleton",
        members=(PlacedMember(ref=ref, x=0.0, y=0.0, rotation_deg=0.0),),
        polygon=courtyard_polygon(fp),
        ports=tuple(ports),
        proof=CellProof(checks=("singleton",)),
    )


def _net_pad_map(
    requirements: ProjectRequirements,
) -> dict[str, tuple[PadRef, ...]]:
    return {
        net.name: tuple(PadRef(c.ref, c.pin) for c in net.connections)
        for net in requirements.nets
    }


def _external_nets_for(
    member_refs: frozenset[str],
    net_pads: Mapping[str, tuple[PadRef, ...]],
) -> dict[str, tuple[PadRef, ...]]:
    """Nets that cross the cell boundary -> internal pads carrying them."""
    out: dict[str, tuple[PadRef, ...]] = {}
    for net, pads in net_pads.items():
        inside = tuple(p for p in pads if p.ref in member_refs)
        if inside and len(inside) < len(pads):
            out[net] = inside
    return out


def _plan_cell_members(
    detected: tuple[DetectedSubCircuit, ...],
    constraints: ConstraintSet,
    all_refs: set[str],
    frozen_refs: frozenset[str] = frozenset(),
) -> dict[str, list[str]]:
    """Final member list per anchor: detection + merges + absorption.

    Three steps, all driven by the attachment graph (a pin-attach bound
    is only enforceable INSIDE one rigid cell):

    1. Mirror the claiming the generation loop will do (first sub wins
       a ref), counting only subs that actually become cells — seeding
       from skipped subs left strays looking "claimed".
    2. MERGE small linked subcircuits: an attachment crossing two cells
       (voltage divider <-> ADC clamp/connector) folds the smaller cell
       into the larger one's member list.
    3. ABSORB unclaimed strays in BOTH directions: a stray src joins
       its dst's cell (pull-up -> MCU) and a stray dst joins its src's
       cell (button <- pull-up), to a fixpoint.
    """
    members: dict[str, list[str]] = {}
    ref_to_anchor: dict[str, str] = {}
    for sub in detected:
        refs = [r for r in sub.refs if r in all_refs and r not in ref_to_anchor]
        if sub.anchor_ref not in refs or len(refs) < 2:
            continue
        members[sub.anchor_ref] = refs
        for r in refs:
            ref_to_anchor[r] = sub.anchor_ref

    def _merge(into: str, victim: str) -> None:
        for r in members.pop(victim):
            ref_to_anchor[r] = into
            members[into].append(r)

    for _ in range(len(constraints.pin_attach) + 1):
        progressed = False
        for a in sorted(
            constraints.pin_attach, key=lambda a: (a.src.ref, a.dst.ref)
        ):
            src_a = ref_to_anchor.get(a.src.ref)
            dst_a = ref_to_anchor.get(a.dst.ref)
            if src_a is not None and dst_a is not None:
                # Merge only true FRAGMENTS (a 2-resistor divider) into
                # their partner, and never grow past 8 members — a
                # looser rule cascaded whole power chains into one
                # unpackable mega-cell.
                if src_a != dst_a and (
                    min(len(members[src_a]), len(members[dst_a])) <= 2
                    and len(members[src_a]) + len(members[dst_a]) <= 8
                ):
                    small, big = sorted(
                        (src_a, dst_a), key=lambda x: (len(members[x]), x)
                    )
                    _merge(big, small)
                    progressed = True
                continue
            # Edge-pinned connectors are never absorbed: pulling a J
            # into a cell bloats the group and forfeits the
            # connector's freedom to find its own board edge.
            if (src_a is None and dst_a is not None
                    and a.src.ref in all_refs
                    and a.src.ref not in frozen_refs):
                ref_to_anchor[a.src.ref] = dst_a
                members[dst_a].append(a.src.ref)
                progressed = True
            elif (dst_a is None and src_a is not None
                    and a.dst.ref in all_refs
                    and a.dst.ref not in frozen_refs):
                ref_to_anchor[a.dst.ref] = src_a
                members[src_a].append(a.dst.ref)
                progressed = True
        if not progressed:
            break
    return members


def _obstacle_cell(name: str, polygon: tuple[Point, ...]) -> PlacedCell:
    """An immovable reserved region (mounting-hole corner) as a cell.

    The pseudo member's ref never appears in the footprint map, so it
    is skipped at emission; its only job is to keep real cells out.
    """
    cx = sum(p.x for p in polygon) / len(polygon)
    cy = sum(p.y for p in polygon) / len(polygon)
    local = tuple(Point(p.x - cx, p.y - cy) for p in polygon)
    cell = Cell(
        name=f"reserved:{name}",
        kind="reserved",
        members=(PlacedMember(ref=f"__{name}", x=0.0, y=0.0, rotation_deg=0.0),),
        polygon=local,
        ports=(),
        proof=CellProof(checks=("reserved",)),
    )
    return PlacedCell(cell, cx, cy, 0)


def _attachment_islands(
    constraints: ConstraintSet, unclaimed: set[str],
) -> list[list[str]]:
    """Connected components (size >= 2) of the attachment graph over
    refs no detected subcircuit claimed."""
    adj: dict[str, set[str]] = {}
    for a in constraints.pin_attach:
        if a.src.ref in unclaimed and a.dst.ref in unclaimed:
            adj.setdefault(a.src.ref, set()).add(a.dst.ref)
            adj.setdefault(a.dst.ref, set()).add(a.src.ref)
    seen: set[str] = set()
    islands: list[list[str]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack, comp = [start], []
        while stack:
            r = stack.pop()
            if r in seen:
                continue
            seen.add(r)
            comp.append(r)
            stack.extend(adj.get(r, ()))
        if len(comp) >= 2:
            islands.append(sorted(comp))
    return islands


def _group_for(
    refs: frozenset[str], requirements: ProjectRequirements,
) -> str:
    """FeatureBlock owning the majority of a cell's refs (else 'misc')."""
    best, best_count = "misc", 0
    for feature in requirements.features:
        count = len(refs & set(feature.components))
        if count > best_count:
            best, best_count = feature.name, count
    return best


def run_placement_v2(
    requirements: ProjectRequirements,
    footprints: Mapping[str, Footprint],
    *,
    board_width_mm: float | None = None,
    board_height_mm: float | None = None,
    part_rules_path: Path | None = None,
    certificate_store_path: Path | None = None,
    bootstrap_certificates: bool = True,
    ledger_path: Path | None = None,
    timestamp: str = "",
    reserved_zones: tuple[tuple[str, tuple[Point, ...]], ...] = (),
) -> PlacementV2Result:
    """Run the full v2 pipeline. Halts (with ledger evidence) on failure.

    *bootstrap_certificates* certifies previously-unseen footprints on
    first use (recording their hashes so any later drift is detected);
    with it off, every part must already be certified or the build
    halts at Stage 0.
    """
    ledger = BuildLedger(ledger_path) if ledger_path is not None else None
    log_ = _StageLog(ledger=ledger, timestamp=timestamp)
    req_hash = sha256_text(repr(requirements))

    def _halt(
        stage: str,
        violations: tuple[Violation, ...],
        floorplan: Floorplan | None = None,
    ) -> PlacementV2Result:
        return PlacementV2Result(
            positions=(), rotations=(),
            board_width=0.0, board_height=0.0,
            # The failed floorplan (when one exists) rides along for
            # diagnosis — halted results are never emitted as boards.
            floorplan=floorplan or Floorplan((), 0.0, 0.0),
            constraints=constraints,
            violations=violations,
            halted_stage=stage,
        )

    # ---- Compile constraint IR -------------------------------------------
    constraints = compile_constraints(
        requirements, part_rules_path=part_rules_path,
    )
    _log.info("v2: compiled %d constraints", constraints.count())

    # ---- Stage 0: certification gate -------------------------------------
    store = (
        CertificateStore.load(certificate_store_path)
        if certificate_store_path is not None
        else CertificateStore()
    )
    keys = {
        ref: (fp.lcsc or f"parametric:{fp.lib_id}")
        for ref, fp in footprints.items()
    }
    if bootstrap_certificates:
        added = 0
        for ref, fp in sorted(footprints.items()):
            key = keys[ref]
            if key not in store:
                store = store.add(build_certificate(
                    fp, key=key, certified_at=timestamp or "bootstrap",
                    checks_passed=("bootstrap",),
                ))
                added += 1
        if added and certificate_store_path is not None:
            store.save(certificate_store_path)
        if added:
            _log.info("v2: bootstrapped %d certificates", added)
    cert_violations = certify_board_footprints(footprints, keys, store)
    log_.add(
        "certify", not cert_violations,
        (f"certificates x{len(footprints)}",), cert_violations,
        req_hash, sha256_text(",".join(
            f"{r}:{compute_footprint_sha256(fp)[:12]}"
            for r, fp in sorted(footprints.items())
        )),
    )
    if cert_violations:
        return _halt("certify", cert_violations)

    # ---- Stage 1: cells ---------------------------------------------------
    from kicad_pipeline.optimization.functional_grouper import detect_subcircuits

    net_pads = _net_pad_map(requirements)
    detected = detect_subcircuits(requirements)
    cells: list[Cell] = []
    cell_violations: list[Violation] = []
    claimed: set[str] = set()
    planned = _plan_cell_members(
        detected, constraints, set(footprints),
        frozen_refs=frozenset(ep.ref for ep in constraints.edge_pins),
    )
    for sub in detected:
        member_refs = [
            r for r in planned.get(sub.anchor_ref, ())
            if r in footprints and r not in claimed
        ]
        if sub.anchor_ref not in member_refs or len(member_refs) < 2:
            continue
        refs_set = frozenset(member_refs)
        externals = _external_nets_for(refs_set, net_pads)
        # Connector-facing nets orient THT anchors (contacts toward the
        # terminals, support halo away) — power rails carry no direction.
        edge_refs = frozenset(ep.ref for ep in constraints.edge_pins)
        interface = frozenset(
            net for net in externals
            if any(pr.ref in edge_refs for pr in net_pads.get(net, ()))
        )
        try:
            cell = generate_cell(
                name=f"{sub.circuit_type.value}:{sub.anchor_ref}",
                kind=sub.circuit_type.value,
                anchor=sub.anchor_ref,
                footprints={r: footprints[r] for r in member_refs},
                constraints=constraints.for_refs(refs_set),
                external_nets=externals,
                interface_nets=interface,
            )
        except CellGenerationError as exc:
            cell_violations.extend(exc.violations)
            continue
        cells.append(cell)
        claimed.update(member_refs)
    # Attachment islands: unclaimed refs bound to EACH OTHER (pull-up
    # <-> button) form their own cluster cell so the bound is enforced
    # inside one rigid body instead of dangling across singletons.
    for island in _attachment_islands(
        constraints, set(footprints) - claimed,
    ):
        anchor_ref = max(
            island, key=lambda r: (len(footprints[r].pads), r),
        )
        refs_set = frozenset(island)
        try:
            cells.append(generate_cell(
                name=f"cluster:{anchor_ref}",
                kind="attachment_cluster",
                anchor=anchor_ref,
                footprints={r: footprints[r] for r in island},
                constraints=constraints.for_refs(refs_set),
                external_nets=_external_nets_for(refs_set, net_pads),
            ))
        except CellGenerationError as exc:
            cell_violations.extend(exc.violations)
            continue
        claimed.update(island)
    for ref in sorted(set(footprints) - claimed):
        cells.append(_singleton_cell(ref, footprints[ref], net_pads))

    log_.add(
        "cells", not cell_violations,
        tuple(f"{c.name}: {', '.join(c.proof.checks)}" for c in cells),
        tuple(cell_violations),
        req_hash, sha256_text(",".join(c.name for c in cells)),
        detail=f"{len(cells)} cells ({len(detected)} subcircuits detected)",
    )
    if cell_violations:
        return _halt("cells", tuple(cell_violations))

    # ---- Stages 2+3: floorplan + legalize ---------------------------------
    edge_pinned = frozenset(ep.ref for ep in constraints.edge_pins)
    seq_refs = frozenset(r for s in constraints.sequences for r in s.refs)
    by_group: dict[str, list[Cell]] = {}
    lifted: list[Cell] = []
    for cell in cells:
        # LONE edge-pinned connectors outside a ladder strip become
        # board-level groups of their own: the group snap can only
        # satisfy ONE edge per group, so each free connector must be
        # free to find its own edge (USB north, power south, ...).
        # Multi-member cells (a connector with its ESD island) stay in
        # their functional group — splitting them would break their
        # own attachment bounds.
        if (len(cell.refs) == 1 and (cell.refs & edge_pinned)
                and not (cell.refs & seq_refs)):
            lifted.append(cell)
        else:
            by_group.setdefault(
                _group_for(cell.refs, requirements), []
            ).append(cell)
    plans = tuple(
        pack_group(
            gname,
            tuple(sorted(gcells, key=lambda c: c.name)),
            sequences=constraints.sequences,
            edge_pinned=edge_pinned,
        )
        for gname, gcells in sorted(by_group.items())
    ) + tuple(
        pack_group(f"conn:{sorted(cell.refs)[0]}", (cell,))
        for cell in sorted(lifted, key=lambda c: c.name)
    )
    obstacles = tuple(_obstacle_cell(n, poly) for n, poly in reserved_zones)
    plan = None
    try:
        plan = pack_board(
            plans, constraints,
            board_width=board_width_mm, board_height=board_height_mm,
            obstacles=obstacles,
        )
        # Edge-snapped groups stay put during legalization: residual
        # overlaps push the INTERIOR groups, never a connector off
        # its edge.
        pinned_groups = frozenset(
            f"group:{p.name}" for p in plans
            if p.edge_facing is not None or p.name.startswith("conn:")
        ) | frozenset(o.cell.name for o in obstacles)
        plan = legalize(plan, pinned=pinned_groups)
    except LegalizationError as exc:
        log_.add("floorplan", False, ("pack_board", "legalize"),
                 exc.violations, req_hash, "")
        return _halt("floorplan", exc.violations, floorplan=plan)
    except Exception as exc:
        v = Violation(
            constraint="floorplan", refs=(), severity=Severity.CRITICAL,
            measured=0.0, limit=0.0, message=str(exc),
        )
        log_.add("floorplan", False, ("pack_board",), (v,), req_hash, "")
        return _halt("floorplan", (v,))

    # ---- Emit in KiCad conventions ----------------------------------------
    positions, rotations = emit_layout(plan, footprints)
    log_.add(
        "floorplan", True,
        (f"groups x{len(plans)}", f"cells x{len(cells)}"),
        (), req_hash,
        sha256_text(repr(positions) + repr(rotations)),
        detail=f"board {plan.board_width:.1f}x{plan.board_height:.1f}mm",
    )

    return PlacementV2Result(
        positions=positions,
        rotations=rotations,
        board_width=plan.board_width,
        board_height=plan.board_height,
        floorplan=plan,
        constraints=constraints,
    )


def to_kicad_rotation(v2_rotation_deg: float) -> float:
    """v2 rotations ARE KiCad rotations (no conversion seam).

    The whole placement_v2 package uses the KiCad convention natively
    (see ``cells`` module docstring); this normalizes to [0, 360).
    """
    return v2_rotation_deg % 360.0


def emit_layout(
    plan: Floorplan, footprints: Mapping[str, Footprint],
) -> tuple[tuple[tuple[str, float, float], ...], tuple[tuple[str, float], ...]]:
    """Convert a floorplan to KiCad origin positions + rotations."""
    positions: list[tuple[str, float, float]] = []
    rotations: list[tuple[str, float]] = []
    for pc in plan.placed:
        for m in pc.members_in_board():
            fp = footprints.get(m.ref)
            if fp is None:
                continue
            kicad_rot = to_kicad_rotation(m.rotation_deg)
            ox, oy = centroid_to_origin(fp, m.x, m.y, kicad_rot)
            positions.append((m.ref, ox, oy))
            rotations.append((m.ref, kicad_rot))
    return (tuple(sorted(positions)), tuple(sorted(rotations)))


__all__ = [
    "PlacementV2Result",
    "emit_layout",
    "instantiate_array",
    "run_placement_v2",
    "to_kicad_rotation",
]

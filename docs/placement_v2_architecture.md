# Placement Engine v2 — Cells, Contracts, and Proofs

**Status**: PROPOSAL (2026-06-10)
**Replaces**: the 25→32-phase L3 engine AND the partial bottom-up rewrite (subcircuit_layout.py / board_packer.py)
**Supersedes**: docs/bottom_up_placement_plan.md (absorbed — see §9)

---

## 0. The Diagnosis (why every previous attempt failed the same way)

Five structural defects, each verified in the current code:

| # | Defect | Evidence |
|---|--------|----------|
| D1 | **Mutable shared state, 32 phases** — any phase can move any component placed by any earlier phase. "This is the last word — no phase runs after this" appears 3× in placement_optimizer.py. | placement_types.py `PlacementContext` (deliberately unfrozen); phases 22–32 exist solely to undo damage from phases 1–21 |
| D2 | **Placement by TYPE, not NET** — decoupling caps placed "near the IC's bounding box", not at the VCC pad. Net proximity is a late-stage correction (`_enforce_net_proximity`, phase 29) instead of the placement principle. | 65+ net-connected pairs >30mm apart on a 160×80 board |
| D3 | **Derived geometry doesn't follow its component** — antenna keepout is built once in footprint frame and NOT re-derived after rotation or late moves; connector rotation computed against `ctx.initial_pcb` (stale Level-1 positions). | footprints.py `_esp32_make_antenna_keepout`; ee_phases.py:260 `_best_connector_rotation` |
| D4 | **Checks don't block, fixes don't verify** — DRC rules 003–006 are check-only; `no_collisions` gate allows 25 collisions; tracked-gate enforcement deadlines (April 2026) silently lapsed. The model could *claim* success because nothing forced it not to. | placement_guard.py, evals/ |
| D5 | **Footprint/3D correctness is best-effort** — JLCPCB rejection silently falls back to a parametric footprint that may be a different package (KI-022); registry lookup uses substring matching ("SOT-23-3" matches "SOT-23-5"); registry offsets are hand-maintained with no sync check against the KiCad library. | footprints.py:3598, :596 |

**The meta-defect**: correctness in the current system is a *claim* (a score, a log line, an agent saying "fixed"). Nothing in the architecture makes false claims impossible. The redesign's organizing principle is:

> **Nothing is true because an agent said it. Everything is true because a deterministic verifier re-derived it from the artifact on disk.**

---

## 1. The Architecture in One Picture

Borrowed from VLSI physical design (cell library → PCells → floorplan → legalize → sign-off), adapted to PCB:

```
                    ┌─────────────────────────────────────────────┐
                    │ CONSTRAINT IR (single source of truth)       │
                    │ compiled from netlist + part rules + user    │
                    └──────┬──────────────────────────┬───────────┘
                           │ consumed by SOLVERS      │ consumed by VERIFIER
                           ▼                          ▼ (same objects — cannot drift)
┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐
│ S0 Cert  │──▶│ S1 Cell Gen  │──▶│ S2 Floorplan │──▶│ S3 Legalize  │──▶│ S4 Sign-off │
│ footprint│   │ (bottom-up)  │   │ (top-down)   │   │ (one QP pass)│   │ (verify the │
│ + 3D     │   │ subcircuit → │   │ cells → board│   │ minimal moves│   │  FILE, then │
│ library  │   │ rigid cell   │   │ global optim │   │ whole cells  │   │  vision)    │
└──────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └─────────────┘
     │               │                  │                  │                  │
     ▼               ▼                  ▼                  ▼                  ▼
  manifest        manifest           manifest           manifest           manifest
  ════════════════════ append-only BUILD LEDGER (hashes, invariants, verdicts) ═══════
```

Each stage is a **pure function**: frozen artifact in → frozen artifact out + a machine-checked **contract**. A stage that violates an upstream invariant doesn't get "fixed later" — the build **halts** with the violated invariant named. There are no fixup phases because there is nothing to fix up: once placed, geometry is immutable.

---

## 2. The Constraint IR — rules as data, shared by solver and verifier

Today, placement logic lives in ~9,000 LOC of phase code and checking logic lives separately in placement_guard.py / scoring.py / review_agent.py — three parallel encodings of the same intent that drift apart. v2 collapses them into one typed IR:

```python
@dataclass(frozen=True)
class PinAttach:        # decoupling cap C3 pads straddle U1.VDD/GND pads
    src: PadRef; dst: PadRef; max_mm: float; ideal_mm: float

@dataclass(frozen=True)
class SequenceAlong:    # buck chain: VIN → L → SW → FB left-to-right; ADC/relay arrays
    axis: Axis; refs: tuple[str, ...]; pitch_mm: float | None  # None = derived

@dataclass(frozen=True)
class EdgePin:          # J1 on south edge, opening outward
    ref: str; edge: Edge; face_out: bool

@dataclass(frozen=True)
class CellKeepout:      # antenna keepout, relay isolation slot — CELL-LOCAL frame
    owner: str; polygon: Polygon; applies_to: KeepoutKind  # transforms WITH the cell

@dataclass(frozen=True)
class IsolationGap:     # mains relay domain vs logic domain
    domain_a: str; domain_b: str; min_mm: float

@dataclass(frozen=True)
class BoardContain:     # every pad + courtyard + 3D body inside outline, rotation-aware
    margin_mm: float
```

**Compilation sources** (in priority order):
1. Netlist topology — every passive's pin adjacency derived from actual pad-to-pad nets (`PinAttach` for decoupling, `SequenceAlong` from `trace_linear_chains()` for dividers/buck chains)
2. Part rules — per-part-class YAML (relay → isolation slot under COM, ESP32 → antenna keepout polygon, crystal → guard ring) shipped with the cell library
3. Human feedback locks — every sign-off comment becomes a persisted constraint, so a human correction can never regress (this replaces "reported 3×" bugs)

**The key property**: `solve(constraints)` and `verify(board_file, constraints)` take the *same objects*. A constraint cannot be enforced one way and checked another. When the verifier passes, it is a proof, not an opinion.

---

## 3. Stage 0 — Certified Part Library (kills D5)

A board may only instantiate **certified part tuples**. Certification is content-addressed:

```
Certificate {
  lcsc, footprint_sha256, model_sha256,
  model_offset (computed via pin_map.compute_centroid_offset — already the
                single source of truth), model_rotation,
  pad_map (pin → pad geometry), courtyard_polygon, body_polygon_3d,
  cell_keepouts (antenna, isolation slots — in footprint-local frame),
  evidence: { renders_sha256[], vision_verdict, geometry_checks: 10/10 },
  certified_at, kicad_lib_version
}
```

Rule changes from today:
- **No silent fallback.** JLCPCB footprint rejected → build error naming the part, never a parametric substitute of a possibly-different package. Parametric generation is allowed only when the requirements explicitly say `footprint_source: parametric`, and the generated footprint must then pass the same certification (pad count/pitch/size vs. the requested package spec) before use. *(closes KI-022)*
- **Exact-key registry lookup only**: LCSC → certificate. Substring/lib-id fuzzy matching deleted. A missing certificate is a build error with a one-command remedy (`verify_components.py --certify C12345`).
- **Library-sync check**: certification records `kicad_lib_version`; a nightly job re-hashes referenced KiCad lib footprints and *invalidates* certificates whose source hash changed (stale-registry problem becomes impossible to not notice).
- **3D ground truth**: the certificate's `body_polygon_3d` (projected outline of the STEP body at its computed offset) is what S3/S4 use for body-collision and board-containment — so "3D body sits exactly over the pads" is verified at certification time once, and board-level checks reuse that proven geometry.

Most of this exists (component_verifier's 10 checks, golden baselines, vision checks). What changes: it becomes a **hard gate keyed by hash**, and its outputs (courtyard, body polygon, keepouts) become the *inputs* to placement rather than a parallel system.

---

## 4. Stage 1 — Cell Generators (bottom-up; kills D2, and makes "frozen" structural)

For every detected subcircuit (relay_driver, buck_converter, adc_channel, decoupling, crystal_osc, esd_protection, poe_filter, optocoupler, generic_ic_cluster), a **generator** produces a `Cell`:

```python
@dataclass(frozen=True)
class Cell:
    name: str                                  # "relay_ch2", "buck_5v"
    members: tuple[PlacedMember, ...]          # ref, xy, rotation — CELL-LOCAL, final
    polygon: Polygon                           # tight hull + clearance margin
    ports: tuple[Port, ...]                    # where each external net exits the hull
    keepouts: tuple[CellKeepout, ...]          # antenna zone etc., cell-local
    proof: CellProof                           # every internal constraint verified
```

How a generator places (pin-connectivity, never type-proximity):
1. Anchor (IC/relay/connector) at origin, rotation chosen by port budget (where external nets need to leave).
2. Each support component attached at its **actual pad pair**: a decoupling cap is centered on the segment between the IC's VDD pad and the nearest GND pad, not "near the IC". Uses the certificate's `pad_map` — exact coordinates, no bounding-box guessing.
3. `SequenceAlong` constraints realized literally: buck chain VIN→L→SW→FB laid on the signal axis; divider R-R-clamp-filter as a strip pointing at its terminal port.
4. Tiny force-directed relaxation (members ≤ ~12) with hard no-overlap, then **internal verification**: every `PinAttach` within `max_mm`, zero courtyard overlaps, keepouts clear. Fails → generator error, build halts. No downstream rescue.

**Arrays are template instancing, not repeated optimization.** Relay channels / ADC channels / buck phases: lay out **one** channel cell, verify it, then *instantiate it N times* — identical by construction, which is exactly the "repeatable patterns" requirement. The array itself is a parent cell with a `SequenceAlong` over child cells at fixed pitch, terminals bound to channels **by net** (terminal Jx belongs to the relay whose COM it carries — kills the recurring J↔K misalignment bug, R01/PLI-015).

**After verification a Cell is immutable.** Downstream stages hold only `(cell, translation, rotation ∈ {0,90,180,270})`. Member positions cannot be touched because there is no API to touch them. This is what relay_template.py wanted to be; here it's enforced by the type system, not by a `relay_support_refs` flag that 20 phases must remember to respect.

**Keepouts transform with the cell** (kills D3): the antenna keepout is part of the ESP32 cell; rotate the cell, the keepout rotates. S4 re-derives the expected keepout from the final position+rotation and asserts the board file's zone matches it exactly.

---

## 5. Stage 2+3 — Floorplan (top-down; one global objective, kills D1)

Replace 25 heuristic phases with **one optimization problem solved twice (coarse, then exact)**:

**Variables**: per group then per cell — translation + rotation (0/90/180/270).
**Objective**: total **port-to-port** ratsnest length (cell ports, not member centroids — much better proxy for routability) + isolation-aware domain penalty + board-area term (for shrink-to-fit).
**Hard constraints** (never traded off):
- No polygon overlap between cells (no-fit-polygon test on actual hulls)
- `BoardContain`: pad extent AND 3D body extent inside outline at the chosen rotation
- `EdgePin`: connectors on their edge, opening outward, placed **first** as anchors
- `CellKeepout` regions clear of foreign cells
- `IsolationGap` between voltage domains

**Solvers, by scale**:
- **Group level (~6 groups)**: small enough for branch-and-bound / exhaustive over rotation assignments with an NFP packer for positions — *exact*, deterministic, no annealing randomness at the level the human sees most.
- **Cell level within group (3–15 cells)**: sequence-pair simulated annealing with **seeded RNG** (same input → same board, so golden tests are byte-exact), warm-started from the group's signal-flow order.
- **Board shrink**: solve on a 2× oversized outline, take the bounding hull + margin, re-verify containment, optionally iterate compression to a target size. (Keeps the "oversized then shrink" decision from prior feedback.)

**What top-down contributes**: the floorplanner sees the whole board — connector edges, domain isolation, antenna-to-edge — *before* any cell is committed, so global structure is never the victim of local fixes. **What bottom-up contributes**: every cell is already perfect inside, so the floorplanner's moves can't break a decoupling distance — it is physically unable to.

---

## 6. Stage 3.5 — Legalization (one pass, minimal-motion, not push-apart loops)

Coarse solvers can leave sub-millimeter residuals. Today's answer is 10-pass push-apart loops that cause cascades. v2: a single **constraint-projection QP** — find the minimum total cell motion that satisfies all hard constraints simultaneously (linearized non-overlap + containment around the current solution; cells move as wholes). If infeasible → the board is genuinely too small → report exactly which constraints conflict, halt. No silent degradation, no oscillation, provably terminates because it runs once.

---

## 7. Stage 4 — Sign-off: verify the FILE, then the PIXELS, then the HUMAN

Three gates, all blocking, all recorded in the ledger:

**Gate A — Geometric proof (deterministic):** re-parse the **written `.kicad_pcb` from disk** (verifying the artifact, not in-memory state — also catches writer bugs) and re-check the *entire Constraint IR* plus the 7 DRC rules: zero pads off board, **zero** courtyard collisions (not ≤25 — that gate is deleted), every PinAttach within bound, arrays in sequence, keepout polygons exactly where the final position+rotation says, isolation slots under COM pins, THT connectors at edge facing out. Every threshold is the IR's threshold — no second encoding.

**Gate B — Visual adversarial review (vision as scout, geometry as garrison):** render the 4-view standard; vision agents (in subagents, never main context) get a structured checklist per view and return machine-readable verdicts (`{check_id, pass, refs, evidence}`). Any FAIL blocks. Plus golden-reference image diff per training board (renders are deterministic, so a byte-identical placement diffs clean). **Standing rule**: when vision finds a defect class Gate A missed, fixing the board is not sufficient — a new IR constraint or verifier rule MUST be added in the same change (rule-engine candidate), so vision findings monotonically convert into deterministic checks.

**Gate C — Human sign-off**, group-by-group as today, but every comment is compiled into a persisted constraint (§2 source 3) before the next iteration. Human feedback becomes part of the proof obligations forever.

**The Build Ledger** makes "lying about it" structurally impossible: each stage appends `{stage, input_sha256, output_sha256, constraints_checked, constraints_passed, verdicts, duration}` computed by the harness, not by the model. "Done" is defined as *ledger shows all gates green*; any agent claim is checkable against the ledger in one command (`pipeline ledger show <build>`). The existing eval suite becomes a consumer of ledgers (golden cases assert exact ledger contents), and the commit gate checks the ledger instead of re-running heuristics.

---

## 8. Scoring's new job

The 19-dimension score stops being a correctness instrument (it provably isn't one — Grade A with 12 visual defects). Correctness = Gates A/B/C, binary. The score survives only as the **floorplanner's objective function** (ratsnest length, compactness, flow) — a preference among already-correct boards. This single change resolves the "scores lie" class of failure permanently: a lying score can no longer approve anything.

---

## 9. What is kept, absorbed, deleted

| Existing | Fate |
|----------|------|
| geometry.py, pin_map.py (centroid math), BoardZone | **KEEP** — add convex_hull, NFP, polygon offset |
| functional_grouper.py (subcircuit detection) | **KEEP/EXTEND** — feeds cell generators; add ESD/POE/opto/generic-IC types (from bottom-up plan Phase A) |
| component_verifier + evidence + golden baselines + visual_inspector | **PROMOTE** → Stage 0 certification gate |
| placement_guard.py checks, drc_rules_spec.md rules | **ABSORB** → verifier rules over the Constraint IR (single encoding) |
| relay_template.py | **ABSORB** → relay_driver cell generator + array instancing |
| subcircuit_layout.py / board_packer.py (April WIP) | **ABSORB** — correct instincts (pin-connectivity layout, hull polygons, edge pinning); rewrite onto Cell/IR types; Stage 2 was never written anyway |
| scoring.py | **DEMOTE** → floorplan objective only (§8) |
| ee_phases.py, ee_phases_groups.py, ee_phases_refinement.py, level3_phases.py, subnet_placer.py, collision_resolver.py (~9,000 LOC) | **DELETE** after cutover (kept on a branch) |
| reference_comparator.py | **KEEP** — Gate B golden diffs |

**Migration**: v2 is built alongside v1 behind `placement_mode="v2"`. Cutover per training board (relay → analog → MCU → power → ethernet, the established order) when v2 passes Gate A with zero violations where v1 currently shows 65+. v1 is never "improved" again — every fix goes into v2.

**Also fix immediately (pre-work)**: the orchestrator/models circular import that crashes the eval suite — nothing can be gated until `python -m kicad_pipeline.evals` runs at all.

---

## 10. Implementation plan

| Phase | Deliverable | Est. | Depends on |
|-------|-------------|------|------------|
| P0 | Fix evals circular import; geometry.py: convex_hull, NFP, polygon offset, transform | 0.5 d | — |
| P1 | Constraint IR + compiler (netlist → constraints; part-rule YAML; feedback-lock store) | 1.5 d | P0 |
| P2 | Stage 0: certificate schema, hash-keyed registry, no-fallback enforcement, library-sync invalidation | 1 d | P0 |
| P3 | Cell type + generators: decoupling, relay_driver, buck, adc_channel, crystal, esd, generic_ic; array instancing | 2 d | P1, P2 |
| P4 | Floorplanner: group-level exact packer + cell-level seeded SA + shrink-to-fit; legalization QP | 2 d | P3 |
| P5 | Gate A verifier (file-level, full IR re-check) + Build Ledger + eval-suite/commit-gate integration | 1 d | P1 |
| P6 | Gate B: vision checklist protocol, machine-readable verdicts, golden diffs, "vision finding → new rule" workflow | 1 d | P5 |
| P7 | Cutover: relay training board end-to-end on v2, group-by-group human sign-off, then remaining boards | 1–2 d each | P4–P6 |

Each phase lands with tests (mypy --strict, targeted pytest), and P5 onward every build emits a ledger. Parallelizable: P1/P2 independent; cell generators in P3 are one-agent-per-generator with non-overlapping files.

---

## 11. Why this cannot fail the way v1 failed

| v1 failure mode | v2 structural answer |
|---|---|
| Phases undo each other | Pure-function DAG; placed geometry has no mutation API |
| Decoupling cap far from IC | PinAttach to actual pads inside an immutable cell — violation is impossible post-Stage-1 |
| Arrays inconsistent / out of order | One cell instantiated N times + SequenceAlong — identical and ordered by construction |
| Parts off board | BoardContain over pad+body extent is a hard floorplan constraint AND re-proven on the written file |
| Keepout not under the antenna | Keepout is cell geometry; transforms with the cell; Gate A re-derives and compares |
| Wrong footprint silently substituted | No uncertified tuple can be instantiated; fallback is a build error |
| 3D body off its pads | Certified offset + body polygon proven once at Stage 0, reused everywhere |
| Agent claims false success | Ledger computed by harness; "done" = ledger green, not prose |
| Score says A, human says F | Score demoted to preference; correctness is binary gates |
| Human-reported bug recurs | Every sign-off comment compiles to a persisted constraint checked forever |

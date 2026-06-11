# Why the Framework Didn't Apply the Board Spec by Default

Human review of the first nl-s-3c v2 build (2026-06-11) found the
layout ignored standing spec requirements: screw terminals scattered
across edges instead of one side, MCU/IO not on the opposite side, no
isolation zones around Analog/Relay/24V with ferrite boundaries, relays
not in a tight 1x4 bank with their harness terminal adjacent, group/
subgroup hierarchy not preserved.

## The one-sentence root cause

**Placement v2 honors exactly what reaches its ConstraintSet — netlist
topology, `data/part_rules.json`, and feedback locks — and the board
spec lives in none of them.** The spec exists as prose
(`docs/placement_group_requirements.md`, written from the human's
manual reference design) and as the reference board itself
(`rereference/nl-s-3c-complete-2026-03-09_234333`). Nothing in the
build path reads either. The floorplanner optimized the constraints it
could see (pairwise attach bounds, per-connector edge pins, HPWL) and
was structurally blind to everything the spec says about SETS of
components: edge cohorts, isolation regions, zone ordering, hierarchy.

This is the same defect class as the un-calibrated terminal openings
(Gate C item 1): an input the system needed was asserted to exist —
"the spec is documented" — but never became machine-readable data, so
every check downstream validated the wrong thing or nothing.

## Per-miss trace

| Spec requirement (placement_group_requirements.md) | Reference board (measured) | Why v2 missed it |
|---|---|---|
| §3: all field-wiring terminals on one edge; analog connectors same edge | J1/J3/J6 flush NORTH; J4/J5 inside the analog zone | No "edge cohort" concept. Each lifted connector independently picks nearest/preferred edge. EdgePin.edge exists per-ref but nothing sets it. |
| MCU + IO (USB, ethernet, SD, GPIO) on the opposite side | J16/J2/J13/J15 + U3 all SOUTH | Same gap — no cohort, no "opposite side" relation. |
| J14 display vertical at an edge (ribbon) | EAST edge, rot 90 | Edge yes (generic), VERTICAL only by accident of the long-axis rule. No way to say "this edge, this orientation". |
| §4.2: Analog and Relay/24V in isolation zones, ferrites L3-L6 on the zone borders | Keepout zones (0,0)-(55,44) analog, (74,0)-(151,34) relay, RF zone at U3; L1-L6 in the central corridor; outline SLOTS between zones | The only isolation IR is a pairwise courtyard gap (MAINS/LOGIC from part rules). There is NO zone-region IR, no ferrite-boundary semantics, no Gate A region check — the framework cannot even represent the requirement. |
| §1/§2: zone → group → subgroup hierarchy, subgroup spread bounds | Hierarchy visible in the reference layout | Requirements FeatureBlocks are FLAT. Cells capture the subgroup level, groups the FeatureBlock level — the ZONE level and the spread bounds exist nowhere, and Gate A never verifies group/subgroup cohesion. |
| Relay bank 1x4 with harness terminal adjacent | K1-K4 at y=30.6, pitch 16.8, J1 directly north | The K1-K4 row sequence DOES compile, but strips align cell origins, not relay centroids (cells have different internal shapes), and nothing pairs the bank with its shared harness connector (the trainer's ladder pairing needs per-channel terminals). |

## Why "documented" wasn't enough — the standing principle

A spec requirement only counts as implemented when it is (a) expressed
as data the compiler reads, (b) enforced or optimized by a solver, and
(c) re-derived from the artifact by a gate. The openings calibration,
courtyard containment, and fanout checks all followed this path this
week; the board-level spec never did. Prose docs are calibration
SOURCES, not constraints.

## The fix: spec-as-data, with framework defaults

Ordered so each step is independently shippable; defaults mean future
boards get this WITHOUT writing a spec file.

1. **Edge cohorts (default behavior)**. Part-class derived: every
   TerminalBlock-class connector joins the `field_wiring` cohort; USB/
   RJ45/SD/pin-header IO joins the `io` cohort; cohorts claim OPPOSITE
   edges (which physical edge is free is the floorplanner's choice;
   explicit per-board override via feedback locks `edge_pin.edge`).
   Gate A: new check — cohort members share one edge.
   *Bridge available today*: feedback locks already carry per-ref
   edges; threading `feedback_locks_path` through `build_pcb` lets a
   board pin J1=north, J2=south, ... with zero new IR (done 2026-06-11).

2. **Isolation zones (default from voltage domains)**. The analysis
   already exists: `functional_grouper.PowerFlowTopology`,
   `detect_cross_domain_affinities`, and the ferrite-boundary doctrine
   ("L-series at group boundaries"). New IR `IsolationRegion(name,
   refs, min_gap_mm, boundary_refs)`; compiler derives regions from
   ferrite-separated rail subtrees (RELAY_5V behind L3/L5, AVCC behind
   L4/L6, 24V input side); floorplan packs each region contiguously
   with the gap and places boundary ferrites ON the border; Gate A
   re-derives region hulls from the artifact and checks gap + ferrite
   betweenness + no foreign component inside a region.

3. **Zone-first packing (macro-first, spec §6.1)**. pack_board gains a
   zone level above groups: regions placed by power-flow order, groups
   packed inside their region. This is also where the board-utilization
   objective (the standing fab-persona blocker) belongs.

4. **Subgroup spread gate**. Detected-subcircuit members already ARE
   the subgroups; add Gate A check: member spread from cell anchor
   ≤ 10mm passives / 15mm with ICs (spec §2.3 numbers).

5. **Bank-terminal pairing**. Generalize the trainer's ladder pairing:
   a connector whose pins carry a relay bank's contact nets is the
   bank's terminal — place flush on the cohort edge directly outboard
   of the row, pitch-aligned. Strip alignment by relay CENTROID rather
   than cell origin fixes the ragged row.

## Status

- Bridge (step 1's lock-file form) implemented for nl-s-3c on
  2026-06-11; J14 vertical-east, terminals north, IO south.
- Steps 1 (default cohorts) through 5 are the placement-v2 work queue,
  ahead of the silkscreen pass and utilization objective from
  `gate_c_resolution_2026-06-11.md`.


## Addendum (2026-06-11, evening): connector-as-subgroup supersedes edge-sliding

The crossing-reorder pass (shipped) fixed along-edge ORDER (J1 landed
right-of-terminals above the relays with zero hints — the acceptance
test). But the J14 case exposed the next layer, stated by the board
owner: "Not just across edges — this is part of grouping and
subgrouping. J14 should be a subgroup of the IC group and naturally be
closer to the IC."

Design consequence: an edge-pinned connector whose nets terminate
overwhelmingly in ONE group (J14 -> U3 SPI) is that group's EDGE
SUBGROUP — it must claim the edge SEGMENT adjacent to its parent
group's placement, not an independent shelf slot. This replaces the
lone-connector slide for associated connectors:

1. compile: derive connector->group association (majority of the
   connector's signal-net partners in one FeatureBlock) — emit it on
   the EdgePin (or a new GroupAssoc IR).
2. floorplan: locked/lifted connectors WITH an association pre-place
   on their edge at the along-position nearest their parent group's
   packed position (group first, connector second), then the
   crossing-reorder permutes within that neighborhood only.
3. Gate A: connector-to-parent-group distance check (the K3/K4
   "stranded from serving terminals" fab finding, generalized).

Known issue: the lone-connector slide does not fire for J14 on the
final plan despite an offline probe showing a strict improvement at
centroid y~43 (collision-clear) — discrepancy unresolved; superseded
by the subgroup mechanism above for associated connectors, but the
probe/slide mismatch should be root-caused before the slide is
trusted for genuinely unassociated connectors.

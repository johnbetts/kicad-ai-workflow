# Project Specs – AI PCB Generation Framework
> **Single source of truth** for placement, routing, optimization, and quality rules.  
> **Version:** March 2026 | Generic edition – no board-specific names

---

## ⚠️ Pre-Task Checklist (Claude Must Confirm Before ANY Code Change)

Before generating, refactoring, or improving layout code, explicitly state which of these you are addressing:

- [ ] All passives placed adjacent to dominant/primary pin(s) when role allows
- [ ] No via ping-pong or unnecessary layer jumps
- [ ] No circular, looping, or indirect detour traces
- [ ] All nets 100% routed — zero opens or dangling segments
- [ ] GND copper pours present, connected, and stitched on **both** F.Cu and B.Cu

Mark N/A with reason if a change does not affect an item.

---

## Core Objectives (Non-Negotiable)

| # | Rule                                      | Target / Metric                              |
|---|-------------------------------------------|----------------------------------------------|
| 1 | **100% routed**                           | Zero unrouted nets, zero open pins           |
| 2 | **DRC/ERC clean**                         | Zero violations at final export              |
| 3 | **Via minimization**                      | Avg < 0.5 vias/net; hard max 2 per net       |
| 4 | **Trace efficiency**                      | ≤ 1.4× Manhattan ideal; no loops/detours     |
| 5 | **Passive proximity**                     | ≤ 6–8 mm (target < 5 mm) to dominant pin     |
| 6 | **Ground planes**                         | Both layers; stitching every 10–20 mm        |
| 7 | **JLCPCB defaults**                       | 2-layer 1.6 mm FR4; 0.20/0.20 mm min rules; LCSC Basic preferred |

---

## Placement Rules

### Priority Order (Always enforce top-down)

1. Fixed mechanicals (outline, holes, mounting, large fixed connectors)
2. Main ICs, MCUs, regulators, power devices
3. All connectors, ports, headers, sensor interfaces
4. Crystals, oscillators, timing components (with keep-outs)
5. Decoupling / bypass capacitors — **must** be placed first among passives
6. Series resistors, pull resistors, address/configuration resistors, terminators
7. Filter networks, protection diodes, beads
8. LEDs, status elements, test points

### Passive Placement Classification & Enforcement

Classify **every** passive component by net connectivity/role before final placement:

**Class A — Single-dominant-pin passives**  
- Connected primarily to **one** high-degree pin (MCU/connector/IC pad) + simple net (GND/power/rail)  
- Typical roles: pull-up / pull-down / series termination / address select / current-limit / level-shift resistor  
→ **Rule:** Place **≤ 6 mm** (ideally < 5 mm) from the dominant pin.  
→ **Priority:** Highest — relocate these **before** routing or global optimization.

**Class B — Local decoupling / bypass capacitors**  
→ Place within **5–10 mm** (target < 6 mm) of the IC/regulator power pin(s) they serve.  
→ Position to minimize loop area (between pin and nearest GND via/plane).

**Class C — Paired / multi-passive networks** (RC filters, dividers, protection clusters)  
→ Place as tight group near the most critical or highest-degree node in the subcircuit.

**Implementation – Passive Proximity Pass** (run after initial placement & repeatedly in loop)

```python
for comp in board.get_components():
    if comp.is_passive() and not comp.is_fixed():
        dominant_pins = get_dominant_connected_pins(comp)  # highest degree or connector/IC pins
        if dominant_pins:
            target = average_position_of(dominant_pins)
            move_closer_to(comp, target, max_dist=8.0)     # force proximity
            optimize_rotation_for_min_path_length(comp, dominant_pins)
            snap_to_grid_and_resolve_overlaps(comp)

### Passive Proximity Constraint (Automated — Section 4b)

Every passive component (R, C, L, D) sharing a **signal** net with an IC,
switch, or connector MUST be placed within **5 mm** of that dominant component,
targeting the specific connected pin. Decoupling capacitors have a tighter
3 mm limit (section 4). Power-only nets are excluded from this rule to avoid
pulling everything toward VCC/GND pins. Priority 25 (below decoupling at 30,
above GROUP at 16).

## Routing Rules

### MST Seed Selection

The autorouter MST seed pad is the pad closest to the **centroid** of all pads
in the net, not an arbitrary first pad. This produces more balanced spanning
trees and shorter total trace length, especially for nets with asymmetric pad
layouts (e.g., one outlier connector + clustered passives).

### Via & Layer Policy

Single-layer preference mandatory — attempt full net routing on one layer first.
Forbidden: unnecessary layer ping-pong (top → bottom → top).
→ Detect & rip-up any net with ≥2 direction changes on the same axis.
Via cost: 14–20× normal trace segment.
Hard limit: 2 vias/net maximum; 0–1 strongly preferred.

Trace Path Quality & Straightness

Direct endpoint alignment — when a trace connects a connector/sensor pin to a nearby passive, the path must go straight or near-straight toward the passive pad — no circling around it.
No loops or unnecessary curvature — any path that increases length >15–20% over shortest orthogonal route is invalid.
Manhattan bias — heavily penalize non-orthogonal segments unless blocked.
Rip-up triggers (apply after every routing attempt):
length > 1.55–1.65 × Manhattan between farthest endpoints
4 bends on net < 40 mm long
trace visibly detours around an adjacent component it could connect directly to
trace makes large arc/loop instead of straight or L-shape path


Python# Aggressive trace straightening & directness pass
for net in board.nets:
    if (net.length > 1.6 * net.manhattan_length()
            or net.bend_count > 4
            or net.has_loop_or_arc()
            or net.has_avoidable_detour_around_nearby_passive()):
        rip_up(net)
        re_route_with_straightness_and_endpoint_bias(net)
Routing Priority Sequence (Strict)

Power distribution nets (wide traces or zones)
Ground nets → fill pours on both layers
Critical / timing-sensitive nets
Short high-connectivity nets
All remaining nets

Ground Plane Rules (Mandatory)

Create GND zone on F.Cu and GND zone on B.Cu — priority high
Connect every GND pad/pin directly to nearest zone (thermal relief if current >500 mA)
Via stitching: grid every 10–20 mm; minimum 4–6 vias per 50×50 mm GND area
Extra vias near connectors / regulators / high-current paths
Clearance: 0.3–0.5 mm from non-GND features
Penalty: +50 to cost if either layer lacks connected GND pour


Cost Function (Updated Weights)
Pythonboard_cost = (
    1.00  * total_trace_length
  + 16.0  * total_vias                     # even higher to crush ping-pong
  + 3.00  * total_bends
  + 6.00  * sum(max(0, length_ratio - 1.55) for net in nets)
  + 70.0  * drc_violations
  + 200.0 * num_unrouted_segments_or_pins  # absolute top priority to eliminate
  + 12.0  * max_congestion
  + 18.0  * avg_passive_to_dominant_pin_distance
  + 25.0  * num_via_chains_or_ping_pong
  + 10.0  * missing_gnd_pour_penalty_per_layer
  + 8.00  * detour_around_nearby_passive_penalty   # new – catches circling traces
)

Global Optimization Loop Structure
textrepeat 5–12 times or cost improvement < 3%:

  1. PASSIVE PROXIMITY & ROTATION PASS
     → Force Class A/B passives ≤6–8 mm of dominant pins
     → Optimize rotations for straightest paths

  2. ROUTING PHASE
     → Power/GND first (zones)
     → Critical nets
     → Remaining nets with strict via & straightness limits

  3. AGGRESSIVE CLEANUP
     → Rip-up any net violating length, bend, loop, detour, or unrouted rules
     → Re-route with highest straightness/endpoint bias
     → Fill & verify GND pours both layers + stitching
     → Detect & eliminate via ping-pong patterns

  4. VALIDATION
     → Full DRC/ERC
     → Connectivity check (zero opens)
     → If violations/opens remain → small placement perturbation + retry

Final Export & Quality Gates

Must pass: zero DRC violations, zero unrouted pins/nets
GND pours verified filled and stitched on both layers
All passives within distance spec (especially Class A)
Include test points on power rails + key signals
Silkscreen: version, polarity, connector labels
Export full JLCPCB package (Gerbers, drill, BOM+CPL with LCSC mapping)

Use this generic spec consistently — it should prevent far passives, circling traces, ping-pong vias, and unrouted nets on any generated board.

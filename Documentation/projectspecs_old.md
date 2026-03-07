# PCB Auto-Generation Optimization Guide  
For Claude + Python + KiCad (pcbnew) Frameworks  
Complete Edition – March 2026

This document is the master reference to feed to Claude (or any LLM) when asking it to improve, refactor, extend, or debug your AI-driven PCB generation code.  
It consolidates every major recommendation from our discussion to eliminate the most common generated-PCB problems:

- scattered passives far from pins  
- excessive vias (10+ per net)  
- long, zig-zagging, detour-heavy traces  
- poor component rotations and orientations  
- central congestion with empty board edges  
- unnecessary layer changes  
- weak power/ground handling  
- lack of global optimization / rip-up loops  

Goal: Produce clean, short-trace, low-via, balanced, JLCPCB-ready 2–4 layer boards that look professionally hand-routed.

## 1. Definition of Success (Ranked Priorities)

1. Zero DRC violations & zero unconnected pins  
2. Total via count ≤ number_of_nets / 3 (ideally << 1 via per net average)  
3. Most nets routed within 1.3–1.5× their Manhattan ideal length  
4. Very few bends (>4–5 per net is suspicious)  
5. Passives placed within 6–15 mm of their primary connected pin(s)  
6. Strong functional clustering (sensors near ports, decoupling near ICs)  
7. Power & ground use planes (4-layer) or wide traces/zones (2-layer)  
8. Visually balanced layout — no extreme hot-spots or empty regions  
9. LCSC Basic/Preferred parts prioritized, realistic fab rules enforced  

## 2. Fixed → Movable Placement Order (Critical First Step)

Always place in roughly this sequence:

1. Board outline / keep-out / mounting holes / edge connectors (Raspberry Pi HAT, ESP32 devkit footprint, etc.)  
2. Large / fixed / high-pin-count parts (microcontrollers, regulators, relays, displays, magnetics)  
3. Connectors & ports (JST, screw terminals, sensor headers, USB, Ethernet)  
4. Crystals / oscillators (with keep-out zones and short traces)  
5. Decoupling & bypass capacitors (very close to power pins)  
6. Series resistors, terminators, pull-ups/pull-downs  
7. Filter networks (RC, LC, ferrite beads)  
8. LEDs, status indicators, test points  

## 3. Placement Optimization Strategies (Implement at Least 2–3)

### A. Net-Proximity Greedy Placement (Fast & Effective Baseline)

```python
for comp in sorted_by_degree(movable_components, reverse=True):  # high connectivity first
    targets = get_closest_connected_pads_or_modules(comp)        # prefer IC pins over long nets
    candidates = generate_rectangular_grid_around(targets,
                                                  step_mm=0.5,
                                                  max_radius_mm=30,
                                                  respect_board_bounds=True)
    best = min(candidates,
               key=lambda c: (
                   sum(euclidean(pad, t) for pad in comp.pads for t in targets) / len(comp.pads)
                   + 12 * count_overlaps_or_violations(c)
                   + 6 * orientation_mismatch_score(c.rotation)
                   + 4 * density_penalty_around(c.position)
               ))
    place_and_rotate(comp, best.position, best.rotation)
B. Force-Directed / Spring Relaxation Pass (After Greedy)
Pythonfor iteration in range(120, 301):
    for comp in movable_components:
        attract = sum( k_attract * (target_pos - comp.center) / dist**2
                       for target in connected_centroids(comp) )
        repulse = sum( k_repulse / dist**3 * (comp.center - other.center)
                       for other in nearby_components(comp, radius=25) )
        velocity += (attract + repulse - damping * velocity)
        comp.center += velocity * timestep
    snap_to_grid_and_recheck_overlaps()
C. Rotation Scoring
Always try 0/90/180/270. Score rotations by:

How many pads align horizontally vs vertically with primary connection direction
Reduction in expected crossing count
Avoidance of acute angles to nearest target

4. Routing Strategy Ladder (Try in Order)

Power & GND nets first → wide traces (0.8–2 mm) or zones
High-speed / analog-critical nets (clocks, I2C/SPI with pull-ups, ADC inputs)
Short digital I/O nets
General long I/O nets

Routing attempt order per net:

Strategy 1: Single layer (top), orthogonal only
Strategy 2: Single layer (bottom)
Strategy 3: Two-layer, prefer top horizontal / bottom vertical
Strategy 4: Allow 1 via (very reluctant)
Strategy 5: Allow 2–3 vias (last resort)

Use A* or similar with very asymmetric costs:
segment cost          = length × 1.0
via cost              = 8.0 – 15.0
layer change (no via) = 0.5
45° segment penalty   = 1.2 – 2.0
acute angle penalty   = 4.0+
congestion multiplier = 1.0 → 5.0 in crowded areas
5. Post-Routing Cleanup & Rip-up Loop
Must-have final passes:
Pythonfor _ in range(4, 9):
    old_cost = compute_board_cost()

    # 1. Fix high-via nets
    for net in [n for n in nets if n.via_count > 2 or n.length_ratio > 1.65]:
        rip_up(net)
        re_route_with_higher_cost_tolerance(net)

    # 2. Fix long / bendy nets
    for net in [n for n in nets if n.bend_count > 5 or n.length > 1.8 * n.manhattan]:
        rip_up(net)
        try_straighter_path(net)

    # 3. Spread congestion
    if max_local_congestion > threshold:
        perturb_placement_in_region(high_congestion_zone, small_random_shifts)

    # 4. Add ground pours & via stitching
    fill_all_unused_areas_with_gnd_zone(board)
    add_via_stitching(every=12–20)

    new_cost = compute_board_cost()
    if new_cost >= old_cost * 0.97:
        break
6. Comprehensive Cost Function (Use This!)
Pythondef board_cost(board):
    return (
        1.00 * total_trace_length_mm()
      + 10.0 * total_via_count()
      + 1.50 * total_bend_or_corner_count()
      + 3.00 * sum(max(0, length / manhattan - 1.45) for net in nets)
      + 40.0 * drc_violation_count()
      + 100  * (1 if any_unconnected_pins() else 0)
      + 5.00 * max_local_congestion()
      + 8.00 * avg_passive_to_pin_distance_penalty()
      + 12.0 * power_nets_inductance_or_narrow_penalty()
    )
7. JLCPCB & Fabrication / Quality Guardrails

Trace / clearance minimum: 0.20 / 0.20 mm (0.15/0.15 only if necessary)
Default stackup: 2-layer 1.6 mm FR4, 1 oz, HASL lead-free
LCSC priority: Basic (green) > Preferred > Extended
Always add:
4–8 test points (power rails + key signals)
≥2 fiducials (opposite corners)
Clear silkscreen: connector labels, polarity, version/date

Output package must include:
Gerbers, Excellon drill, BOM, pick-and-place (CPL with rotations)
README.md with 3D render screenshot, estimated JLCPCB cost, assembly notes


8. Recommended Claude Prompt Prefix
You are optimizing an AI-driven PCB generation framework using Python + KiCad pcbnew.
STRICTLY follow EVERY rule, technique, cost model, and priority in the attached file:
PCB-Auto-Generation-Optimization-Guide-Complete-v2026.md
Core objectives:

Near-zero vias (target <<1 per net average)
Short, mostly orthogonal traces
Passives placed immediately adjacent to their main pins
Clean, balanced, professional-looking layout
100% DRC-clean & JLCPCB-ready

Current code / snippet:
[paste your code here]
Task:
[describe exactly what you want – e.g. "rewrite the placement function", "add global rip-up-and-retry loop", "improve via minimization", "add ground plane logic", etc.]
Before writing any code:

State which sections of the guide you are applying and why
Describe expected before → after improvements (vias, length, visual quality)
If helper functions are missing, suggest them first

Then provide clean, commented, incremental Python code changes or full functions.
text

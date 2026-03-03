# AI-Assisted KiCad EDA Pipeline — Comprehensive Plan

## Vision

A complete AI-assisted electronic design workflow that takes a project from
high-level requirements ("I need an ESP32 board with WiFi, BLE, Ethernet, and
4 analog inputs") all the way to production-ready manufacturing files — with
every step version-controlled in GitHub and tracked via integrated project management.

The human stays in the loop at every phase gate, but AI does the heavy lifting.

---

## Architecture Overview

```
  Human: "I need an ESP32 board with Ethernet and 4 analog inputs..."
    │
    ▼
  PHASE 1: Requirements Engine
    → Feature decomposition, component selection, pin/power budgets
    → Output: requirements.json, pin_assignment.json, component_selections.json
    → GitHub: init repo, commit, create project board
    │
    ▼
  PHASE 2: Schematic Generation
    → Symbol resolution, subcircuit generation, wiring, ERC
    → Output: project.kicad_sch + erc_report.json
    → GitHub: commit, tag milestone, issues for ERC failures
    │
    ▼
  PHASE 3: PCB Design Rules & Layout
    → AI recommends specs, places components, defines zones/pours
    → Output: project.kicad_pcb (placed, unrouted) + design_rules.json
    → GitHub: commit, update board
    │
    ▼
  PHASE 4: Autorouting
    → Multi-pass routing, via optimization, FreeRouting integration
    → Output: project.kicad_pcb (routed) + routing_report.json
    → GitHub: commit, issues for unrouted nets
    │
    ▼
  PHASE 5: Validation & Testing
    → DRC, electrical validation, JLCPCB manufacturing checks, thermal, SI
    → Output: validation_report.json, issues auto-created
    → GitHub: commit, tag if passing
    │
    ▼
  PHASE 6: Production Artifacts
    → Gerbers, drill files, BOM (LCSC), CPL, assembly drawings, order guide
    → Output: production/ directory, ready-to-upload zips
    → GitHub: commit, tag release, create GitHub Release with assets
    │
    ▼
  PHASE 7: GitHub Integration & Project Management (continuous)
    → Auto-commits, project board, issue tracking, changelog, releases
```

---

## PHASE 1: REQUIREMENTS ENGINE

### 1.1 Purpose

Transform natural-language project description into a structured, validated
requirements document. AI interprets intent, selects components, resolves
conflicts, produces a complete BOM before any schematic work.

### 1.2 AI Decomposition Process

User says: "ESP32-S3 with WiFi/BLE, Ethernet (W5500 SPI), 4 analog inputs
(0-10V industrial), USB-C power/programming, I2C OLED header, buzzer, status
LEDs. 0805 passives. Fit Hammond 1551K enclosure (80x40mm)."

AI decomposes into functional blocks:

```
├── Core MCU: ESP32-S3-WROOM-1 (WiFi+BLE integrated)
│   └── Needs: 3.3V, decoupling, boot/reset circuit, pin budget analysis
│
├── Ethernet: W5500 + RJ45 with magnetics
│   └── SPI bus (4 GPIO) + INT (1 GPIO) + RST (1 GPIO)
│
├── Analog Inputs (×4): 0-10V → voltage dividers → ESP32 ADC1
│   └── Per channel: R divider + TVS protection + RC filter
│   └── Must use ADC1 channels (ADC2 unavailable during WiFi)
│
├── USB-C: ESP32-S3 native USB, CC resistors, ESD protection
│
├── Power: 5V USB → 3.3V LDO
│   └── Current budget: ESP32(240mA) + W5500(132mA) + misc = ~500mA
│   └── AI recommends: AP2112K-3.3 or switching reg if thermal concern
│
├── I2C OLED Header: 4-pin, pull-ups (2 GPIO)
├── Buzzer: NPN driver + flyback diode (1 PWM GPIO)
├── Status LEDs: power LED + 2× GPIO LEDs (2 GPIO)
└── Passives: 0805, JLCPCB basic parts preferred
```

### 1.3 Component Selection Engine

For each need:
1. Query component database (offline JLCPCB parts cache or API)
2. Filter by: category, package, basic/extended, stock
3. Validate: voltage/current ratings, temperature, pin compatibility
4. Resolve conflicts: pin budget, power budget, physical size
5. Produce warnings and recommendations with rationale

### 1.4 Outputs

**requirements.json** — Complete structured document: all components with
values/footprints/LCSC numbers, all pin assignments, all nets, all subcircuits,
power tree, mechanical constraints, AI recommendations log.

**pin_assignment.json** — Dedicated MCU pin map with function, net, and notes
per pin. Tracks unassigned GPIO for future expansion.

**component_selections.json** — Every part with LCSC number, price, stock
status, and alternative parts.

**power_budget.json** — Current consumption per rail, thermal dissipation
calculations, regulator margin analysis.

### 1.5 GitHub Actions

- Initialize repo with full directory structure and templates
- Commit all requirements files
- Create GitHub Project board (Kanban: Backlog → Requirements → Schematic → PCB → Validate → Released)
- Tag: v0.1.0-requirements

---

## PHASE 2: SCHEMATIC GENERATION

### 2.1 Purpose

Transform validated requirements into a complete, electrically correct KiCad
schematic (.kicad_sch).

### 2.2 Symbol Resolution

1. Check built-in library (Device:R, Device:C, Device:LED, power symbols, etc.)
2. Check custom symbol cache
3. Auto-generate rectangular box symbols from pin definitions for ICs
   - Pins placed by function: inputs left, outputs right, power top, GND bottom
4. Optional: import from KiCad's installed library files

### 2.3 Subcircuit Templates

Standard templates instantiated per requirements:

- **Voltage Divider**: VIN → R_top → junction → R_bot → GND, with optional RC filter and TVS clamp. Auto-calculate from E24/E96 series.
- **Decoupling Network**: 100nF per VCC pin (close) + 10uF bulk per IC
- **Reset/Boot Circuit**: pull-up + filter cap + optional button
- **USB-C Input**: CC resistors, ESD protection, fuse
- **LED Drive**: current limit resistor calculation from (Vgpio - Vf) / I_target
- **Buzzer Drive**: NPN transistor + base resistor + flyback diode
- **LDO Circuit**: input caps + output caps + enable pull-up

### 2.4 Schematic Layout

Organized by functional blocks in defined regions:
```
┌────────────────────────────────────────────────┐
│  POWER (top-left)     │  MCU (center)          │
│  USB-C → LDO → rails  │  ESP32 + boot/reset    │
├────────────────────────┼────────────────────────┤
│  ETHERNET (bot-left)   │  ANALOG (bot-right)    │
│  W5500 + RJ45          │  4× divider + filter   │
├────────────────────────┴────────────────────────┤
│  PERIPHERALS: buzzer, LEDs, OLED header, etc.   │
└─────────────────────────────────────────────────┘
```

Global labels connect between blocks. Wires within blocks.

### 2.5 ERC Validation

Run before output:
- Unconnected pins (every non-NC pin must have a net)
- Power net conflicts (no output-output on same net)
- Missing power connections (every IC needs VCC+GND)
- Duplicate references
- Missing decoupling capacitors
- Pin type mismatches

Report as JSON with errors, warnings, info. Each error → GitHub Issue.

### 2.6 GitHub Actions

- Commit schematic + ERC report
- Auto-create issues for ERC failures (labels: bug, schematic, erc)
- Tag: v0.2.0-schematic

---

## PHASE 3: PCB DESIGN RULES & LAYOUT

### 3.1 Purpose

AI analyzes schematic and constraints to recommend PCB specs, then places
components with intelligent positioning.

### 3.2 AI Design Rule Recommendations

AI produces recommendations for:

**Layer Count**: 2 vs 4 based on complexity, signal integrity needs, board size.
Analysis considers component count, signal types, and cost.

**Trace Widths by Net Class**:
- Default (signals): 0.25mm
- Power (+5V, +3V3, GND): 0.5mm — calculated from current capacity
- USB (differential pair): 0.3mm — impedance-matched
- Analog: 0.2mm with extra clearance — crosstalk reduction

**Board Dimensions**: derived from enclosure constraints minus clearance.

**Manufacturing Constraints (JLCPCB-specific)**:
- Min trace/space: 0.127mm (but recommend 0.2mm for reliability)
- Min via drill: 0.3mm, min annular ring: 0.13mm
- Min silkscreen width: 0.153mm
- Board outline clearance: 0.3mm
- Standard drill chart rounding

**Silkscreen & Labeling Plan**:
- Component refs (values hidden for cleanliness)
- Pin labels near connectors
- Functional zone labels ("POWER", "ETHERNET", "ANALOG IN")
- Board title, version, date
- Polarity marks, pin-1 indicators

**Mounting Strategy**: hole positions, drill sizes, pad sizes, enclosure fit.

**Thermal Management**: identify high-dissipation components, recommend thermal
vias, copper pour area requirements.

### 3.3 Component Placement

Zone-based placement:
```
┌───────────────────────────────────────────────────┐
│ ○  USB-C │ POWER (LDO+caps) │ STATUS LEDs     ○  │
│──────────┤                   ├─────────────────────│
│          │   ESP32-S3-WROOM (center)              │
│          │   (decoupling caps tucked beside)       │
│──────────┤                   ├─────────────────────│
│ ETHERNET │  W5500 + caps     │  ANALOG dividers   │
│──────────┤                   ├─────────────────────│
│ RJ45     │  OLED HDR  │ ANALOG HDR  │  BUZZER    │
│ ○        │            │             │          ○  │
└───────────────────────────────────────────────────┘
```

Rules: connectors on edges, MCU central, decoupling within 3mm of IC pins,
analog isolated from switching noise, antenna keepout zone clear.

### 3.4 Copper Zones & Keep-outs

- GND pour on both F.Cu and B.Cu (full board, priority 0)
- Thermal relief: 0.3mm gap, 0.5mm bridge
- Antenna keepout zone: no copper on any layer under ESP32 antenna
- Edge clearance from board outline

### 3.5 GitHub Actions

- Commit placed PCB + design rules report
- Tag: v0.3.0-pcb-layout

---

## PHASE 4: AUTOROUTING

### 4.1 Purpose

Route all copper traces between pads according to netlist, respecting design
rules, signal integrity, and manufacturing constraints.

### 4.2 Routing Strategy — Multi-Pass

**Pass 1 — Power Distribution**: widest traces, star topology from regulator
to consumers. Power vias where layer changes needed.

**Pass 2 — Critical Signal Pairs**: USB differential (matched length, controlled
spacing), SPI bus (grouped, minimize stubs).

**Pass 3 — Analog Signals**: shortest possible, extra clearance from digital,
optional guard traces.

**Pass 4 — General Signals**: GPIO, LEDs, buzzer. Standard width.

**Pass 5 — Ground**: mostly handled by pour. Explicit traces only where pour
can't reach.

### 4.3 Routing Algorithms

**Built-in Grid Router**:
- A* pathfinding on 0.05mm grid, 2 layers
- Cost function: Manhattan distance + layer_change_penalty + congestion
- Via insertion at layer transitions
- Rip-up and retry (max 3 iterations per blocked net)
- Post-processing: remove unnecessary vias, straighten traces, optimize corners

**FreeRouting Integration** (recommended for complex boards):
1. Export .kicad_pcb → .dsn (Specctra Design format)
2. Run: `java -jar freerouting.jar -de input.dsn -do output.ses -mp 20`
3. Import .ses → update .kicad_pcb with routed traces
4. Post-process and clean up

**Configuration**:
```json
{
  "routing_config": {
    "engine": "builtin | freerouting | kicad_scripting",
    "allow_45_degree": true,
    "allow_via_in_pad": false,
    "preferred_direction": {"F.Cu": "horizontal", "B.Cu": "vertical"},
    "via_cost_multiplier": 10,
    "max_ripup_iterations": 3
  }
}
```

### 4.4 Routing Report

Reports: total/routed/unrouted nets, via count, total trace length, layer
utilization, routing time. Unrouted nets include reason and suggestion.

### 4.5 GitHub Actions

- Commit routed PCB + routing report
- Auto-create issues for unrouted nets (labels: routing, manual-fix)
- Tag: v0.4.0-pcb-routed

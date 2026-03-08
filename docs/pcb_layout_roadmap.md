# PCB Layout Roadmap

Planned improvements to the automated PCB placement and layout engine.

## Current State (Sprint 16+)

- Constraint-based placement solver with occupancy grid
- Board template support (RPi HAT, generic rectangular)
- Edge connector placement (screw terminals, pin headers)
- Dynamic zone sizing based on component footprint areas
- Signal adjacency analysis for component grouping
- Decoupling cap identification and IC association
- Rotation optimization for passives
- Courtyard collision detection
- GND zone auto-generation (front/back strategies)

## Near-Term Improvements

### Subcircuit-Aware Placement
- Group decoupling caps adjacent to their IC power pins (implemented)
- Group passives with their nearest IC/connector by net adjacency (implemented)
- Relay isolation zones — dedicated placement zone for relay clusters (implemented)
- Power section grouping — regulators with their input/output caps as a unit

### Edge-Sensitive Component Handling
- WiFi/BLE modules placed at board edge with antenna facing outward (implemented)
- USB, RJ45, SD card connectors at board edges (implemented)
- Dependent passives kept near their edge-placed parent

### Design Review Generation
- Automated checklist of manual tasks the framework can't automate (implemented)
- Antenna keepout zone reminders
- Relay isolation slot recommendations
- High-current trace width warnings
- Decoupling cap placement verification
- Zone fill reminders

## Medium-Term Goals

### Automatic Decoupling Cap Placement
- Place caps within 3mm of IC power pins automatically
- Multiple caps per IC: smallest closest, bulk further away
- Via placement for shortest path to GND plane

### Antenna Keepout Zone Generation
- Auto-generate copper keepout zones around WiFi/BLE antenna areas
- 5mm clearance from antenna to nearest copper pour
- Ground plane cutout under antenna matching module datasheet

### Relay Isolation Slot Generation
- Board cutout slots between relay contact traces and digital logic
- Minimum 1mm slot width, positioned along relay boundary

### Power Plane Partitioning
- Mixed-voltage designs: separate copper zones for each voltage rail
- Star topology for ground connections (analog vs digital GND)
- Automatic thermal relief pattern for power pads

### Courtyard-Aware Placement
- Hard constraint: no footprint courtyard overlaps
- Soft constraint: minimum 0.5mm between courtyards for rework access
- Back-side component placement for space-constrained boards

## Long-Term Vision

### Signal-Flow Placement Optimization
- Input connectors on left/top, output on right/bottom
- Processing ICs in middle following signal chain
- Minimize total trace length via iterative optimization

### Thermal Analysis Integration
- Identify hot components (regulators, power FETs, LEDs)
- Place heat sources away from temperature-sensitive components (ADCs, crystals)
- Thermal via array sizing based on power dissipation

### DFM Checks (JLCPCB-Specific)
- Minimum trace/space for selected PCB technology tier
- Drill size validation against JLCPCB drill chart
- Solder paste stencil aperture ratio checks
- Panelization-aware edge clearances

### Interactive Placement via KiCad IPC API
- Real-time push/pull of component positions with KiCad 9
- User drags component in KiCad, pipeline updates constraints
- Pipeline suggests placement, KiCad previews in real-time

### Multi-Board Support
- Daughter board/mother board placement coordination
- Connector alignment across boards
- Stacking height validation

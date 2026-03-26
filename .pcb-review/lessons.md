# PCB Review Lessons Learned

## Board-Specific Notes
- Relay: 39/39 compliance, 3D alignment fixed, production-ready pattern
- Power: J2/J3 pushed to bottom by ordering phase, constraint guard added but proximity not sticking
- MCU: Antenna keepout now extends to board edge, via fence refreshed, decoupling on correct side
- Analog: Screw terminal rotation fixed (180 for top edge), channel strips organized
- Ethernet: RJ45 intentionally overhangs edge, U1/J1 gap fixed

## What Works
- Removing pin-1 3D offset (use 0,0,0 for JLCPCB footprints)
- Board-level antenna keepout + via fence refresh after optimizer
- Constraint guard in phases that move components
- Ordering before proximity in constraint phase
- 4-view rendering (2D + 3D-top + 3D-iso + 3D-iso-back)

## What Doesn't Work
- Pin-1 based 3D model offset (shifts body to wrong position)
- Footprint-level keepout zones (rotate to wrong side with component)
- Single iso angle for 3D review (hides alignment issues)
- Trusting compliance scores without visual verification
- Coordinate math reasoning (always verify against actual PCB file)

## Pin Header 3D Appearance
Pin header bodies (J2, J3, J5, etc.) naturally sit 2-3mm above the PCB surface. This is correct — THT pin headers have plastic bodies on top and pins going through the board. Do NOT flag this as a "floating" issue. The body-to-board gap is by design.

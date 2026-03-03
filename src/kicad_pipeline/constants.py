"""Project-wide constants for the kicad-ai-pipeline.

All physical values are in millimetres unless the name explicitly states
otherwise (e.g. ``_MA``, ``_MW``, ``_OHMS``, ``_UF``).

Coordinate system
-----------------
Both schematic and PCB share the same convention: origin at the top-left
corner, X increases to the right, Y increases downward.
"""

# ---------------------------------------------------------------------------
# KiCad format identifiers
# ---------------------------------------------------------------------------

KICAD_SCH_VERSION: int = 20231120
"""KiCad schematic file format version integer."""

KICAD_PCB_VERSION: int = 20231120
"""KiCad PCB file format version integer."""

KICAD_GENERATOR: str = "kicad-ai-pipeline"
"""String written into the ``generator`` field of generated KiCad files."""

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

MM_PER_MIL: float = 0.0254
"""Millimetres per mil (thousandth of an inch)."""

MIL_PER_MM: float = 39.3701
"""Mils per millimetre."""

# ---------------------------------------------------------------------------
# JLCPCB manufacturing constraints
# ---------------------------------------------------------------------------

JLCPCB_MIN_TRACE_MM: float = 0.127
"""Minimum copper trace width supported by JLCPCB (mm)."""

JLCPCB_RECOMMENDED_TRACE_MM: float = 0.2
"""JLCPCB recommended minimum trace width for reliable production (mm)."""

JLCPCB_MIN_CLEARANCE_MM: float = 0.127
"""Minimum copper-to-copper clearance supported by JLCPCB (mm)."""

JLCPCB_RECOMMENDED_CLEARANCE_MM: float = 0.2
"""JLCPCB recommended minimum clearance for reliable production (mm)."""

JLCPCB_MIN_VIA_DRILL_MM: float = 0.3
"""Minimum via drill diameter supported by JLCPCB (mm)."""

JLCPCB_MIN_VIA_ANNULAR_RING_MM: float = 0.13
"""Minimum annular ring width for vias supported by JLCPCB (mm)."""

JLCPCB_MIN_DRILL_MM: float = 0.2
"""Minimum through-hole drill diameter supported by JLCPCB (mm)."""

JLCPCB_MAX_DRILL_MM: float = 6.3
"""Maximum through-hole drill diameter supported by JLCPCB (mm)."""

JLCPCB_MIN_ANNULAR_RING_MM: float = 0.13
"""Minimum annular ring width for through-holes supported by JLCPCB (mm)."""

JLCPCB_MIN_SILK_WIDTH_MM: float = 0.153
"""Minimum silkscreen line width supported by JLCPCB (mm)."""

JLCPCB_BOARD_EDGE_CLEARANCE_MM: float = 0.3
"""Minimum clearance from copper features to the board edge (mm)."""

JLCPCB_MIN_BOARD_SIZE_MM: tuple[float, float] = (10.0, 10.0)
"""Minimum board dimensions (width_mm, height_mm) accepted by JLCPCB."""

JLCPCB_MAX_BOARD_SIZE_MM: tuple[float, float] = (500.0, 500.0)
"""Maximum board dimensions (width_mm, height_mm) accepted by JLCPCB."""

# ---------------------------------------------------------------------------
# Net-class trace widths
# ---------------------------------------------------------------------------

TRACE_WIDTH_DEFAULT_MM: float = 0.25
"""Default copper trace width for general-purpose nets (mm)."""

TRACE_WIDTH_POWER_MM: float = 0.5
"""Trace width for power-rail nets (mm)."""

TRACE_WIDTH_USB_DIFF_MM: float = 0.3
"""Trace width for USB differential-pair nets (mm)."""

TRACE_WIDTH_ANALOG_MM: float = 0.2
"""Trace width for sensitive analogue signal nets (mm)."""

CLEARANCE_DEFAULT_MM: float = 0.2
"""Default copper clearance for general-purpose nets (mm)."""

CLEARANCE_POWER_MM: float = 0.3
"""Copper clearance for power-rail nets (mm)."""

CLEARANCE_USB_MM: float = 0.2
"""Copper clearance for USB nets (mm)."""

CLEARANCE_ANALOG_MM: float = 0.25
"""Copper clearance for analogue signal nets (mm)."""

# ---------------------------------------------------------------------------
# Via defaults
# ---------------------------------------------------------------------------

VIA_DRILL_DEFAULT_MM: float = 0.4
"""Default via drill diameter (mm)."""

VIA_DIAMETER_DEFAULT_MM: float = 0.8
"""Default via pad diameter (mm)."""

THERMAL_RELIEF_GAP_MM: float = 0.3
"""Gap between the copper fill and the thermal-relief spokes (mm)."""

THERMAL_RELIEF_BRIDGE_MM: float = 0.5
"""Width of each thermal-relief spoke connecting pad to copper fill (mm)."""

# ---------------------------------------------------------------------------
# Schematic layout defaults
# ---------------------------------------------------------------------------

SCHEMATIC_PIN_LENGTH_MM: float = 2.54
"""Default symbol pin length in schematic (mm)."""

SCHEMATIC_SYMBOL_PIN_SPACING_MM: float = 2.54
"""Vertical spacing between adjacent pins on a symbol body (mm)."""

SCHEMATIC_WIRE_GRID_MM: float = 1.27
"""Schematic wire routing grid size (mm)."""

SCHEMATIC_TEXT_SIZE_MM: float = 1.27
"""Default text size for annotations in the schematic (mm)."""

SCHEMATIC_LABEL_SIZE_MM: float = 1.27
"""Default net-label text size in the schematic (mm)."""

SCHEMATIC_REF_SIZE_MM: float = 1.27
"""Default reference-designator text size in the schematic (mm)."""

SCHEMATIC_VALUE_SIZE_MM: float = 1.27
"""Default component-value text size in the schematic (mm)."""

# ---------------------------------------------------------------------------
# PCB layout defaults
# ---------------------------------------------------------------------------

PCB_COURTYARD_CLEARANCE_MM: float = 0.25
"""Minimum clearance between a footprint courtyard and other features (mm)."""

PCB_SILKSCREEN_LINE_WIDTH_MM: float = 0.153
"""Default silkscreen line width on the PCB (mm)."""

PCB_EDGE_CUTS_WIDTH_MM: float = 0.05
"""Line width used for the board-outline Edge.Cuts layer (mm)."""

DECOUPLING_CAP_MAX_DISTANCE_MM: float = 3.0
"""Maximum allowed distance from a decoupling capacitor to its IC supply pin (mm)."""

# ---------------------------------------------------------------------------
# Power budget
# ---------------------------------------------------------------------------

THERMAL_WARNING_MW: float = 500.0
"""Flag any component dissipating more than this value (mW)."""

# ---------------------------------------------------------------------------
# Component defaults
# ---------------------------------------------------------------------------

LED_TARGET_CURRENT_MA: float = 10.0
"""Target forward current used when calculating LED current-limiting resistors (mA)."""

DEFAULT_PULLUP_RESISTANCE_OHMS: float = 10000.0
"""Default pull-up resistor value for I2C and general-purpose signals (ohms)."""

DEFAULT_DECOUPLING_VALUE_UF: float = 0.1
"""Default high-frequency decoupling capacitor value (µF)."""

DEFAULT_BULK_DECOUPLING_VALUE_UF: float = 10.0
"""Default bulk bypass capacitor value placed at power-rail entry points (µF)."""

# ---------------------------------------------------------------------------
# Layer names (KiCad convention)
# ---------------------------------------------------------------------------

LAYER_F_CU: str = "F.Cu"
"""Front copper layer."""

LAYER_B_CU: str = "B.Cu"
"""Back copper layer."""

LAYER_F_SILKSCREEN: str = "F.Silkscreen"
"""Front silkscreen layer."""

LAYER_B_SILKSCREEN: str = "B.Silkscreen"
"""Back silkscreen layer."""

LAYER_F_PASTE: str = "F.Paste"
"""Front solder-paste layer."""

LAYER_B_PASTE: str = "B.Paste"
"""Back solder-paste layer."""

LAYER_F_MASK: str = "F.Mask"
"""Front solder-mask layer."""

LAYER_B_MASK: str = "B.Mask"
"""Back solder-mask layer."""

LAYER_F_COURTYARD: str = "F.Courtyard"
"""Front courtyard layer."""

LAYER_B_COURTYARD: str = "B.Courtyard"
"""Back courtyard layer."""

LAYER_F_FAB: str = "F.Fab"
"""Front fabrication-aid layer."""

LAYER_B_FAB: str = "B.Fab"
"""Back fabrication-aid layer."""

LAYER_EDGE_CUTS: str = "Edge.Cuts"
"""Board-outline layer."""

LAYER_IN1_CU: str = "In1.Cu"
"""First inner copper layer (4-layer stackup)."""

LAYER_IN2_CU: str = "In2.Cu"
"""Second inner copper layer (4-layer stackup)."""

LAYER_DWGS_USER: str = "Dwgs.User"
"""User drawings layer for non-manufacturing annotations."""

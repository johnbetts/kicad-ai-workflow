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

KICAD_SCH_VERSION: int = 20250114
"""KiCad schematic file format version integer (KiCad 10 uses same as KiCad 9)."""

KICAD_PCB_VERSION: int = 20260206
"""KiCad PCB file format version integer (KiCad 10)."""

KICAD_GENERATOR: str = "kicad-ai-pipeline"
"""String written into the ``generator`` field of generated KiCad files."""

KICAD_GENERATOR_VERSION: str = "10.0"
"""Generator version string written into generated KiCad files."""

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
# LCSC / JLCPCB parts search
# ---------------------------------------------------------------------------

LCSC_API_BASE_URL: str = "https://wmsc.lcsc.com/ftps/wm/product/detail"
"""LCSC product detail API endpoint (public, no auth)."""

LCSC_STOCK_TIMEOUT_SECONDS: float = 10.0
"""Default timeout for LCSC HTTP requests (seconds)."""

JLCPCB_PARTS_SEARCH_URL: str = "https://jlcpcb.com/parts/componentSearch?searchTxt="
"""JLCPCB parts search URL prefix for manual part lookup."""

JLCPCB_MIN_STOCK_QTY: int = 1000
"""Minimum LCSC stock quantity to accept a part without approval.

Parts with fewer than this many units in stock are flagged as low-stock
and treated as unavailable, since JLCPCB assembly may deplete stock
mid-order or the part may be nearing end-of-life.
"""

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

VIA_DRILL_DEFAULT_MM: float = 0.508
"""Default via drill diameter (mm)."""

VIA_DIAMETER_DEFAULT_MM: float = 0.9
"""Default via pad diameter (mm)."""

VIA_DRILL_SIGNAL_MM: float = 0.3
"""Signal-routing via drill diameter — JLCPCB minimum (mm)."""

VIA_DIAMETER_SIGNAL_MM: float = 0.6
"""Signal-routing via pad diameter — 0.3 drill + 2x0.15 annular ring (mm)."""

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

SCHEMATIC_SYMBOL_GAP_MM: float = 5.08
"""Minimum gap between adjacent symbol extents on the schematic (mm)."""

SCHEMATIC_LABEL_CHAR_WIDTH_MM: float = 1.0
"""Approximate character width for ref/value label text at default size (mm)."""

SCHEMATIC_MAX_LABEL_CHARS: int = 10
"""Minimum label width estimate (characters) for spacing calculations."""

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

DECOUPLING_CAP_MIN_DISTANCE_MM: float = 1.0
"""Minimum distance from a decoupling capacitor to its IC (mm)."""

PASSIVE_NEAR_IC_MAX_DISTANCE_MM: float = 5.0
"""Maximum placement distance from a passive to its associated IC (mm)."""

SUBCIRCUIT_MAX_SPREAD_MM: float = 15.0
"""Maximum spread of sub-circuit components from anchor (mm)."""

VOLTAGE_DOMAIN_MIN_GAP_MM: float = 2.0
"""Minimum gap between components in different voltage domains (mm)."""

CONNECTOR_EDGE_MAX_MM: float = 5.0
"""Maximum distance from a connector to the nearest board edge (mm)."""

CRYSTAL_MAX_DISTANCE_MM: float = 5.0
"""Maximum distance from a crystal to its MCU clock pins (mm)."""

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

LAYER_F_SILKSCREEN: str = "F.SilkS"
"""Front silkscreen layer (canonical KiCad file-format name)."""

LAYER_B_SILKSCREEN: str = "B.SilkS"
"""Back silkscreen layer (canonical KiCad file-format name)."""

LAYER_F_PASTE: str = "F.Paste"
"""Front solder-paste layer."""

LAYER_B_PASTE: str = "B.Paste"
"""Back solder-paste layer."""

LAYER_F_MASK: str = "F.Mask"
"""Front solder-mask layer."""

LAYER_B_MASK: str = "B.Mask"
"""Back solder-mask layer."""

LAYER_F_COURTYARD: str = "F.CrtYd"
"""Front courtyard layer (canonical KiCad file-format name)."""

LAYER_B_COURTYARD: str = "B.CrtYd"
"""Back courtyard layer (canonical KiCad file-format name)."""

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

# ---------------------------------------------------------------------------
# Hierarchical schematic defaults
# ---------------------------------------------------------------------------

HIERARCHICAL_MIN_FEATURES: int = 2
"""Minimum number of FeatureBlocks to auto-enable hierarchical schematics."""

HIERARCHICAL_MIN_COMPONENTS: int = 4
"""Minimum total component count to auto-enable hierarchical schematics."""

SHEET_SYMBOL_MIN_WIDTH_MM: float = 25.0
"""Minimum width of a sheet symbol on the root schematic (mm)."""

SHEET_SYMBOL_MIN_HEIGHT_MM: float = 15.0
"""Minimum height of a sheet symbol on the root schematic (mm)."""

SHEET_SYMBOL_PIN_SPACING_MM: float = 2.54
"""Vertical spacing between pins on a sheet symbol (mm)."""

SHEET_SYMBOL_GRID_MARGIN_MM: float = 15.0
"""Margin between sheet symbols on the root schematic grid (mm)."""

# ---------------------------------------------------------------------------
# Zone defaults
# ---------------------------------------------------------------------------

ZONE_CLEARANCE_DEFAULT_MM: float = 0.3
"""Default copper zone clearance to pads/tracks (mm)."""

ZONE_MIN_THICKNESS_MM: float = 0.25
"""Default minimum zone fill thickness (mm)."""

RF_VIA_FENCE_SPACING_MM: float = 2.0
"""Target via-to-via spacing along an RF via fence perimeter (mm)."""

GND_STITCH_SPACING_MM: float = 15.0
"""Target spacing for GND stitching vias (mm), midpoint of 10-20mm range."""

GND_STITCH_FP_CLEARANCE_MM: float = 2.0
"""Minimum clearance from GND stitching vias to footprint bounding boxes (mm)."""

# ---------------------------------------------------------------------------
# 3D model paths
# ---------------------------------------------------------------------------

KICAD_3DMODEL_VAR: str = "${KICAD10_3DMODEL_DIR}"
"""KiCad environment variable prefix for 3D model file paths."""

# ---------------------------------------------------------------------------
# Routing cost tuning
# ---------------------------------------------------------------------------

ROUTING_VIA_COST: float = 16.0
"""Penalty per via transition (16x a single grid step).

Spec mandates 14-20x normal trace segment; cost function uses 16x.
Higher values bias the router toward single-layer traces, using vias only
when the F.Cu detour would be longer than the via penalty.
"""

ROUTING_BEND_PENALTY: float = 0.3
"""Extra cost added on each direction change during A* routing.

Discourages staircase / zigzag paths, producing cleaner traces.
"""

ROUTING_CONGESTION_MAX: float = 4.0
"""Maximum cost multiplier applied near existing routed tracks.

Biases later nets away from congested areas, improving overall routability.
"""

# ---------------------------------------------------------------------------
# Voltage-based clearance thresholds (IPC-2221 simplified)
# ---------------------------------------------------------------------------

VOLTAGE_CLEARANCE_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (50.0, 0.2),
    (100.0, 0.5),
    (250.0, 1.0),
    (500.0, 2.5),
)
"""Voltage-to-clearance mapping: (max_voltage_V, min_clearance_mm).

The first entry whose voltage >= the net voltage determines clearance.
Based on IPC-2221 simplified external-layer distances.
"""

# ---------------------------------------------------------------------------
# Multi-agent coordination
# ---------------------------------------------------------------------------

AGENT_REGISTRY_DIR: str = "~/.claude/kicad-agents"
"""Default directory for the agent registry and per-agent state files."""

AGENT_STATUS_FILENAME: str = "status.json"
"""Filename for a board agent's status file within its agent directory."""

AGENT_COMMANDS_FILENAME: str = "commands.json"
"""Filename for commands issued to a board agent."""

AGENT_REGISTRY_FILENAME: str = "registry.json"
"""Filename for the global agent registry."""

AGENT_PIPELINE_VERSION_FILENAME: str = "pipeline-version.json"
"""Filename for the pipeline version marker."""

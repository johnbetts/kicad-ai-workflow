"""Golden cases — the 5 training boards as formal eval cases.

Each training board script (``scripts/train_*.py``) has a
``_build_requirements()`` function that returns a fully-specified
``ProjectRequirements``.  This module wraps those into ``EvalCase``
instances with explicit hard gates and soft targets.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.evals.models import EvalCase, HardGate, SoftTarget

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements

# ---------------------------------------------------------------------------
# Locate the scripts/ directory relative to the package
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def _import_train_script(module_name: str) -> object:
    """Import a training script module by name.

    Adds scripts/ and src/ to sys.path if not already present, mimicking
    how the scripts themselves set up their import environment.
    """
    src_dir = str(_REPO_ROOT / "src")
    scripts_dir = str(_SCRIPTS_DIR)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Requirement builder closures
# ---------------------------------------------------------------------------

def _mcu_core_requirements() -> ProjectRequirements:
    mod = _import_train_script("train_mcu_core")
    return mod._build_requirements()  # type: ignore[attr-defined]


def _power_chain_requirements() -> ProjectRequirements:
    mod = _import_train_script("train_power_chain")
    return mod._build_requirements()  # type: ignore[attr-defined]


def _relay_group_requirements() -> ProjectRequirements:
    mod = _import_train_script("train_relay_group")
    return mod._build_requirements()  # type: ignore[attr-defined]


def _analog_input_requirements() -> ProjectRequirements:
    mod = _import_train_script("train_analog_input")
    return mod._build_requirements()  # type: ignore[attr-defined]


def _ethernet_requirements() -> ProjectRequirements:
    mod = _import_train_script("train_ethernet")
    return mod._build_requirements()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Standard hard gates — applied to all golden cases
# ---------------------------------------------------------------------------

STANDARD_HARD_GATES: tuple[HardGate, ...] = (
    # Original gates
    HardGate("build_succeeds", "build_pcb + optimize_placement_ee complete without exception"),
    HardGate("zero_critical_integrity", "No critical integrity issues"),
    HardGate("all_components_placed", "Every requirement component exists in PCB"),
    HardGate("all_on_board", "No component center outside board outline + 2mm margin"),
    HardGate("renders_generated", "2D + 3D renders (4 views) generated via kicad-image-gen"),
    # DFM gates — defect taxonomy #1,3,4,6,7,8,9,10,11
    HardGate("footprint_registry_match", "Pad count/type consistent per footprint type (#1)"),
    HardGate("no_collisions", "No courtyard overlaps between components (#3)"),
    HardGate("all_pads_within_board", "All component pads within board outline (#3)"),
    HardGate("mounting_hole_clearance", "No components over mounting holes (#3)"),
    HardGate("decoupling_proximity", "Decoupling caps within threshold of ICs (#6)"),
    HardGate("subcircuit_spread", "Subcircuit components within spread limits (#6)"),
    HardGate("zone_membership", "Components placed in assigned functional zones (#7)"),
    HardGate("package_match", "PCB footprint matches requirements package spec (#8)"),
    HardGate("subcircuit_completeness", "ICs have required companion components (#9)"),
    HardGate("schematic_pcb_sync", "All requirements components present in PCB (#10)"),
    HardGate("component_isolation_zones", "RF modules have antenna keepout zones (#11)"),
    HardGate("board_sizing", "Board size appropriate for component count (#4)"),
)


# ---------------------------------------------------------------------------
# Golden case definitions
# ---------------------------------------------------------------------------

def _mcu_core_case() -> EvalCase:
    return EvalCase(
        case_id="mcu_core",
        board_name="MCU Core (ESP32-S3)",
        description="ESP32-S3-WROOM-1 with decoupling, USB-C, buttons, LED, UART header",
        build_fn=_mcu_core_requirements,
        board_width_mm=70.0,
        board_height_mm=50.0,
        hard_gates=STANDARD_HARD_GATES,
        soft_targets=(
            SoftTarget("overall_score", min_value=0.70),
            SoftTarget("Collisions", min_value=0.50),
            SoftTarget("Voltage Isolation", min_value=0.60),
        ),
        tags=("golden", "mcu"),
    )


def _power_chain_case() -> EvalCase:
    return EvalCase(
        case_id="power_chain",
        board_name="Power Supply Chain",
        description="Buck converters and LDO regulators with input/output filtering",
        build_fn=_power_chain_requirements,
        board_width_mm=60.0,
        board_height_mm=40.0,
        hard_gates=STANDARD_HARD_GATES,
        soft_targets=(
            SoftTarget("overall_score", min_value=0.68),
            SoftTarget("Decoupling Proximity", min_value=0.50),
        ),
        tags=("golden", "power"),
    )


def _relay_group_case() -> EvalCase:
    return EvalCase(
        case_id="relay_group",
        board_name="Relay Output Group",
        description="4-channel relay drivers with flyback diodes and terminal blocks",
        build_fn=_relay_group_requirements,
        board_width_mm=100.0,
        board_height_mm=60.0,
        hard_gates=STANDARD_HARD_GATES,
        soft_targets=(
            SoftTarget("overall_score", min_value=0.65),
            SoftTarget("Group Cohesion", min_value=0.35),
        ),
        tags=("golden", "relay"),
    )


def _analog_input_case() -> EvalCase:
    return EvalCase(
        case_id="analog_input",
        board_name="Analog Input Section",
        description="ADC channels with voltage dividers, clamping, and filtering",
        build_fn=_analog_input_requirements,
        board_width_mm=65.0,
        board_height_mm=40.0,
        hard_gates=STANDARD_HARD_GATES,
        soft_targets=(
            SoftTarget("overall_score", min_value=0.65),
            SoftTarget("Signal Flow", min_value=0.40),
        ),
        tags=("golden", "analog"),
    )


def _ethernet_case() -> EvalCase:
    return EvalCase(
        case_id="ethernet",
        board_name="Ethernet + PoE",
        description="Ethernet PHY, magnetics, RJ45 connector, PoE circuitry",
        build_fn=_ethernet_requirements,
        board_width_mm=55.0,
        board_height_mm=40.0,
        hard_gates=STANDARD_HARD_GATES,
        soft_targets=(
            SoftTarget("overall_score", min_value=0.65),
        ),
        tags=("golden", "ethernet"),
    )


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

def all_golden_cases() -> tuple[EvalCase, ...]:
    """Return all 5 golden eval cases."""
    return (
        _mcu_core_case(),
        _power_chain_case(),
        _relay_group_case(),
        _analog_input_case(),
        _ethernet_case(),
    )

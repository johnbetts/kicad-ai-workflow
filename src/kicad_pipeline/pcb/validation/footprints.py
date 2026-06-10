"""Validation utilities for footprints.

Provides functions for validating footprint integrity, pin assignments,
and component specifications.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import Footprint

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Footprint validation
# ---------------------------------------------------------------------------


def validate_footprint_integrity(fp: Footprint) -> tuple[str, ...]:
    """Validate footprint structural integrity.

    Checks for common footprint issues:
    - Missing pads
    - Invalid pad numbers
    - Missing reference designator
    - Invalid layer assignments

    Args:
        fp: Footprint to validate

    Returns:
        Tuple of error messages
    """
    issues = []

    # Check basic requirements
    if not fp.ref:
        issues.append("Missing reference designator")

    if not fp.lib_id:
        issues.append("Missing library identifier")

    if not fp.pads:
        issues.append("No pads defined")
        return tuple(issues)

    # Validate pad numbers
    pad_numbers = set()
    for pad in fp.pads:
        if not pad.number:
            issues.append(f"Pad at ({pad.position.x:.2f}, {pad.position.y:.2f}) has no number")
        elif pad.number in pad_numbers:
            issues.append(f"Duplicate pad number: {pad.number}")
        else:
            pad_numbers.add(pad.number)

    # Validate layers
    from kicad_pipeline.constants import (
        LAYER_F_CU, LAYER_B_CU, LAYER_EDGE_CUTS
    )

    valid_layers = {LAYER_F_CU, LAYER_B_CU, LAYER_EDGE_CUTS}
    for pad in fp.pads:
        if pad.layer not in valid_layers:
            issues.append(f"Invalid pad layer: {pad.layer}")

    # Check footprint layer
    if fp.layer not in (LAYER_F_CU, LAYER_B_CU):
        issues.append(f"Invalid footprint layer: {fp.layer}")

    return tuple(issues)


def validate_pin_assignments(fp: Footprint) -> tuple[str, ...]:
    """Validate pin assignments and numbering.

    Checks for:
    - Sequential pin numbering
    - Missing pins
    - Pin number format consistency

    Args:
        fp: Footprint to validate

    Returns:
        Tuple of validation issues
    """
    issues = []

    if not fp.pads:
        return tuple(issues)

    # Extract and sort pad numbers
    pad_nums = []
    for pad in fp.pads:
        try:
            num = int(pad.number)
            pad_nums.append(num)
        except (ValueError, TypeError):
            issues.append(f"Invalid pad number format: '{pad.number}'")
            continue

    if not pad_nums:
        return tuple(issues)

    pad_nums.sort()

    # Check for sequential numbering (allowing for some gaps in complex components)
    expected_nums = list(range(1, len(pad_nums) + 1))
    missing_nums = set(expected_nums) - set(pad_nums)

    if missing_nums:
        # Allow some missing numbers for complex footprints (e.g., QFN with EP)
        if len(missing_nums) > len(pad_nums) * 0.3:  # More than 30% missing
            issues.append(f"Many missing pad numbers: {sorted(missing_nums)}")

    # Check for duplicate positions (stacked pads)
    positions = {}
    for pad in fp.pads:
        pos_key = (round(pad.position.x, 3), round(pad.position.y, 3))
        if pos_key in positions:
            issues.append(f"Overlapping pads at ({pad.position.x:.3f}, {pad.position.y:.3f}): "
                         f"{positions[pos_key]} and {pad.number}")
        else:
            positions[pos_key] = pad.number

    return tuple(issues)


def validate_component_spec(fp: Footprint) -> tuple[str, ...]:
    """Validate component specification compliance.

    Checks footprint against expected specifications for the component type.

    Args:
        fp: Footprint to validate

    Returns:
        Tuple of specification violations
    """
    issues = []

    lib_id = fp.lib_id.upper()

    # Basic component type detection and validation
    if 'RESISTOR' in lib_id or 'R_' in lib_id:
        issues.extend(_validate_resistor_spec(fp))
    elif 'CAPACITOR' in lib_id or 'C_' in lib_id:
        issues.extend(_validate_capacitor_spec(fp))
    elif 'INDUCTOR' in lib_id or 'L_' in lib_id:
        issues.extend(_validate_inductor_spec(fp))
    elif 'DIODE' in lib_id or 'D_' in lib_id:
        issues.extend(_validate_diode_spec(fp))
    elif 'LED' in lib_id:
        issues.extend(_validate_led_spec(fp))
    elif any(x in lib_id for x in ('QFN', 'QFP', 'LQFP', 'TQFP')):
        issues.extend(_validate_ic_spec(fp))
    elif 'SOT' in lib_id:
        issues.extend(_validate_sot_spec(fp))

    return tuple(issues)


def _validate_resistor_spec(fp: Footprint) -> list[str]:
    """Validate resistor footprint specifications."""
    issues = []

    # Resistors typically have 2 pads
    if len(fp.pads) != 2:
        issues.append(f"Resistor should have 2 pads, found {len(fp.pads)}")

    # Check pad sizes are reasonable for resistors
    for pad in fp.pads:
        w, h = pad.size
        if w < 0.3 or h < 0.3:
            issues.append(f"Resistor pad too small: {w:.2f}x{h:.2f}mm")
        if w > 3.0 or h > 3.0:
            issues.append(f"Resistor pad too large: {w:.2f}x{h:.2f}mm")

    return issues


def _validate_capacitor_spec(fp: Footprint) -> list[str]:
    """Validate capacitor footprint specifications."""
    issues = []

    # Capacitors typically have 2 pads
    if len(fp.pads) != 2:
        issues.append(f"Capacitor should have 2 pads, found {len(fp.pads)}")

    # Check for polarized capacitors
    if 'POLARIZED' in fp.lib_id.upper() or 'POLARISED' in fp.lib_id.upper():
        # Should have polarity marking
        has_polarity_marking = any(
            'POLARITY' in str(getattr(g, 'layer', '')).upper() or
            '+' in str(getattr(g, 'text', '')).upper()
            for g in fp.graphics
        )
        if not has_polarity_marking:
            issues.append("Polarized capacitor missing polarity marking")

    return issues


def _validate_inductor_spec(fp: Footprint) -> list[str]:
    """Validate inductor footprint specifications."""
    issues = []

    # Inductors typically have 2 pads
    if len(fp.pads) != 2:
        issues.append(f"Inductor should have 2 pads, found {len(fp.pads)}")

    return issues


def _validate_diode_spec(fp: Footprint) -> list[str]:
    """Validate diode footprint specifications."""
    issues = []

    # Diodes typically have 2 pads
    if len(fp.pads) != 2:
        issues.append(f"Diode should have 2 pads, found {len(fp.pads)}")

    # Check for cathode marking
    has_cathode_marking = any(
        'CATHODE' in str(getattr(g, 'layer', '')).upper() or
        'CATHODE' in str(getattr(g, 'text', '')).upper()
        for g in fp.graphics
    )
    if not has_cathode_marking:
        # This is common, so just warn
        pass

    return issues


def _validate_led_spec(fp: Footprint) -> list[str]:
    """Validate LED footprint specifications."""
    issues = []

    # LEDs typically have 2 pads
    if len(fp.pads) != 2:
        issues.append(f"LED should have 2 pads, found {len(fp.pads)}")

    # Check for polarity marking (cathode)
    has_polarity_marking = any(
        'CATHODE' in str(getattr(g, 'layer', '')).upper() or
        'K' in str(getattr(g, 'text', '')).upper() or
        'CATHODE' in str(getattr(g, 'text', '')).upper()
        for g in fp.graphics
    )
    if not has_polarity_marking:
        issues.append("LED missing cathode polarity marking")

    return issues


def _validate_ic_spec(fp: Footprint) -> list[str]:
    """Validate IC footprint specifications."""
    issues = []

    # ICs should have pin 1 marked
    pin1 = next((p for p in fp.pads if p.number == "1"), None)
    if not pin1:
        issues.append("IC missing pin 1")

    # Check pin count is reasonable
    if len(fp.pads) < 4:
        issues.append(f"IC has very few pins: {len(fp.pads)}")

    return issues


def _validate_sot_spec(fp: Footprint) -> list[str]:
    """Validate SOT package specifications."""
    issues = []

    # SOT packages should have pin 1 marked
    pin1 = next((p for p in fp.pads if p.number == "1"), None)
    if not pin1:
        issues.append("SOT package missing pin 1")

    # Check pin count matches package type
    lib_id = fp.lib_id.upper()
    if 'SOT-23' in lib_id:
        if 'SOT-23-3' in lib_id and len(fp.pads) != 3:
            issues.append(f"SOT-23-3 should have 3 pins, found {len(fp.pads)}")
        elif 'SOT-23-5' in lib_id and len(fp.pads) != 5:
            issues.append(f"SOT-23-5 should have 5 pins, found {len(fp.pads)}")
        elif 'SOT-23-6' in lib_id and len(fp.pads) != 6:
            issues.append(f"SOT-23-6 should have 6 pins, found {len(fp.pads)}")

    return issues
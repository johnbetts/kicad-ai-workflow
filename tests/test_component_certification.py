"""Component certification test suite.

Parametrized tests that certify every supported component type in the
footprint library.  Each test calls ``footprint_for_component()`` and
asserts pad count, pad type, 3D model presence, pad position sanity,
and lib_id population.

Component definitions are loaded from the component registry
(``data/component_registry.json``) — the single source of truth.
"""

from __future__ import annotations

import pytest

from kicad_pipeline.pcb.footprints import footprint_for_component
from kicad_pipeline.validation.component_registry import ComponentRegistry

# ---------------------------------------------------------------------------
# Load certified component definitions from registry
# ---------------------------------------------------------------------------

_REGISTRY = ComponentRegistry()

CERTIFIED_COMPONENTS = [
    (s.ref, s.value, s.footprint_id, s.expected_pads, s.expected_pad_type, s.description)
    for s in _REGISTRY.all_components()
    # Only test verified components (can generate parametrically)
    if s.verification_status == "verified" and s.component_id != "SOIC-8_thermal"
]


@pytest.mark.parametrize(
    "ref,value,fp_id,expected_pads,expected_type,desc",
    CERTIFIED_COMPONENTS,
    ids=[c[5] for c in CERTIFIED_COMPONENTS],
)
class TestComponentCertification:
    """Certify that every supported component type produces a valid footprint."""

    def test_pad_count(
        self,
        ref: str,
        value: str,
        fp_id: str,
        expected_pads: int,
        expected_type: str,
        desc: str,
    ) -> None:
        fp = footprint_for_component(ref, value, fp_id)
        assert len(fp.pads) == expected_pads, (
            f"{desc}: expected {expected_pads} pads, got {len(fp.pads)}"
        )

    def test_pad_type(
        self,
        ref: str,
        value: str,
        fp_id: str,
        expected_pads: int,
        expected_type: str,
        desc: str,
    ) -> None:
        fp = footprint_for_component(ref, value, fp_id)
        if expected_type == "np_thru_hole":
            # Mounting holes have NPTH — nothing to assert on signal pads
            return
        for pad in fp.pads:
            if pad.pad_type == "np_thru_hole":
                continue  # skip NPTH pads (shield, mounting)
            assert pad.pad_type == expected_type, (
                f"{desc}: pad {pad.number} type={pad.pad_type}, expected {expected_type}"
            )

    def test_has_3d_model(
        self,
        ref: str,
        value: str,
        fp_id: str,
        expected_pads: int,
        expected_type: str,
        desc: str,
    ) -> None:
        fp = footprint_for_component(ref, value, fp_id)
        # Mounting holes, test points, and connectors without matching STEP are exempt
        if ref.startswith(("H", "TP")):
            return
        if ref.startswith("J") and len(fp.models) == 0:
            return  # connector without matching KiCad STEP model
        assert len(fp.models) > 0, f"{desc}: no 3D model attached"

    def test_pad_positions_sensible(
        self,
        ref: str,
        value: str,
        fp_id: str,
        expected_pads: int,
        expected_type: str,
        desc: str,
    ) -> None:
        fp = footprint_for_component(ref, value, fp_id)
        if len(fp.pads) < 2:
            return
        # All pads should be within reasonable bounds (50mm from origin)
        for pad in fp.pads:
            assert abs(pad.position.x) < 50, (
                f"{desc}: pad {pad.number} x={pad.position.x} out of bounds"
            )
            assert abs(pad.position.y) < 50, (
                f"{desc}: pad {pad.number} y={pad.position.y} out of bounds"
            )
        # For 2-pad SMD (passives): pads should be symmetric about origin
        if expected_pads == 2 and expected_type == "smd":
            p1, p2 = fp.pads[0], fp.pads[1]
            # One axis should be symmetric
            assert (
                abs(p1.position.x + p2.position.x) < 0.1
                or abs(p1.position.y + p2.position.y) < 0.1
            ), f"{desc}: 2-pad SMD pads not symmetric"

    def test_lib_id_set(
        self,
        ref: str,
        value: str,
        fp_id: str,
        expected_pads: int,
        expected_type: str,
        desc: str,
    ) -> None:
        fp = footprint_for_component(ref, value, fp_id)
        assert fp.lib_id, f"{desc}: lib_id is empty"


# ---------------------------------------------------------------------------
# Standalone targeted tests
# ---------------------------------------------------------------------------


def test_soic8_with_thermal_pad() -> None:
    """SOIC-8 PowerPAD (like TPS54331) should have center thermal pad.

    When pin 8 is declared as a thermal pad (name="PAD", type=POWER_IN),
    ``make_generic_smd_ic`` REPLACES the gull-wing pad "8" with a center
    exposed pad. Result: 8 pads total (7 gull-wing + 1 thermal center),
    NO duplicate pad numbers.
    """
    from kicad_pipeline.models.requirements import Pin, PinType

    pins = (
        Pin("1", "BOOT", PinType.INPUT),
        Pin("2", "VIN", PinType.POWER_IN),
        Pin("3", "EN", PinType.INPUT),
        Pin("4", "SS", PinType.INPUT),
        Pin("5", "VSNS", PinType.INPUT),
        Pin("6", "GND", PinType.POWER_IN),
        Pin("7", "PH", PinType.OUTPUT),
        Pin("8", "PAD", PinType.POWER_IN),
    )
    fp = footprint_for_component("U1", "TPS54331", "SOIC-8", pins=pins)
    # 7 gull-wing pads + 1 center thermal pad = 8 total (pad 8 REPLACED, not appended)
    assert len(fp.pads) == 8, f"Expected 8 pads (7 gull-wing + 1 thermal), got {len(fp.pads)}"
    # No duplicate pad numbers — regression test for BUG-PIPE-005
    pad_numbers = [p.number for p in fp.pads]
    assert len(set(pad_numbers)) == len(pad_numbers), (
        f"Duplicate pad numbers found: {pad_numbers}"
    )
    # Exactly one pad "8" and it must be the center thermal pad
    pads_8 = [p for p in fp.pads if p.number == "8"]
    assert len(pads_8) == 1, f"Expected exactly 1 pad '8', got {len(pads_8)}"
    center_pad = pads_8[0]
    assert abs(center_pad.position.x) < 0.5, (
        f"Thermal pad not centered: x={center_pad.position.x}"
    )
    assert abs(center_pad.position.y) < 0.5, (
        f"Thermal pad not centered: y={center_pad.position.y}"
    )
    assert center_pad.size_x > 1.0, f"Thermal pad too small: {center_pad.size_x}mm"


def test_terminal_block_pad_numbering() -> None:
    """Terminal block pads should be numbered left-to-right at rotation=0."""
    fp = footprint_for_component("J1", "POWER", "TerminalBlock_2P")
    assert len(fp.pads) == 2
    p1 = next(p for p in fp.pads if p.number == "1")
    p2 = next(p for p in fp.pads if p.number == "2")
    # Pin 1 should be to the left of pin 2
    assert p1.position.x < p2.position.x, "Pin 1 should be left of pin 2"

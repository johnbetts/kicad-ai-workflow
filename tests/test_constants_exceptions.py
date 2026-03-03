"""Tests for constants.py and exceptions.py foundation modules."""

import pytest

from kicad_pipeline import constants, exceptions
from kicad_pipeline.exceptions import (
    ComponentError,
    ConfigurationError,
    DRCError,
    ERCError,
    FileFormatError,
    GerberError,
    GitHubError,
    KiCadPipelineError,
    PCBError,
    ProductionError,
    RequirementsError,
    RoutingError,
    SchematicError,
    SExpError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_exception_hierarchy() -> None:
    """All custom exceptions inherit from KiCadPipelineError."""
    leaf_classes: list[type[KiCadPipelineError]] = [
        RequirementsError,
        ComponentError,
        SchematicError,
        ERCError,
        PCBError,
        RoutingError,
        ValidationError,
        DRCError,
        ProductionError,
        GerberError,
        GitHubError,
        SExpError,
        exceptions.SExpParseError,
        exceptions.SExpWriteError,
        FileFormatError,
        ConfigurationError,
    ]
    for cls in leaf_classes:
        assert issubclass(cls, KiCadPipelineError), (
            f"{cls.__name__} does not inherit from KiCadPipelineError"
        )


def test_exception_messages() -> None:
    """Exceptions accept and preserve messages."""
    message = "something went wrong"
    for cls in (
        KiCadPipelineError,
        RequirementsError,
        ComponentError,
        SchematicError,
        ERCError,
        PCBError,
        RoutingError,
        ValidationError,
        DRCError,
        ProductionError,
        GerberError,
        GitHubError,
        SExpError,
        FileFormatError,
        ConfigurationError,
    ):
        exc = cls(message)
        assert str(exc) == message, (
            f"{cls.__name__} did not preserve message correctly"
        )


def test_sexp_parse_error_preserves_position() -> None:
    """SExpParseError attaches an optional character-position hint."""
    err_no_pos = exceptions.SExpParseError("bad token")
    assert err_no_pos.position is None
    assert "bad token" in str(err_no_pos)

    err_with_pos = exceptions.SExpParseError("bad token", position=42)
    assert err_with_pos.position == 42
    assert "42" in str(err_with_pos)


# ---------------------------------------------------------------------------
# JLCPCB manufacturing constraint self-consistency
# ---------------------------------------------------------------------------


def test_jlcpcb_min_trace_less_than_recommended() -> None:
    """Manufacturing constants are self-consistent."""
    assert constants.JLCPCB_MIN_TRACE_MM < constants.JLCPCB_RECOMMENDED_TRACE_MM
    assert constants.JLCPCB_MIN_CLEARANCE_MM < constants.JLCPCB_RECOMMENDED_CLEARANCE_MM


def test_via_drill_meets_jlcpcb_minimum() -> None:
    """Via defaults meet JLCPCB constraints."""
    assert constants.VIA_DRILL_DEFAULT_MM >= constants.JLCPCB_MIN_VIA_DRILL_MM
    assert constants.VIA_DIAMETER_DEFAULT_MM >= (
        constants.VIA_DRILL_DEFAULT_MM + 2 * constants.JLCPCB_MIN_VIA_ANNULAR_RING_MM
    )


def test_thermal_relief_gap_positive() -> None:
    """Thermal relief gap is positive."""
    assert constants.THERMAL_RELIEF_GAP_MM > 0.0
    assert constants.THERMAL_RELIEF_BRIDGE_MM > 0.0


# ---------------------------------------------------------------------------
# Layer name constants
# ---------------------------------------------------------------------------


def test_layer_names_are_strings() -> None:
    """All layer name constants are non-empty strings."""
    layer_constants = [
        constants.LAYER_F_CU,
        constants.LAYER_B_CU,
        constants.LAYER_F_SILKSCREEN,
        constants.LAYER_B_SILKSCREEN,
        constants.LAYER_F_PASTE,
        constants.LAYER_B_PASTE,
        constants.LAYER_F_MASK,
        constants.LAYER_B_MASK,
        constants.LAYER_F_COURTYARD,
        constants.LAYER_B_COURTYARD,
        constants.LAYER_F_FAB,
        constants.LAYER_B_FAB,
        constants.LAYER_EDGE_CUTS,
        constants.LAYER_IN1_CU,
        constants.LAYER_IN2_CU,
        constants.LAYER_DWGS_USER,
    ]
    for name in layer_constants:
        assert isinstance(name, str) and name, (
            f"Layer constant is empty or not a string: {name!r}"
        )


# ---------------------------------------------------------------------------
# PCB layout defaults
# ---------------------------------------------------------------------------


def test_decoupling_distance_positive() -> None:
    """Decoupling cap max distance is positive."""
    assert constants.DECOUPLING_CAP_MAX_DISTANCE_MM > 0.0


# ---------------------------------------------------------------------------
# Additional sanity checks
# ---------------------------------------------------------------------------


def test_board_size_tuple_structure() -> None:
    """Board size constants are 2-tuples of positive floats."""
    for name, val in (
        ("JLCPCB_MIN_BOARD_SIZE_MM", constants.JLCPCB_MIN_BOARD_SIZE_MM),
        ("JLCPCB_MAX_BOARD_SIZE_MM", constants.JLCPCB_MAX_BOARD_SIZE_MM),
    ):
        assert isinstance(val, tuple) and len(val) == 2, (
            f"{name} must be a 2-tuple"
        )
        assert all(v > 0 for v in val), f"{name} values must be positive"


def test_min_board_smaller_than_max_board() -> None:
    """JLCPCB minimum board size is smaller than maximum in both dimensions."""
    min_w, min_h = constants.JLCPCB_MIN_BOARD_SIZE_MM
    max_w, max_h = constants.JLCPCB_MAX_BOARD_SIZE_MM
    assert min_w < max_w
    assert min_h < max_h


def test_unit_conversion_roundtrip() -> None:
    """MM_PER_MIL and MIL_PER_MM are approximate reciprocals."""
    product = constants.MM_PER_MIL * constants.MIL_PER_MM
    assert abs(product - 1.0) < 1e-4


def test_trace_widths_meet_jlcpcb_minimum() -> None:
    """All net-class trace widths are at or above the JLCPCB minimum."""
    widths = [
        constants.TRACE_WIDTH_DEFAULT_MM,
        constants.TRACE_WIDTH_POWER_MM,
        constants.TRACE_WIDTH_USB_DIFF_MM,
        constants.TRACE_WIDTH_ANALOG_MM,
    ]
    for w in widths:
        assert w >= constants.JLCPCB_MIN_TRACE_MM, (
            f"Trace width {w} mm is below JLCPCB minimum"
        )


def test_kicad_version_constants_are_positive_ints() -> None:
    """KiCad version integers are positive."""
    assert isinstance(constants.KICAD_SCH_VERSION, int)
    assert isinstance(constants.KICAD_PCB_VERSION, int)
    assert constants.KICAD_SCH_VERSION > 0
    assert constants.KICAD_PCB_VERSION > 0


def test_kicad_generator_is_non_empty_string() -> None:
    """KICAD_GENERATOR is a non-empty string."""
    assert isinstance(constants.KICAD_GENERATOR, str) and constants.KICAD_GENERATOR


def test_component_defaults_positive() -> None:
    """Component default values are positive numbers."""
    assert constants.LED_TARGET_CURRENT_MA > 0.0
    assert constants.DEFAULT_PULLUP_RESISTANCE_OHMS > 0.0
    assert constants.DEFAULT_DECOUPLING_VALUE_UF > 0.0
    assert constants.DEFAULT_BULK_DECOUPLING_VALUE_UF > 0.0
    assert constants.THERMAL_WARNING_MW > 0.0


def test_exception_is_catchable_as_base() -> None:
    """Leaf exceptions can be caught by the base KiCadPipelineError type."""
    with pytest.raises(KiCadPipelineError):
        raise DRCError("drc failed")

    with pytest.raises(KiCadPipelineError):
        raise GerberError("gerber failed")

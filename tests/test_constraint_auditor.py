"""Tests for placement constraint auditor."""
from __future__ import annotations

from kicad_pipeline.models.pcb import (
    OrderingChain,
    ProximityConstraint,
)
from kicad_pipeline.validation.constraint_auditor import (
    ConstraintViolation,
    _check_group_cohesion,
    _check_ordering,
    _check_proximity,
)


def test_proximity_within_limit() -> None:
    """No violation when component is within max distance."""
    c = ProximityConstraint(ref="C1", target_ref="U1", max_distance_mm=10.0)
    positions = {"C1": (5.0, 5.0, 0.0), "U1": (10.0, 5.0, 0.0)}
    assert _check_proximity(c, positions) is None


def test_proximity_violation() -> None:
    """Violation when component exceeds max distance."""
    c = ProximityConstraint(ref="C1", target_ref="U1", max_distance_mm=5.0)
    positions = {"C1": (0.0, 0.0, 0.0), "U1": (20.0, 0.0, 0.0)}
    v = _check_proximity(c, positions)
    assert v is not None
    assert v.constraint_type == "proximity"
    assert v.measured_value > 5.0
    assert "C1" in v.message


def test_proximity_missing_ref() -> None:
    """No violation when ref is missing (can't check)."""
    c = ProximityConstraint(ref="C1", target_ref="U1", max_distance_mm=5.0)
    positions = {"U1": (10.0, 5.0, 0.0)}
    assert _check_proximity(c, positions) is None


def test_proximity_with_pin() -> None:
    """Violation message includes pin name."""
    c = ProximityConstraint(
        ref="C1", target_ref="U1", target_pin="VIN", max_distance_mm=3.0,
    )
    positions = {"C1": (0.0, 0.0, 0.0), "U1": (20.0, 0.0, 0.0)}
    v = _check_proximity(c, positions)
    assert v is not None
    assert "VIN" in v.message


def test_proximity_severity_critical() -> None:
    """Critical severity when distance > 2x threshold."""
    c = ProximityConstraint(ref="C1", target_ref="U1", max_distance_mm=5.0)
    positions = {"C1": (0.0, 0.0, 0.0), "U1": (50.0, 0.0, 0.0)}
    v = _check_proximity(c, positions)
    assert v is not None
    assert v.severity == "critical"


def test_ordering_correct() -> None:
    """No violations when order matches X positions."""
    chain = OrderingChain(group="buck", refs=("J1", "C1", "U1"))
    positions = {"J1": (5.0, 10.0, 0.0), "C1": (15.0, 10.0, 0.0), "U1": (25.0, 10.0, 0.0)}
    assert _check_ordering(chain, positions) == []


def test_ordering_violation() -> None:
    """Violation when components are out of order."""
    chain = OrderingChain(group="buck", refs=("J1", "C1", "U1"))
    positions = {"J1": (25.0, 10.0, 0.0), "C1": (15.0, 10.0, 0.0), "U1": (5.0, 10.0, 0.0)}
    violations = _check_ordering(chain, positions)
    assert len(violations) >= 1
    assert violations[0].constraint_type == "ordering"


def test_ordering_y_axis() -> None:
    """Uses Y axis when Y spread > X spread."""
    chain = OrderingChain(group="power", refs=("J1", "U1", "J2"))
    # All same X, different Y — should use Y axis
    positions = {"J1": (10.0, 5.0, 0.0), "U1": (10.0, 15.0, 0.0), "J2": (10.0, 25.0, 0.0)}
    assert _check_ordering(chain, positions) == []


def test_ordering_single_ref() -> None:
    """No violations with only one ref present."""
    chain = OrderingChain(group="buck", refs=("J1", "C1"))
    positions = {"J1": (5.0, 10.0, 0.0)}
    assert _check_ordering(chain, positions) == []


def test_group_cohesion_ok() -> None:
    """No violation when group is compact."""
    v = _check_group_cohesion("power", ("U1", "C1", "L1"), {
        "U1": (10.0, 10.0, 0.0), "C1": (15.0, 10.0, 0.0), "L1": (20.0, 10.0, 0.0),
    })
    assert v is None


def test_group_cohesion_violation() -> None:
    """Violation when group spread exceeds threshold."""
    v = _check_group_cohesion("power", ("U1", "C1", "L1"), {
        "U1": (0.0, 0.0, 0.0), "C1": (50.0, 50.0, 0.0), "L1": (10.0, 10.0, 0.0),
    }, max_spread_mm=30.0)
    assert v is not None
    assert v.constraint_type == "group"
    assert "power" in v.message


def test_group_cohesion_single_member() -> None:
    """No violation with only one member."""
    v = _check_group_cohesion("power", ("U1",), {"U1": (10.0, 10.0, 0.0)})
    assert v is None


def test_violation_dataclass() -> None:
    """ConstraintViolation is a proper frozen dataclass."""
    v = ConstraintViolation(
        constraint_type="proximity",
        severity="major",
        refs=("C1", "U1"),
        message="C1 is 15mm from U1 (max 5mm)",
        measured_value=15.0,
        threshold=5.0,
        suggested_fix="Move C1 closer",
    )
    assert v.constraint_type == "proximity"
    assert v.refs == ("C1", "U1")
    assert v.measured_value == 15.0

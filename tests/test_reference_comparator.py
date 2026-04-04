"""Tests for the reference-driven placement comparator."""

from __future__ import annotations

import pytest

from kicad_pipeline.optimization.reference_comparator import (
    compare_to_reference,
    worst_group,
)


@pytest.fixture
def group_map() -> dict[str, str]:
    return {
        "R1": "Power",
        "R2": "Power",
        "C1": "Power",
        "U1": "MCU",
        "C2": "MCU",
        "J1": "Relay",
        "K1": "Relay",
    }


class TestCompareToReference:
    """Tests for compare_to_reference()."""

    def test_perfect_match_returns_100(self, group_map: dict[str, str]) -> None:
        positions = {
            "R1": (10.0, 20.0, 0.0),
            "R2": (15.0, 20.0, 0.0),
            "C1": (12.0, 25.0, 0.0),
            "U1": (50.0, 40.0, 0.0),
            "C2": (52.0, 40.0, 0.0),
            "J1": (80.0, 10.0, 0.0),
            "K1": (85.0, 10.0, 0.0),
        }
        result = compare_to_reference(positions, positions, group_map)
        assert result["overall"] == pytest.approx(100.0)
        assert result["Power"] == pytest.approx(100.0)
        assert result["MCU"] == pytest.approx(100.0)
        assert result["Relay"] == pytest.approx(100.0)

    def test_offset_reduces_similarity(self, group_map: dict[str, str]) -> None:
        reference = {
            "R1": (10.0, 20.0, 0.0),
            "R2": (15.0, 20.0, 0.0),
            "C1": (12.0, 25.0, 0.0),
            "U1": (50.0, 40.0, 0.0),
            "C2": (52.0, 40.0, 0.0),
            "J1": (80.0, 10.0, 0.0),
            "K1": (85.0, 10.0, 0.0),
        }
        # Shift Power group 10mm right
        current = dict(reference)
        current["R1"] = (20.0, 20.0, 0.0)
        current["R2"] = (25.0, 20.0, 0.0)
        current["C1"] = (22.0, 25.0, 0.0)

        result = compare_to_reference(current, reference, group_map)
        assert result["overall"] < 100.0
        assert result["Power"] < 100.0
        # MCU and Relay untouched
        assert result["MCU"] == pytest.approx(100.0)
        assert result["Relay"] == pytest.approx(100.0)

    def test_max_error_returns_zero(self, group_map: dict[str, str]) -> None:
        reference = {"R1": (10.0, 20.0, 0.0), "R2": (15.0, 20.0, 0.0)}
        # 20mm+ offset should give 0% similarity
        current = {"R1": (35.0, 20.0, 0.0), "R2": (40.0, 20.0, 0.0)}
        result = compare_to_reference(current, reference, group_map)
        assert result["Power"] <= 5.0  # ~0% with 20mm+ error

    def test_unmatched_refs_excluded(self, group_map: dict[str, str]) -> None:
        reference = {"R1": (10.0, 20.0, 0.0)}
        current = {"R1": (10.0, 20.0, 0.0), "X99": (99.0, 99.0, 0.0)}
        result = compare_to_reference(current, reference, group_map)
        assert result["overall"] == pytest.approx(100.0)

    def test_empty_inputs(self) -> None:
        result = compare_to_reference({}, {}, {})
        assert result["overall"] == 0.0


class TestWorstGroup:
    """Tests for worst_group()."""

    def test_returns_lowest_group(self) -> None:
        sim = {"Power": 90.0, "MCU": 50.0, "Relay": 70.0, "overall": 70.0}
        assert worst_group(sim) == "MCU"

    def test_excludes_overall(self) -> None:
        sim = {"Power": 90.0, "overall": 10.0}
        assert worst_group(sim) == "Power"

    def test_empty_returns_none(self) -> None:
        assert worst_group({"overall": 50.0}) is None

"""Part-rule loader tests — opening calibration (Gate C feedback 2026-06-11).

The TerminalBlock opening was once ASSERTED as [0,-1] (a guess) and the
face_out check then *confirmed the guess*: three training boards shipped
with screw terminals facing the board interior. These tests pin the
MEASURED openings (isolated-render calibration, 2026-06-11) and the
loader's calibration semantics so a silent regression to a guessed
value cannot pass.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import pytest

from kicad_pipeline.exceptions import ConfigurationError
from kicad_pipeline.placement_v2.part_rules import load_part_rules

_REPO = Path(__file__).resolve().parents[2]
_PART_RULES = _REPO / "data" / "part_rules.json"


def _write_rules(tmp_path: Path, rules: list[dict[str, object]]) -> Path:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"rules": rules}), encoding="utf-8")
    return path


class TestCalibratedFlag:
    def test_calibrated_opening_parses(self, tmp_path: Path) -> None:
        path = _write_rules(tmp_path, [{
            "match": {"footprint_contains": "TerminalBlock"},
            "edge_pin": True,
            "opening_mm": [0.0, 1.0],
            "calibrated": True,
        }])
        rule = load_part_rules(path).rules[0]
        assert rule.opening == (0.0, 1.0)
        assert rule.calibrated is True

    def test_uncalibrated_opening_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = _write_rules(tmp_path, [{
            "match": {"footprint_contains": "USB"},
            "opening_mm": [0.0, -1.0],
        }])
        with caplog.at_level(logging.WARNING):
            rule = load_part_rules(path).rules[0]
        assert rule.calibrated is False
        assert any("UNCALIBRATED" in rec.message for rec in caplog.records)

    def test_calibrated_without_opening_rejected(self, tmp_path: Path) -> None:
        path = _write_rules(tmp_path, [{
            "match": {"footprint_contains": "RJ45"},
            "calibrated": True,
        }])
        with pytest.raises(ConfigurationError, match="without an 'opening_mm'"):
            load_part_rules(path)

    def test_non_bool_calibrated_rejected(self, tmp_path: Path) -> None:
        path = _write_rules(tmp_path, [{
            "match": {"footprint_contains": "RJ45"},
            "opening_mm": [0.0, -1.0],
            "calibrated": "yes",
        }])
        with pytest.raises(ConfigurationError, match="must be a boolean"):
            load_part_rules(path)


class TestShippedCalibrationData:
    """Pin the 2026-06-11 isolated-render measurements in data/part_rules.json."""

    #: Measured opening per footprint_contains match (KiCad frame, Y down,
    #: rotation 0). Source: output/calibration/verdicts_2026-06-11.json.
    MEASURED: ClassVar[dict[str, tuple[float, float]]] = {
        "TerminalBlock": (0.0, 1.0),  # wire entry faces SOUTH (was guessed [0,-1])
        "USB": (0.0, 1.0),  # plug mouth faces SOUTH
        "RJ45": (0.0, -1.0),  # jack mouth faces NORTH
        "ESP32": (0.0, -1.0),  # antenna section on the NORTH side
    }

    def test_every_shipped_opening_is_calibrated(self) -> None:
        rules = load_part_rules(_PART_RULES).rules
        for rule in rules:
            if rule.opening is not None:
                assert rule.calibrated, (
                    f"rule {rule.match} has an uncalibrated opening — "
                    "run scripts/calibrate_part_openings.py"
                )

    def test_shipped_openings_match_measurements(self) -> None:
        rules = load_part_rules(_PART_RULES).rules
        seen: set[str] = set()
        for rule in rules:
            match = rule.match.footprint_contains
            if match in self.MEASURED:
                assert rule.opening == self.MEASURED[match], (
                    f"{match} opening drifted from the 2026-06-11 calibration; "
                    "re-measure with scripts/calibrate_part_openings.py before changing"
                )
                seen.add(match)
        assert seen == set(self.MEASURED), f"missing rules for {set(self.MEASURED) - seen}"

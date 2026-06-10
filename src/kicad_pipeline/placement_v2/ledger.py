"""Build ledger — harness-computed proof of what each stage did.

Every pipeline stage appends a :class:`StageRecord` with input/output
hashes, the constraints it checked, and quantified verdicts. The ledger
is written by deterministic code, never by a model, so "done" is
defined as *ledger shows all gates green* — an agent's prose claim of
success is checkable against the ledger in one call.

Storage is append-only JSONL: corruption or truncation of one line
never destroys earlier records, and re-runs append rather than rewrite.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import KiCadPipelineError
from kicad_pipeline.placement_v2.ir import Severity, Violation

if TYPE_CHECKING:
    from pathlib import Path


class LedgerError(KiCadPipelineError):
    """The ledger file is unreadable or a record is malformed."""


def sha256_text(text: str) -> str:
    """Stable hash of a text artifact."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Stable hash of a file on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class StageRecord:
    """One stage's entry: what went in, what came out, what was proven."""

    stage: str  # "certify", "cells", "floorplan", "gate_a", "gate_b", "sign_off"
    input_sha256: str
    output_sha256: str
    checks_run: tuple[str, ...]
    violations: tuple[Violation, ...]
    passed: bool
    detail: str = ""
    timestamp: str = ""  # ISO; provided by caller to keep records pure

    def to_json(self) -> str:
        """Serialize to a single JSONL line (stable key order)."""
        d = asdict(self)
        for v in d["violations"]:
            v["severity"] = v["severity"].value
        return json.dumps(d, sort_keys=True)

    @staticmethod
    def from_json(line: str) -> StageRecord:
        """Parse one JSONL line back into a record."""
        try:
            d = json.loads(line)
            violations = tuple(
                Violation(
                    constraint=v["constraint"],
                    refs=tuple(v["refs"]),
                    severity=Severity(v["severity"]),
                    measured=float(v["measured"]),
                    limit=float(v["limit"]),
                    message=v["message"],
                )
                for v in d["violations"]
            )
            return StageRecord(
                stage=d["stage"],
                input_sha256=d["input_sha256"],
                output_sha256=d["output_sha256"],
                checks_run=tuple(d["checks_run"]),
                violations=violations,
                passed=bool(d["passed"]),
                detail=d.get("detail", ""),
                timestamp=d.get("timestamp", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerError(f"malformed ledger record: {exc}") from exc


@dataclass(frozen=True)
class BuildLedger:
    """Append-only ledger for one board build."""

    path: Path

    def append(self, record: StageRecord) -> None:
        """Append one stage record (creates the file on first write)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    def records(self) -> tuple[StageRecord, ...]:
        """All records, oldest first. Missing file -> empty."""
        if not self.path.exists():
            return ()
        out: list[StageRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(StageRecord.from_json(line))
        return tuple(out)

    def latest(self, stage: str) -> StageRecord | None:
        """Most recent record for a stage, or None."""
        for rec in reversed(self.records()):
            if rec.stage == stage:
                return rec
        return None

    def all_green(self, required_stages: tuple[str, ...]) -> bool:
        """True iff the LATEST record of every required stage passed.

        This is the machine definition of "done" — prose claims do not
        count; only this function does.
        """
        return all(
            (rec := self.latest(stage)) is not None and rec.passed
            for stage in required_stages
        )

    def summary(self) -> str:
        """Human-readable one-line-per-stage status."""
        lines = []
        seen: set[str] = set()
        for rec in reversed(self.records()):
            if rec.stage in seen:
                continue
            seen.add(rec.stage)
            mark = "PASS" if rec.passed else "FAIL"
            extra = f" ({len(rec.violations)} violations)" if rec.violations else ""
            lines.append(f"{rec.stage}: {mark}{extra}")
        return "\n".join(reversed(lines))


#: Stages that must be green before a board may be signed off.
REQUIRED_STAGES: tuple[str, ...] = (
    "certify",
    "cells",
    "floorplan",
    "gate_a",
)

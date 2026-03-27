"""JSONL-based file persistence for the evidence ledger.

Each board gets one JSONL file at `.evidence/{board_name}.jsonl`.
Records are appended one per line — crash-safe, no read-modify-write.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from kicad_pipeline.evidence.models import EvidenceLedger, EvidenceRecord

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

EVIDENCE_DIR = ".evidence"


def ledger_dir(project_root: Path) -> Path:
    """Return the evidence directory for a project."""
    return project_root / EVIDENCE_DIR


def ledger_path(board_path: Path) -> Path:
    """Return the JSONL file path for a board.

    Given ``output/train_mcu_core/train_mcu_core.kicad_pcb``, returns
    ``output/train_mcu_core/.evidence/train_mcu_core.jsonl``.
    """
    board_dir = board_path.parent
    board_name = board_path.stem
    return board_dir / EVIDENCE_DIR / f"{board_name}.jsonl"


def append_record(board_path: Path, record: EvidenceRecord) -> Path:
    """Append a single evidence record to the board's JSONL ledger.

    Creates the file and parent directory if they don't exist.
    Returns the path to the JSONL file.
    """
    path = ledger_path(board_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
    return path


def load_ledger(board_path: Path) -> EvidenceLedger:
    """Load all evidence records for a board from its JSONL file.

    Returns an empty ledger if the file does not exist.
    Skips corrupted lines gracefully with a warning.
    """
    path = ledger_path(board_path)
    board_name = board_path.stem
    records: list[EvidenceRecord] = []

    if not path.exists():
        return EvidenceLedger(board=board_name)

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(EvidenceRecord.model_validate(data))
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning(
                    "Skipping corrupted line %d in %s: %s", line_num, path, exc
                )

    return EvidenceLedger(board=board_name, records=records)


def clear_ledger(board_path: Path) -> None:
    """Delete the ledger file for a board (for testing)."""
    path = ledger_path(board_path)
    if path.exists():
        path.unlink()

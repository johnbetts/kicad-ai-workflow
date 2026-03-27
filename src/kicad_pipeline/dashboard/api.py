"""FastAPI endpoints mounted on the NiceGUI app for programmatic access."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel

from kicad_pipeline.evidence.ledger import append_record, load_ledger
from kicad_pipeline.evidence.models import EvidenceKind, EvidenceRecord

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.applications import Starlette
    from starlette.requests import Request

logger = logging.getLogger(__name__)

# Module-level output root — set during register_api_routes().
_output_root: Path | None = None


def _project_root() -> Path:
    """Return the project root (parent of the output directory)."""
    if _output_root is None:
        msg = "Output root not configured"
        raise RuntimeError(msg)
    return _output_root.parent


class LogMessage(BaseModel):
    """Payload for the POST /api/log endpoint."""

    message: str
    level: str = "info"


class ApproveRequest(BaseModel):
    """Payload for POST /api/approve."""

    stage: str = "pcb"


class RejectRequest(BaseModel):
    """Payload for POST /api/reject."""

    stage: str = "pcb"
    feedback: str = ""


class CommandRequest(BaseModel):
    """Payload for POST /api/command."""

    command: str
    board: str = ""


def _board_pcb_path(board_name: str) -> Path:
    """Resolve a board name to its .kicad_pcb path."""

    if _output_root is None:
        msg = "Output root not configured"
        raise RuntimeError(msg)
    board_dir = _output_root / board_name
    # Find the first .kicad_pcb file
    pcbs = list(board_dir.glob("*.kicad_pcb"))
    if pcbs:
        return pcbs[0]
    return board_dir / f"{board_name}.kicad_pcb"


def _discover_board_names() -> list[str]:
    """Return sorted list of board directory names."""
    if _output_root is None or not _output_root.is_dir():
        return []
    boards: list[str] = []
    for child in sorted(_output_root.iterdir()):
        if (
            child.is_dir()
            and not child.name.startswith(("_", "."))
            and any(child.glob("*.kicad_pcb"))
        ):
            boards.append(child.name)
    return boards


def register_api_routes(app: Starlette, output_root: Path) -> None:
    """Register FastAPI-style API routes on the NiceGUI/Starlette app."""
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from starlette.routing import Route

    global _output_root
    _output_root = output_root

    async def post_evidence(request: Request) -> JSONResponse:
        """Accept an EvidenceRecord JSON body and append to ledger."""
        body = await request.json()
        try:
            record = EvidenceRecord.model_validate(body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        board_pcb = _board_pcb_path(record.board)
        append_record(board_pcb, record)
        return JSONResponse({"status": "ok", "id": record.id}, status_code=201)

    async def post_log(request: Request) -> JSONResponse:
        """Accept a log message and push to the shared buffer."""
        body = await request.json()
        msg = LogMessage.model_validate(body)
        from kicad_pipeline.dashboard.app import _log_buffer

        ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        _log_buffer.append(f"[{ts}] [{msg.level.upper()}] {msg.message}")
        return JSONResponse({"status": "ok"}, status_code=201)

    async def get_ledger(request: Request) -> JSONResponse:
        """Return the full evidence ledger for a board as JSON."""
        board_name = request.path_params["board_name"]
        board_pcb = _board_pcb_path(board_name)
        ledger = load_ledger(board_pcb)
        return JSONResponse(ledger.model_dump(mode="json"))

    async def post_approve(request: Request) -> JSONResponse:
        """Write a HUMAN_APPROVAL record to the ledger."""
        board_name = request.path_params["board_name"]
        body = await request.json()
        req = ApproveRequest.model_validate(body)
        board_pcb = _board_pcb_path(board_name)
        record = EvidenceRecord(
            kind=EvidenceKind.HUMAN_APPROVAL,
            stage=req.stage,
            step="api_approval",
            board=board_name,
            passed=True,
            summary=f"Human approved {req.stage} via API",
            producer="api",
        )
        append_record(board_pcb, record)
        return JSONResponse({"status": "ok", "id": record.id}, status_code=201)

    async def post_reject(request: Request) -> JSONResponse:
        """Write a HUMAN_REJECTION record to the ledger."""
        board_name = request.path_params["board_name"]
        body = await request.json()
        req = RejectRequest.model_validate(body)
        board_pcb = _board_pcb_path(board_name)
        record = EvidenceRecord(
            kind=EvidenceKind.HUMAN_REJECTION,
            stage=req.stage,
            step="api_rejection",
            board=board_name,
            passed=False,
            summary=f"Human rejected {req.stage} via API",
            feedback=req.feedback,
            producer="api",
        )
        append_record(board_pcb, record)
        return JSONResponse({"status": "ok", "id": record.id}, status_code=201)

    async def get_boards(request: Request) -> JSONResponse:
        """List available board names from the output directory."""
        boards = _discover_board_names()
        return JSONResponse({"boards": boards})

    # -- Kanban endpoints --------------------------------------------------

    async def get_kanban(request: Request) -> JSONResponse:
        """Return the full kanban board as JSON."""
        from kicad_pipeline.dashboard.kanban import load_kanban

        board = load_kanban(_project_root())
        return JSONResponse(board.model_dump())

    async def post_kanban_card(request: Request) -> JSONResponse:
        """Create a new kanban card."""
        from kicad_pipeline.dashboard.kanban import KanbanCard, add_card

        body = await request.json()
        try:
            card = KanbanCard.model_validate(body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = add_card(_project_root(), card)
        return JSONResponse(result.model_dump(), status_code=201)

    async def put_kanban_card(request: Request) -> JSONResponse:
        """Update an existing kanban card."""
        from kicad_pipeline.dashboard.kanban import update_card

        card_id = request.path_params["card_id"]
        body = await request.json()
        try:
            result = update_card(_project_root(), card_id, body)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(result.model_dump())

    async def put_kanban_card_move(request: Request) -> JSONResponse:
        """Move a kanban card to a new status column."""
        from kicad_pipeline.dashboard.kanban import move_card

        card_id = request.path_params["card_id"]
        body = await request.json()
        new_status = body.get("status", "")
        try:
            result = move_card(_project_root(), card_id, new_status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(result.model_dump())

    async def delete_kanban_card(request: Request) -> JSONResponse:
        """Delete a kanban card by ID."""
        from kicad_pipeline.dashboard.kanban import delete_card

        card_id = request.path_params["card_id"]
        try:
            delete_card(_project_root(), card_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"status": "ok"})

    async def post_kanban_import_roadmap(request: Request) -> JSONResponse:
        """Import cards from a roadmap.md file."""
        from pathlib import Path

        from kicad_pipeline.dashboard.kanban import import_from_roadmap

        body = await request.json()
        roadmap_path = Path(body.get("roadmap_path", ""))
        if not roadmap_path.exists():
            raise HTTPException(status_code=404, detail=f"Roadmap not found: {roadmap_path}")
        count = import_from_roadmap(_project_root(), roadmap_path)
        return JSONResponse({"status": "ok", "imported": count})

    async def post_command(request: Request) -> JSONResponse:
        """Execute a dashboard command and return the response."""
        body = await request.json()
        req = CommandRequest.model_validate(body)
        cmd_text = req.command.strip()
        if not cmd_text.startswith("/"):
            # Non-command text is echoed to the log buffer.
            ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
            from kicad_pipeline.dashboard.app import _log_buffer

            _log_buffer.append(f"[{ts}] > {cmd_text}")
            return JSONResponse({"response": cmd_text})

        board_dir: Path | None = None
        if _output_root and req.board:
            candidate = _output_root / req.board
            if candidate.is_dir():
                board_dir = candidate

        from kicad_pipeline.dashboard.panels import _dispatch_command

        response = _dispatch_command(cmd_text, board_dir, None)
        return JSONResponse({"response": response})

    # Mount routes on the Starlette app
    api_routes = [
        Route("/api/evidence", post_evidence, methods=["POST"]),
        Route("/api/log", post_log, methods=["POST"]),
        Route("/api/command", post_command, methods=["POST"]),
        Route("/api/ledger/{board_name}", get_ledger, methods=["GET"]),
        Route("/api/approve/{board_name}", post_approve, methods=["POST"]),
        Route("/api/reject/{board_name}", post_reject, methods=["POST"]),
        Route("/api/boards", get_boards, methods=["GET"]),
        Route("/api/kanban", get_kanban, methods=["GET"]),
        Route("/api/kanban/cards", post_kanban_card, methods=["POST"]),
        Route("/api/kanban/cards/{card_id}", put_kanban_card, methods=["PUT"]),
        Route("/api/kanban/cards/{card_id}", delete_kanban_card, methods=["DELETE"]),
        Route("/api/kanban/cards/{card_id}/move", put_kanban_card_move, methods=["PUT"]),
        Route("/api/kanban/import-roadmap", post_kanban_import_roadmap, methods=["POST"]),
    ]
    app.routes.extend(api_routes)

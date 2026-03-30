"""Chat API router — placeholder with canned responses.

Future: Will call Claude API for real AI assistance.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    context: str = ""
    history: list[dict[str, str]] = []


class ChatResponse(BaseModel):
    response: str
    suggestion: str | None = None


@router.post("/message", response_model=ChatResponse)
async def chat_message(body: ChatRequest) -> ChatResponse:
    """Process a chat message. Currently returns helpful canned responses.

    Future: Will call Claude API for real AI assistance.
    """
    response = _generate_response(body.message, body.context)
    return ChatResponse(response=response)


def _generate_response(message: str, context: str) -> str:
    """Generate a helpful response based on the message content."""
    lower = message.lower()

    if any(w in lower for w in ("component", "part", "capacitor", "resistor", "inductor")):
        return (
            "For component selection, I recommend starting with JLCPCB basic parts "
            "to avoid setup fees. Use the Parts Search (Cmd+K or /parts) to find "
            "alternatives. What specific component are you looking for?"
        )
    if any(w in lower for w in ("placement", "layout", "move", "position")):
        return (
            "The placement optimizer runs in 3 levels:\n"
            "1. Zone partitioning (power/signal/RF domains)\n"
            "2. Group placement (related components together)\n"
            "3. Subcircuit refinement (fine-tuning)\n\n"
            "Click 'Fix It' on any finding to apply corrections automatically."
        )
    if any(w in lower for w in ("drc", "error", "violation", "clearance")):
        return (
            "DRC checks validate against JLCPCB manufacturing rules by default "
            "(0.15mm min trace, 0.2mm min drill, 0.3mm min clearance). "
            "Each violation includes a fix suggestion. "
            "Would you like me to explain a specific error code?"
        )
    if any(w in lower for w in ("cost", "price", "cheap", "expensive", "budget")):
        return (
            "Cost optimization tips:\n"
            "- Use basic parts (no $3/part setup fee)\n"
            "- 2-layer boards save ~$1.50/unit vs 4-layer\n"
            "- Standard 1.6mm FR4 is cheapest\n"
            "- Minimum order is 5 boards at JLCPCB\n\n"
            "Check the Order tab for a full cost breakdown."
        )
    if any(w in lower for w in ("route", "routing", "trace", "wire")):
        return (
            "Routing is done manually in KiCad after exporting. "
            "Download the .kicad_pcb file, open in KiCad, and use the "
            "interactive router. The pipeline generates optimally-placed "
            "but unrouted boards."
        )

    return (
        "I can help with:\n"
        "- **Component selection** -- find JLCPCB parts, check stock\n"
        "- **Placement review** -- analyze and fix layout issues\n"
        "- **DRC explanation** -- understand and resolve violations\n"
        "- **Cost optimization** -- reduce BOM and fabrication costs\n"
        "- **Manufacturing prep** -- generate production files\n\n"
        "What would you like to work on?"
    )

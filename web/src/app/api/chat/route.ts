import { NextRequest, NextResponse } from "next/server";

interface ChatRequestBody {
  message: string;
  context?: string;
  history?: { role: string; content: string }[];
}

/** Generate a helpful canned response based on message content. */
function generateResponse(message: string, _context: string): string {
  const lower = message.toLowerCase();

  if (["component", "part", "capacitor", "resistor", "inductor"].some(w => lower.includes(w))) {
    return (
      "For component selection, I recommend starting with JLCPCB basic parts " +
      "to avoid setup fees. Use the Parts Search (Cmd+K or /parts) to find " +
      "alternatives. What specific component are you looking for?"
    );
  }
  if (["placement", "layout", "move", "position"].some(w => lower.includes(w))) {
    return (
      "The placement optimizer runs in 3 levels:\n" +
      "1. Zone partitioning (power/signal/RF domains)\n" +
      "2. Group placement (related components together)\n" +
      "3. Subcircuit refinement (fine-tuning)\n\n" +
      "Click 'Fix It' on any finding to apply corrections automatically."
    );
  }
  if (["drc", "error", "violation", "clearance"].some(w => lower.includes(w))) {
    return (
      "DRC checks validate against JLCPCB manufacturing rules by default " +
      "(0.15mm min trace, 0.2mm min drill, 0.3mm min clearance). " +
      "Each violation includes a fix suggestion. " +
      "Would you like me to explain a specific error code?"
    );
  }
  if (["cost", "price", "cheap", "expensive", "budget"].some(w => lower.includes(w))) {
    return (
      "Cost optimization tips:\n" +
      "- Use basic parts (no $3/part setup fee)\n" +
      "- 2-layer boards save ~$1.50/unit vs 4-layer\n" +
      "- Standard 1.6mm FR4 is cheapest\n" +
      "- Minimum order is 5 boards at JLCPCB\n\n" +
      "Check the Order tab for a full cost breakdown."
    );
  }
  if (["route", "routing", "trace", "wire"].some(w => lower.includes(w))) {
    return (
      "Routing is done manually in KiCad after exporting. " +
      "Download the .kicad_pcb file, open in KiCad, and use the " +
      "interactive router. The pipeline generates optimally-placed " +
      "but unrouted boards."
    );
  }

  return (
    "I can help with:\n" +
    "- **Component selection** \u2014 find JLCPCB parts, check stock\n" +
    "- **Placement review** \u2014 analyze and fix layout issues\n" +
    "- **DRC explanation** \u2014 understand and resolve violations\n" +
    "- **Cost optimization** \u2014 reduce BOM and fabrication costs\n" +
    "- **Manufacturing prep** \u2014 generate production files\n\n" +
    "What would you like to work on?"
  );
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as ChatRequestBody;
    const response = generateResponse(body.message, body.context ?? "");

    return NextResponse.json({ response, suggestion: null });
  } catch {
    return NextResponse.json(
      { error: "Failed to process chat message" },
      { status: 400 },
    );
  }
}

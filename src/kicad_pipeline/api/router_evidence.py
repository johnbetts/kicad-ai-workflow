"""Evidence CRUD endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from kicad_pipeline.api.schemas import BoardDetailSchema, BoardSummarySchema, EvidenceSubmitSchema

router = APIRouter()


@router.get("/boards", response_model=list[BoardSummarySchema])
async def list_boards(output_dir: str = "output") -> list[BoardSummarySchema]:
    """List all boards with summary info."""
    from kicad_pipeline.evidence.ledger import load_ledger

    boards: list[BoardSummarySchema] = []
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    for board_dir in sorted(output_path.iterdir()):
        if not board_dir.is_dir() or board_dir.name.startswith(("_", ".")):
            continue
        # Find PCB file
        pcb_files = list(board_dir.glob("*.kicad_pcb"))
        if not pcb_files:
            boards.append(BoardSummarySchema(name=board_dir.name))
            continue

        try:
            ledger = load_ledger(pcb_files[0])
            score = ledger.latest_score()
            boards.append(
                BoardSummarySchema(
                    name=board_dir.name,
                    grade=score.grade if score else "",
                    score=score.overall_score if score else None,
                    has_evidence=len(ledger.records) > 0,
                )
            )
        except Exception:
            boards.append(BoardSummarySchema(name=board_dir.name))

    return boards


@router.get("/board/{board_name}", response_model=BoardDetailSchema)
async def get_board_detail(
    board_name: str, output_dir: str = "output"
) -> BoardDetailSchema:
    """Get full board details including image URLs and score."""
    board_dir = Path(output_dir) / board_name
    if not board_dir.exists():
        raise HTTPException(404, f"Board not found: {board_name}")

    # Find images
    images: dict[str, str] = {}
    for png in sorted(board_dir.glob("*.png")):
        name = png.name.lower()
        url = f"/api/boards/{board_name}/{png.name}"
        if "_2d_top" in name or "_2d." in name:
            images["2d"] = url
        elif "_3d_top" in name:
            images["3d-top"] = url
        elif "_3d_iso" in name and "back" not in name:
            images["3d-iso"] = url
        elif "_3d_isoback" in name or "_3d_back" in name:
            images["3d-back"] = url
        elif "_placement" in name:
            images["placement"] = url

    # Find crop images
    crops: list[dict[str, str]] = []
    crops_dir = board_dir / "crops"
    if crops_dir.exists():
        for crop in sorted(crops_dir.glob("*.png")):
            ref = crop.stem.replace("_crop", "").replace("_3d", "").split("_")[0]
            crops.append({
                "ref": ref,
                "url": f"/api/boards/{board_name}/crops/{crop.name}",
                "filename": crop.name,
            })

    # Load score from evidence
    score_data: dict[str, object] | None = None
    pcb_files = list(board_dir.glob("*.kicad_pcb"))
    if pcb_files:
        try:
            from kicad_pipeline.evidence.ledger import load_ledger

            ledger = load_ledger(pcb_files[0])
            score = ledger.latest_score()
            if score:
                score_data = {
                    "grade": score.grade,
                    "overall_score": score.overall_score,
                    "breakdown": score.breakdown,
                }
        except Exception:
            pass

    # List all files
    files = [f.name for f in board_dir.iterdir() if f.is_file()]

    # Extract component positions from PCB for 3D interactive view
    components: list[dict[str, object]] = []
    board_size: dict[str, float] | None = None
    if pcb_files:
        try:
            from kicad_pipeline.sexp.parser import parse_file

            tree = parse_file(str(pcb_files[0]))
            # Board outline from gr_rect or gr_line with Edge.Cuts
            for node in tree:
                if not isinstance(node, list):
                    continue
                if len(node) >= 2 and node[0] == "general":
                    # No direct size — derive from outline below
                    pass

            # Get actual board outline from Edge.Cuts lines
            edge_xs: list[float] = []
            edge_ys: list[float] = []
            for node in tree:
                if not isinstance(node, list) or len(node) < 2:
                    continue
                if node[0] == "gr_line":
                    has_edge = any(
                        isinstance(s, list) and s[0] == "layer"
                        and len(s) > 1 and "Edge" in str(s[1])
                        for s in node
                    )
                    if has_edge:
                        for sub in node:
                            if (
                                isinstance(sub, list)
                                and sub[0] in ("start", "end")
                                and len(sub) >= 3
                            ):
                                edge_xs.append(float(sub[1]))
                                edge_ys.append(float(sub[2]))

            comp_xs: list[float] = []
            comp_ys: list[float] = []
            for node in tree:
                if not isinstance(node, list) or len(node) < 2:
                    continue
                if node[0] == "footprint":
                    fp_x, fp_y, fp_rot = 0.0, 0.0, 0.0
                    fp_ref = ""
                    fp_w, fp_h = 3.0, 2.0  # defaults
                    fp_type = "other"
                    for sub in node:
                        if not isinstance(sub, list):
                            continue
                        if sub[0] == "at" and len(sub) >= 3:
                            fp_x = float(sub[1])
                            fp_y = float(sub[2])
                            if len(sub) >= 4:
                                fp_rot = float(sub[3])
                        elif sub[0] == "property" and len(sub) >= 3:
                            if sub[1] == "Reference":
                                fp_ref = str(sub[2])
                        elif sub[0] == "fp_name" and len(sub) >= 2:
                            name_lower = str(sub[1]).lower()
                            ic_kw = ("qfp", "qfn", "bga", "soic", "ssop", "tqfp")
                            pas_kw = ("_0402", "_0603", "_0805", "_1206", "r_", "c_")
                            con_kw = ("conn", "usb", "rj45", "header", "terminal")
                            if any(k in name_lower for k in ic_kw):
                                fp_type = "ic"
                            elif any(k in name_lower for k in pas_kw):
                                fp_type = "passive"
                                fp_w, fp_h = 2.0, 1.2
                            elif any(k in name_lower for k in con_kw):
                                fp_type = "connector"
                                fp_w, fp_h = 5.0, 8.0
                    # Estimate size from pads
                    pad_xs: list[float] = []
                    pad_ys: list[float] = []
                    for sub in node:
                        if isinstance(sub, list) and sub[0] == "pad":
                            for psub in sub:
                                if isinstance(psub, list) and psub[0] == "at" and len(psub) >= 3:
                                    pad_xs.append(float(psub[1]))
                                    pad_ys.append(float(psub[2]))
                    if pad_xs and pad_ys:
                        fp_w = max(pad_xs) - min(pad_xs) + 2.0
                        fp_h = max(pad_ys) - min(pad_ys) + 2.0

                    comp_xs.append(fp_x)
                    comp_ys.append(fp_y)
                    if fp_ref:
                        components.append({
                            "ref": fp_ref,
                            "x": fp_x,
                            "y": fp_y,
                            "width": round(fp_w, 1),
                            "height": round(fp_h, 1),
                            "rotation": fp_rot,
                            "type": fp_type,
                        })

            # Use actual board outline if found, fall back to component bounds
            if edge_xs and edge_ys:
                board_size = {
                    "width": max(edge_xs) - min(edge_xs),
                    "height": max(edge_ys) - min(edge_ys),
                    "origin_x": min(edge_xs),
                    "origin_y": min(edge_ys),
                }
            elif comp_xs and comp_ys:
                margin = 10.0
                board_size = {
                    "width": max(comp_xs) - min(comp_xs) + margin * 2,
                    "height": max(comp_ys) - min(comp_ys) + margin * 2,
                    "origin_x": min(comp_xs) - margin,
                    "origin_y": min(comp_ys) - margin,
                }
        except Exception:
            pass

    return BoardDetailSchema(
        name=board_name,
        images=images,
        crops=crops,
        score=score_data,
        files=files,
        pcb_file_url=f"/api/evidence/pcb-clean/{board_name}.kicad_pcb" if any(
            f.endswith(".kicad_pcb") for f in files
        ) else "",
        sch_file_url=next(
            (f"/api/boards/{board_name}/{f}" for f in files if f.endswith(".kicad_sch")),
            "",
        ),
        has_pcb=any(f.endswith(".kicad_pcb") for f in files),
        has_schematic=any(f.endswith(".kicad_sch") for f in files),
        has_requirements="requirements.json" in files,
        board_size=board_size,
        components=components,
    )


@router.get("/pcb-clean/{board_name}.kicad_pcb")
async def get_clean_pcb(board_name: str, output_dir: str = "output") -> str:
    """Serve a cleaned .kicad_pcb optimised for KiCanvas viewing.

    Fixes:
    1. Hides footprint/description/datasheet text that clutters the view
    2. Strips library prefixes from footprint IDs
    3. Ratsnest data is preserved (it's in the net assignments on pads)
    """
    import re

    from fastapi.responses import Response

    board_dir = Path(output_dir) / board_name
    pcb_files = list(board_dir.glob("*.kicad_pcb"))
    if not pcb_files:
        raise HTTPException(404, f"No PCB file for {board_name}")

    content = pcb_files[0].read_text(encoding="utf-8")

    # Only strip library prefixes — no other modifications
    # KiCanvas is sensitive to file changes; blank values or size changes break it
    content = re.sub(
        r'\(footprint\s+"[^":]*:([^"]+)"',
        r'(footprint "\1"',
        content,
    )
    # Also strip library prefix from Footprint property values
    content = re.sub(
        r'(property\s+"Footprint"\s+")[^":]*:([^"]+)"',
        r'\1\2"',
        content,
    )

    # Inject ratsnest lines as gr_line on Dwgs.User layer
    # KiCanvas renders these as visible lines showing unrouted connections
    ratsnest_lines = _compute_ratsnest(pcb_files[0])
    if ratsnest_lines:
        # Insert before the final closing paren
        insert_pos = content.rfind(")")
        if insert_pos > 0:
            content = (
                content[:insert_pos]
                + "\n  ; Ratsnest (airwires)\n"
                + "\n".join(ratsnest_lines)
                + "\n"
                + content[insert_pos:]
            )

    return Response(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'inline; filename="{board_name}.kicad_pcb"',
        },
    )


def _compute_ratsnest(pcb_path: Path) -> list[str]:
    """Compute minimum spanning tree ratsnest lines for unrouted nets."""
    import math

    from kicad_pipeline.sexp.parser import parse_file

    tree = parse_file(str(pcb_path))

    # Collect pad positions per net
    nets: dict[str, list[tuple[float, float]]] = {}
    for node in tree:
        if not isinstance(node, list) or node[0] != "footprint":
            continue
        fp_x, fp_y, fp_rot = 0.0, 0.0, 0.0
        for sub in node:
            if isinstance(sub, list) and sub[0] == "at" and len(sub) >= 3:
                fp_x, fp_y = float(sub[1]), float(sub[2])
                if len(sub) >= 4:
                    fp_rot = float(sub[3])
        for sub in node:
            if not isinstance(sub, list) or sub[0] != "pad":
                continue
            pad_x, pad_y = 0.0, 0.0
            net_name = ""
            for psub in sub:
                if isinstance(psub, list):
                    if psub[0] == "at" and len(psub) >= 3:
                        pad_x, pad_y = float(psub[1]), float(psub[2])
                    elif psub[0] == "net" and len(psub) >= 3:
                        net_num = int(psub[1])
                        if net_num > 0:
                            net_name = str(psub[2])
            if net_name:
                # Rotate pad position by footprint rotation
                rad = math.radians(-fp_rot)
                rx = pad_x * math.cos(rad) - pad_y * math.sin(rad)
                ry = pad_x * math.sin(rad) + pad_y * math.cos(rad)
                world_x = fp_x + rx
                world_y = fp_y + ry
                nets.setdefault(net_name, []).append((world_x, world_y))

    # For each net with 2+ pads, compute MST (minimum spanning tree)
    lines: list[str] = []
    for _net_name, pads in nets.items():
        if len(pads) < 2:
            continue
        # Simple greedy MST (Prim's algorithm)
        connected = {0}
        remaining = set(range(1, len(pads)))
        while remaining:
            best_dist = float("inf")
            best_from = 0
            best_to = 0
            for ci in connected:
                for ri in remaining:
                    dx = pads[ci][0] - pads[ri][0]
                    dy = pads[ci][1] - pads[ri][1]
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < best_dist:
                        best_dist = d
                        best_from = ci
                        best_to = ri
            connected.add(best_to)
            remaining.discard(best_to)
            x1, y1 = pads[best_from]
            x2, y2 = pads[best_to]
            lines.append(
                f'  (gr_line (start {x1:.3f} {y1:.3f}) '
                f'(end {x2:.3f} {y2:.3f}) '
                f'(stroke (width 0.15) (type dash)) '
                f'(layer "Dwgs.User"))'
            )

    return lines


@router.get("/ledger/{board_name}")
async def get_ledger(board_name: str, output_dir: str = "output") -> dict:
    """Get full evidence ledger for a board."""
    from kicad_pipeline.evidence.ledger import load_ledger

    pcb_path = Path(output_dir) / board_name / f"{board_name}.kicad_pcb"
    if not pcb_path.exists():
        raise HTTPException(404, f"Board not found: {board_name}")

    ledger = load_ledger(pcb_path)
    return ledger.model_dump()


@router.post("/submit")
async def submit_evidence(
    body: EvidenceSubmitSchema,
    output_dir: str = "output",
) -> dict:
    """Submit an evidence record."""
    from kicad_pipeline.evidence.ledger import append_record
    from kicad_pipeline.evidence.models import EvidenceKind, EvidenceRecord

    pcb_path = Path(output_dir) / body.board / f"{body.board}.kicad_pcb"
    if not pcb_path.exists():
        raise HTTPException(404, f"Board not found: {body.board}")

    try:
        kind = EvidenceKind(body.kind)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid evidence kind: {body.kind}") from exc

    record = EvidenceRecord(
        kind=kind,
        stage=body.stage,
        step=body.step or "api_submission",
        board=body.board,
        passed=body.passed,
        summary=body.summary,
        feedback=body.feedback,
        producer="api",
    )

    append_record(pcb_path, record)
    return {"status": "ok", "record_id": record.id}


@router.get("/gate/{board_name}/{stage}")
async def check_gate(
    board_name: str,
    stage: str,
    output_dir: str = "output",
) -> dict:
    """Check if a stage gate passes."""
    from kicad_pipeline.evidence.gates import check_gate as _check_gate

    pcb_path = Path(output_dir) / board_name / f"{board_name}.kicad_pcb"
    if not pcb_path.exists():
        raise HTTPException(404, f"Board not found: {board_name}")

    result = _check_gate(pcb_path, stage)
    return result.model_dump()

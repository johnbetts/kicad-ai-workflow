"""Component review admin endpoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

def _project_root() -> Path:
    """Find project root by walking up from this file."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    return Path.cwd()

_REGISTRY_PATH = _project_root() / "data" / "component_registry.json"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ComponentSummary(BaseModel):
    component_id: str
    ref: str
    value: str
    footprint_id: str
    description: str
    verification_status: str  # "verified", "failed", "pending", "unknown"
    expected_pads: int
    known_issues_count: int


class CheckResultSchema(BaseModel):
    name: str
    passed: bool
    detail: str
    severity: str  # "critical", "major", "minor", "info"


class PinSchema(BaseModel):
    number: str = ""
    name: str = ""
    type: str = ""


class ComponentDetail(BaseModel):
    component_id: str
    ref: str
    value: str
    footprint_id: str
    description: str
    verification_status: str
    last_verified_commit: str
    expected_pads: int
    expected_pad_type: str
    body_width_mm: float
    body_height_mm: float
    model_rotation_z: float = 0.0
    model_offset_xy_max_mm: float = 0.0
    pins: list[PinSchema] = []
    known_issues: list[dict[str, Any]]
    verified_fixes: list[dict[str, Any]]
    checks: list[CheckResultSchema]
    images: dict[str, str]  # view_name -> URL path
    # Enriched fields (computed at response time)
    lcsc_url: str = ""
    datasheet_url: str = ""
    kicad_footprint_lib: str = ""
    model_3d_path: str = ""
    package_category: str = ""  # "passive", "ic", "connector", "discrete", "module"


class ReviewAction(BaseModel):
    action: str  # "approve", "rework", "flag_issue"
    notes: str = ""
    issue_description: str = ""  # For flag_issue


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/list", response_model=list[ComponentSummary])
async def list_components() -> list[ComponentSummary]:
    """List all components in the registry with status."""
    components = _load_components()
    result: list[ComponentSummary] = []
    for comp in components.values():
        result.append(
            ComponentSummary(
                component_id=comp.get("component_id", ""),
                ref=comp.get("ref", ""),
                value=comp.get("value", ""),
                footprint_id=comp.get("footprint_id", ""),
                description=comp.get("description", ""),
                verification_status=comp.get("verification_status", "unknown"),
                expected_pads=comp.get("expected_pads", 0),
                known_issues_count=len(comp.get("known_issues", [])),
            )
        )
    return result


@router.get("/detail/{component_id}", response_model=ComponentDetail)
async def get_component_detail(component_id: str) -> ComponentDetail:
    """Get full component detail with check results and images."""
    components = _load_components()
    comp = components.get(component_id)
    if not comp:
        raise HTTPException(status_code=404, detail=f"Component not found: {component_id}")

    checks = _run_structural_checks(comp)
    images = _find_component_images(component_id)

    # Enrich with computed fields
    fp_id = comp.get("footprint_id", "")
    ref_prefix = comp.get("ref", "X")[0] if comp.get("ref") else "X"
    category = _classify_component(ref_prefix, fp_id)
    pins_raw = comp.get("pins", [])
    pins = [
        PinSchema(
            number=str(p.get("number", "")),
            name=str(p.get("name", "")),
            type=str(p.get("type", "")),
        )
        for p in pins_raw
        if isinstance(p, dict)
    ]

    # Try to find LCSC part number from JLCPCB basic parts
    lcsc_url = ""
    datasheet_url = ""
    try:
        basic_path = _project_root() / "data" / "jlcpcb_basic_parts.json"
        if basic_path.exists():
            with open(basic_path) as f:
                basic_parts = json.load(f)
            val = comp.get("value", "").lower()
            pkg = fp_id.lower().replace("_", "")
            for part in basic_parts:
                desc = str(part.get("description", "")).lower()
                ppkg = str(part.get("package", "")).lower().replace("-", "")
                if val and val in desc and pkg in ppkg:
                    lcsc = part.get("lcsc", "")
                    if lcsc:
                        lcsc_url = f"https://www.lcsc.com/product-detail/{lcsc}.html"
                        break
    except Exception:
        pass

    # 3D model path
    model_3d = ""
    try:
        from kicad_pipeline.pcb.footprints import _3D_MODEL_MAP
        for pattern, model_path in _3D_MODEL_MAP:
            if pattern in fp_id:
                model_3d = model_path
                break
    except Exception:
        pass

    return ComponentDetail(
        component_id=comp.get("component_id", ""),
        ref=comp.get("ref", ""),
        value=comp.get("value", ""),
        footprint_id=fp_id,
        description=comp.get("description", ""),
        verification_status=comp.get("verification_status", "unknown"),
        last_verified_commit=comp.get("last_verified_commit", ""),
        expected_pads=comp.get("expected_pads", 0),
        expected_pad_type=comp.get("expected_pad_type", "smd"),
        body_width_mm=comp.get("body_width_mm") or 0.0,
        body_height_mm=comp.get("body_height_mm") or 0.0,
        model_rotation_z=comp.get("model_rotation_z") or 0.0,
        model_offset_xy_max_mm=comp.get("model_offset_xy_max_mm") or 0.0,
        pins=pins,
        known_issues=comp.get("known_issues", []),
        verified_fixes=comp.get("verified_fixes", []),
        checks=checks,
        images=images,
        lcsc_url=lcsc_url,
        datasheet_url=datasheet_url,
        kicad_footprint_lib=f"kicad-footprints:{fp_id}" if fp_id else "",
        model_3d_path=model_3d,
        package_category=category,
    )


def _classify_component(ref_prefix: str, fp_id: str) -> str:
    """Classify component into a category."""
    prefix_map = {
        "R": "passive", "C": "passive", "L": "passive",
        "D": "discrete", "Q": "discrete", "LED": "discrete",
        "U": "ic", "Y": "discrete",
        "J": "connector", "P": "connector",
        "K": "discrete", "SW": "discrete",
        "H": "mechanical", "TP": "mechanical",
    }
    for p, cat in prefix_map.items():
        if ref_prefix.startswith(p):
            return cat
    fp_lower = fp_id.lower()
    if "module" in fp_lower or "esp32" in fp_lower:
        return "module"
    return "other"


@router.post("/review/{component_id}")
async def review_component(
    component_id: str, body: ReviewAction
) -> dict[str, str]:
    """Submit a review action for a component."""
    components = _load_components()
    comp = components.get(component_id)
    if not comp:
        raise HTTPException(status_code=404, detail=f"Component not found: {component_id}")

    now = _now_iso()

    if body.action == "approve":
        comp["verification_status"] = "verified"
        if body.notes:
            comp.setdefault("verified_fixes", []).append(
                {"description": body.notes, "date": now}
            )
    elif body.action == "rework":
        comp["verification_status"] = "failed"
        if body.notes:
            comp.setdefault("known_issues", []).append(
                {"description": body.notes, "severity": "major", "status": "open", "date": now}
            )
    elif body.action == "flag_issue":
        comp.setdefault("known_issues", []).append(
            {
                "description": body.issue_description or body.notes,
                "severity": "major",
                "status": "open",
                "date": now,
            }
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {body.action}")

    _save_components(components)
    return {"status": "ok", "verification_status": comp["verification_status"]}


@router.post("/verify/{component_id}")
async def trigger_verification(component_id: str) -> dict[str, Any]:
    """Trigger structural verification for a component (no rendering)."""
    components = _load_components()
    comp = components.get(component_id)
    if not comp:
        raise HTTPException(status_code=404, detail=f"Component not found: {component_id}")

    checks = _run_structural_checks(comp)
    all_passed = all(c.passed for c in checks)

    comp["verification_status"] = "verified" if all_passed else "failed"
    _save_components(components)

    return {
        "component_id": component_id,
        "passed": all_passed,
        "checks": [c.model_dump() for c in checks],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_components() -> dict[str, dict[str, Any]]:
    """Load component registry as {component_id: {...}} dict."""
    if not _REGISTRY_PATH.exists():
        return {}
    with open(_REGISTRY_PATH) as f:
        data = json.load(f)
    # Handle both dict-with-components and flat-dict formats
    if isinstance(data, dict) and "components" in data:
        comps = data["components"]
        # components may be a dict (keyed by id) or a list
        if isinstance(comps, list):
            return {c["component_id"]: c for c in comps}
        return comps
    if isinstance(data, list):
        return {c["component_id"]: c for c in data}
    return data


def _save_components(components: dict[str, dict[str, Any]]) -> None:
    """Save component registry, preserving the wrapper format."""
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Read existing to preserve schema_version
    wrapper: dict[str, Any] = {"schema_version": 1, "components": components}
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH) as f:
            existing = json.load(f)
        if isinstance(existing, dict) and "schema_version" in existing:
            wrapper["schema_version"] = existing["schema_version"]
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(wrapper, f, indent=2)
        f.write("\n")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _run_structural_checks(comp: dict[str, Any]) -> list[CheckResultSchema]:
    """Run structural checks on a component spec (registry-only, no footprint file)."""
    checks: list[CheckResultSchema] = []

    # 1: Expected pad count defined
    expected = comp.get("expected_pads", 0)
    checks.append(
        CheckResultSchema(
            name="pad_count_defined",
            passed=expected > 0,
            detail=f"Expected {expected} pads" if expected > 0 else "No expected pad count",
            severity="critical" if expected == 0 else "info",
        )
    )

    # 2: Pad type defined
    pad_type = comp.get("expected_pad_type", "")
    checks.append(
        CheckResultSchema(
            name="pad_type_defined",
            passed=pad_type in ("smd", "thru_hole"),
            detail=f"Type: {pad_type}" if pad_type else "No pad type defined",
            severity="major" if not pad_type else "info",
        )
    )

    # 3: Body dimensions
    bw = comp.get("body_width_mm") or 0
    bh = comp.get("body_height_mm") or 0
    checks.append(
        CheckResultSchema(
            name="body_dimensions",
            passed=bw > 0 and bh > 0,
            detail=f"{bw}x{bh}mm" if bw > 0 and bh > 0 else "Missing dimensions",
            severity="major" if bw == 0 or bh == 0 else "info",
        )
    )

    # 4: Has description
    desc = comp.get("description", "")
    checks.append(
        CheckResultSchema(
            name="has_description",
            passed=len(desc) > 5,
            detail=desc[:60] if desc else "No description",
            severity="minor",
        )
    )

    # 5: Footprint ID set
    fp = comp.get("footprint_id", "")
    checks.append(
        CheckResultSchema(
            name="footprint_id_set",
            passed=bool(fp),
            detail=fp if fp else "No footprint ID",
            severity="critical" if not fp else "info",
        )
    )

    # 6: No open critical issues
    issues = comp.get("known_issues", [])
    open_critical = [
        i for i in issues if i.get("status") == "open" and i.get("severity") == "critical"
    ]
    checks.append(
        CheckResultSchema(
            name="no_open_critical_issues",
            passed=len(open_critical) == 0,
            detail=(
                f"{len(open_critical)} open critical issues"
                if open_critical
                else "No open critical issues"
            ),
            severity="critical" if open_critical else "info",
        )
    )

    # 7: Model rotation defined
    rot = comp.get("model_rotation_z")
    checks.append(
        CheckResultSchema(
            name="model_rotation_defined",
            passed=rot is not None,
            detail=f"Z rotation: {rot} deg" if rot is not None else "No rotation defined",
            severity="minor",
        )
    )

    return checks


def _find_component_images(component_id: str) -> dict[str, str]:
    """Find evidence images for a component."""
    images: dict[str, str] = {}

    # Primary: dedicated evidence directory
    evidence_dir = _project_root() / "data" / "component_evidence" / component_id
    if evidence_dir.exists():
        for png in sorted(evidence_dir.glob("*.png")):
            images[png.stem] = f"/api/component-evidence/{component_id}/{png.name}"
        return images

    # Fallback: output crops matching the ref prefix
    output_dir = _project_root() / "output"
    if not output_dir.exists():
        return images
    prefix = component_id.split("_")[0]  # R, C, U, etc.
    for board_dir in sorted(output_dir.iterdir()):
        if not board_dir.is_dir():
            continue
        crops_dir = board_dir / "crops"
        if not crops_dir.exists():
            continue
        for crop in crops_dir.glob(f"{prefix}*_3d*.png"):
            images[crop.stem] = f"/api/boards/{board_dir.name}/crops/{crop.name}"

    return images

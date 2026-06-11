#!/usr/bin/env python3
"""Calibrate part-rule ``opening_mm`` vectors from isolated single-part renders.

Gate C feedback 2026-06-11 item 1: the TerminalBlock opening was ASSERTED
([0,-1]) instead of measured, so the face_out check confirmed the guess,
not reality.  Every part-class opening must be calibrated the same way
the relay 3D model was: isolated single-part render at rotation 0,
vision measurement of which side the wire/jack entry faces, measured
vector written back with ``"calibrated": true``.

Workflow (three subcommands)::

    python scripts/calibrate_part_openings.py render
        Build one isolation board per part class (rotation 0, board
        center) and render 2D + 3D top + 3D iso views into
        output/calibration/<part>/.

    python scripts/calibrate_part_openings.py prompt
        Print the measurement instructions handed to a vision review
        subagent (the main orchestrator NEVER reads the images itself).

    python scripts/calibrate_part_openings.py apply <verdicts.json>
        Validate the subagent's verdicts and write the measured opening
        vectors into data/part_rules.json with ``calibrated: true``.

Verdict schema (JSON array, one object per part class)::

    [{"part": "TerminalBlock_5.08mm_3P",
      "rule_match": "TerminalBlock",
      "opening_side": "north",        # side the wire entry faces in the
                                       # render: north/south/east/west
      "body_visible": true,            # 3D body rendered at all
      "body_offset_mm": [0.0, 0.0],    # body displacement vs pads (+x right, +y down)
      "evidence": "one sentence"}]

Side -> footprint-frame vector mapping (KiCad coords, Y grows down, the
renders are axis-aligned with the board): north=[0,-1], south=[0,1],
east=[1,0], west=[-1,0].
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

from kicad_pipeline.pcb.isolation_board import build_isolation_board  # noqa: E402
from kicad_pipeline.validation.component_registry import ComponentRegistry  # noqa: E402

#: Part classes whose part-rule opening must be calibrated.
#: rule_match is the ``footprint_contains`` value in data/part_rules.json.
PART_CLASSES: tuple[tuple[str, str], ...] = (
    # (registry component_id, part_rules footprint_contains match)
    ("TerminalBlock_5.08mm_3P", "TerminalBlock"),
    ("TerminalBlock_5.08mm_2P", "TerminalBlock"),  # confirm 2P agrees with 3P
    ("USB-C", "USB"),
    ("RJ45_HR911105A", "RJ45"),
    ("ESP32-S3-WROOM-1", "ESP32"),  # antenna side (drives keepout/edge checks)
)

_SIDE_TO_VECTOR: dict[str, tuple[float, float]] = {
    "north": (0.0, -1.0),
    "south": (0.0, 1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
}

CALIBRATION_DIR = _repo / "output" / "calibration"
PART_RULES_PATH = _repo / "data" / "part_rules.json"

_VIEWS: tuple[tuple[str, list[str]], ...] = (
    ("2d", ["2d", "-w", "1024"]),
    ("3d_top", ["3d", "--view", "top", "-w", "1024", "--height", "768"]),
    ("3d_iso", ["3d", "--view", "iso", "-w", "1024", "--height", "768"]),
)


def _safe_id(component_id: str) -> str:
    return component_id.replace("/", "_").replace(" ", "_")


def render_all() -> dict[str, dict[str, Path]]:
    """Build + render an isolation board per part class; return render map."""
    registry = ComponentRegistry()
    renders: dict[str, dict[str, Path]] = {}
    for component_id, _match in PART_CLASSES:
        spec = registry.get(component_id)
        if spec is None:
            print(f"  SKIP {component_id}: not in component registry")
            continue
        out_dir = CALIBRATION_DIR / _safe_id(component_id)
        pcb_path = build_isolation_board(spec, out_dir)
        part_renders: dict[str, Path] = {}
        for view_name, args in _VIEWS:
            out_png = out_dir / f"{_safe_id(component_id)}_{view_name}.png"
            cmd = ["kicad-image-gen", args[0], str(pcb_path), *args[1:], "-o", str(out_png)]
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print(f"  WARNING: {component_id} {view_name} render failed: {exc}")
                continue
            if out_png.exists() and out_png.stat().st_size > 1000:
                part_renders[view_name] = out_png
        renders[component_id] = part_renders
        print(f"  {component_id}: {len(part_renders)}/{len(_VIEWS)} views -> {out_dir}")
    return renders


def measurement_prompt() -> str:
    """The instructions block handed to the vision measurement subagent."""
    parts = []
    for component_id, match in PART_CLASSES:
        out_dir = CALIBRATION_DIR / _safe_id(component_id)
        views = "\n".join(
            f"    {view}: {out_dir / f'{_safe_id(component_id)}_{view}.png'}"
            for view, _ in _VIEWS
        )
        parts.append(f"- part: {component_id} (rule_match: {match})\n{views}")
    part_list = "\n".join(parts)
    return (
        "Measure connector opening directions from these isolated "
        "single-part renders. Each board has ONE component at rotation 0, "
        "centered. Image axes match KiCad board axes: +X right, +Y DOWN "
        "(so 'north' = top of image = -Y).\n\n"
        f"{part_list}\n\n"
        "For each part answer:\n"
        "1. opening_side: which side does the wire/jack/plug entry face? "
        "(For the ESP32 module: which side is the antenna section on?) "
        "Use the 3D views; the entry is the open mouth of the connector "
        "or the wire holes of a terminal block. One of north/south/east/west.\n"
        "2. body_visible: is a 3D body rendered at all (not just bare pads)?\n"
        "3. body_offset_mm: displacement of the 3D body center vs the pad "
        "field center, in mm (+x right, +y down). Use the known board size "
        "for scale. [0,0] when aligned; null when no body is visible.\n\n"
        "Answer with ONLY a JSON array:\n"
        '[{"part": "...", "rule_match": "...", "opening_side": "north", '
        '"body_visible": true, "body_offset_mm": [0.0, 0.0], '
        '"evidence": "one sentence"}]'
    )


def apply_verdicts(verdicts_path: Path) -> None:
    """Write measured openings into data/part_rules.json (calibrated: true)."""
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    if not isinstance(verdicts, list):
        raise SystemExit("verdicts must be a JSON array")

    # One opening per rule_match — confirm all parts sharing a match agree.
    measured: dict[str, tuple[float, float]] = {}
    for v in verdicts:
        side = v["opening_side"]
        if side not in _SIDE_TO_VECTOR:
            raise SystemExit(f"{v['part']}: invalid opening_side {side!r}")
        vec = _SIDE_TO_VECTOR[side]
        match = v["rule_match"]
        if match in measured and measured[match] != vec:
            raise SystemExit(
                f"rule_match {match!r}: conflicting measurements "
                f"{measured[match]} vs {vec} — calibrate per-footprint rules instead"
            )
        measured[match] = vec
        if not v.get("body_visible", True):
            print(f"  NOTE {v['part']}: 3D body NOT visible — model needs calibration")

    rules_doc = json.loads(PART_RULES_PATH.read_text(encoding="utf-8"))
    updated = 0
    for rule in rules_doc.get("rules", []):
        match = rule.get("match", {}).get("footprint_contains")
        if match in measured:
            rule["opening_mm"] = list(measured[match])
            rule["calibrated"] = True
            updated += 1
    PART_RULES_PATH.write_text(
        json.dumps(rules_doc, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  Updated {updated} rule(s) in {PART_RULES_PATH}")
    for match, vec in sorted(measured.items()):
        print(f"    {match}: opening_mm={list(vec)} (calibrated)")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("render", "prompt", "apply"):
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "render":
        render_all()
    elif cmd == "prompt":
        print(measurement_prompt())
    else:
        if len(sys.argv) != 3:
            raise SystemExit("usage: calibrate_part_openings.py apply <verdicts.json>")
        apply_verdicts(Path(sys.argv[2]))


if __name__ == "__main__":
    main()

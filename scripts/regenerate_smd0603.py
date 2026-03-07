#!/usr/bin/env python3
"""Regenerate the smd-0603 variant PCB from its requirements.json."""
from __future__ import annotations

import json
from pathlib import Path

from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinFunction,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.pcb.builder import build_pcb, write_pcb
from kicad_pipeline.project_file import write_project_file

VARIANT_DIR = Path("/Users/johnbetts/Dropbox/Source/kicad-test/variants/smd-0603")


def _build_pin(p: dict[str, str]) -> Pin:
    return Pin(
        number=p["number"],
        name=p.get("name", ""),
        pin_type=PinType(p["pin_type"]) if "pin_type" in p else PinType.PASSIVE,
        function=PinFunction(p["function"]) if p.get("function") else None,
    )


def load_requirements() -> ProjectRequirements:
    with open(VARIANT_DIR / "requirements.json") as f:
        d = json.load(f)

    comps = tuple(
        Component(
            ref=c["ref"],
            value=c.get("value", ""),
            footprint=c.get("footprint", ""),
            lcsc=c.get("lcsc"),
            pins=tuple(_build_pin(p) for p in c.get("pins", [])),
            description=c.get("description", ""),
        )
        for c in d["components"]
    )

    nets = tuple(
        Net(
            name=n["name"],
            connections=tuple(
                NetConnection(ref=conn["ref"], pin=conn["pin"])
                for conn in n.get("connections", [])
            ),
        )
        for n in d.get("nets", [])
    )

    pb_data = d.get("power_budget", {})
    power_rails = tuple(
        PowerRail(
            name=pr["name"],
            voltage=pr["voltage"],
            current_ma=pr["current_ma"],
            source_ref=pr.get("source_ref", ""),
        )
        for pr in pb_data.get("rails", [])
    )

    features = tuple(
        FeatureBlock(
            name=fb["name"],
            description=fb.get("description", ""),
            components=tuple(fb.get("components", ())),
            nets=tuple(fb.get("nets", ())),
            subcircuits=tuple(fb.get("subcircuits", ())),
        )
        for fb in d.get("features", [])
    )

    mech = d.get("mechanical", {})
    return ProjectRequirements(
        project=ProjectInfo(**d["project"]),
        features=features,
        components=comps,
        nets=nets,
        mechanical=MechanicalConstraints(
            board_width_mm=mech.get("board_width_mm", 65.0),
            board_height_mm=mech.get("board_height_mm", 56.0),
            board_template="RPI_HAT",
        ),
        power_budget=PowerBudget(
            rails=power_rails,
            total_current_ma=pb_data.get("total_current_ma", 0.0),
            notes=tuple(pb_data.get("notes", ())),
        ),
    )


def main() -> None:
    req = load_requirements()
    print(f"Loaded: {len(req.components)} components, {len(req.nets)} nets")

    pcb = build_pcb(req, auto_route=True)
    pcb_path = VARIANT_DIR / "smd-0603.kicad_pcb"
    write_pcb(pcb, str(pcb_path))
    print(f"PCB written to {pcb_path}")

    write_project_file("smd-0603", VARIANT_DIR, drc_exclusions=pcb.drc_exclusions)
    print(f"Project file written to {VARIANT_DIR / 'smd-0603.kicad_pro'}")


if __name__ == "__main__":
    main()

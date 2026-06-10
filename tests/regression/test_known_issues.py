"""Regression tests for known pipeline issues.

Each test maps to an entry in docs/known_issues.md. These tests reproduce
the original bug scenario and verify the fix holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _passive_pins() -> tuple[Pin, ...]:
    return (
        Pin(number="1", name="~", pin_type=PinType.PASSIVE),
        Pin(number="2", name="~", pin_type=PinType.PASSIVE),
    )


def _make_multi_sheet_requirements() -> ProjectRequirements:
    """Build requirements large enough to trigger hierarchical schematic.

    Uses >15 components to trigger multi-sheet mode in build_project_schematics.
    """
    components: list[Component] = []
    nets: list[Net] = []

    # MCU
    mcu_pins = (
        Pin(number="1", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
        Pin(number="2", name="3V3", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
    )
    # Add enough GPIO pins
    gpio_pins: list[Pin] = list(mcu_pins)
    for i in range(3, 20):
        gpio_pins.append(
            Pin(number=str(i), name=f"GPIO{i}", pin_type=PinType.BIDIRECTIONAL,
                function=PinFunction.GPIO)
        )
    components.append(Component(
        ref="U1", value="ESP32-S3-WROOM-1", footprint="ESP32-S3-WROOM-1",
        lcsc="C2913202", pins=tuple(gpio_pins),
    ))

    # Add 16 passives to trigger hierarchical mode
    for i in range(1, 9):
        components.append(Component(
            ref=f"R{i}", value="10k", footprint="R_0805",
            lcsc="C17414", pins=_passive_pins(),
        ))
        net_name = f"NET_R{i}"
        nets.append(Net(
            name=net_name,
            connections=(
                NetConnection(ref="U1", pin=str(i + 2)),
                NetConnection(ref=f"R{i}", pin="1"),
            ),
        ))

    for i in range(1, 9):
        components.append(Component(
            ref=f"C{i}", value="100nF", footprint="C_0805",
            lcsc="C49678", pins=_passive_pins(),
        ))

    nets.append(Net(name="GND", connections=(NetConnection(ref="U1", pin="1"),)))
    nets.append(Net(name="+3V3", connections=(NetConnection(ref="U1", pin="2"),)))

    features = (
        FeatureBlock(
            name="MCU", description="ESP32", components=("U1",),
            nets=("GND", "+3V3"), subcircuits=(),
        ),
        FeatureBlock(
            name="Passives", description="Pull-ups and decoupling",
            components=tuple(f"R{i}" for i in range(1, 9)) + tuple(f"C{i}" for i in range(1, 9)),
            nets=tuple(f"NET_R{i}" for i in range(1, 9)),
            subcircuits=(),
        ),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="regression-test"),
        features=features,
        components=tuple(components),
        nets=tuple(nets),
        power_budget=PowerBudget(
            rails=(PowerRail(name="+3V3", voltage=3.3, current_ma=500.0, source_ref="U1"),),
            total_current_ma=500.0,
            notes=(),
        ),
        mechanical=MechanicalConstraints(board_width_mm=80.0, board_height_mm=60.0),
    )


def _make_simple_requirements() -> ProjectRequirements:
    """Build minimal requirements for flat schematic + PCB."""
    u1 = Component(
        ref="U1", value="ESP32-WROOM-32E", footprint="ESP32-WROOM-32E",
        lcsc="C165948",
        pins=(
            Pin(number="1", name="GND", pin_type=PinType.POWER_IN, function=PinFunction.GND),
            Pin(number="2", name="3V3", pin_type=PinType.POWER_IN, function=PinFunction.VCC),
            Pin(number="3", name="GPIO2", pin_type=PinType.BIDIRECTIONAL,
                function=PinFunction.GPIO),
        ),
    )
    r1 = Component(
        ref="R1", value="10k", footprint="R_0805",
        lcsc="C17414", pins=_passive_pins(),
    )
    d1 = Component(
        ref="D1", value="Green LED", footprint="LED_0805",
        lcsc="C70187",
        pins=(
            Pin(number="1", name="K", pin_type=PinType.PASSIVE),
            Pin(number="2", name="A", pin_type=PinType.PASSIVE),
        ),
    )

    return ProjectRequirements(
        project=ProjectInfo(name="regression-simple"),
        features=(
            FeatureBlock(name="MCU", description="MCU", components=("U1",),
                         nets=("GND", "+3V3"), subcircuits=()),
            FeatureBlock(name="LED", description="LED circuit",
                         components=("D1", "R1"), nets=("LED_OUT",), subcircuits=()),
        ),
        components=(u1, r1, d1),
        nets=(
            Net(name="GND", connections=(NetConnection(ref="U1", pin="1"),)),
            Net(name="+3V3", connections=(NetConnection(ref="U1", pin="2"),)),
            Net(name="LED_OUT", connections=(
                NetConnection(ref="U1", pin="3"),
                NetConnection(ref="R1", pin="1"),
            )),
            Net(name="LED_A", connections=(
                NetConnection(ref="R1", pin="2"),
                NetConnection(ref="D1", pin="2"),
            )),
        ),
        mechanical=MechanicalConstraints(board_width_mm=50.0, board_height_mm=40.0),
    )


# ---------------------------------------------------------------------------
# KI-001: Ref designator shows '?'
# ---------------------------------------------------------------------------


class TestKI001RefDesignatorQuestionMark:
    """Verify that no placed symbol in any schematic sheet has ref '?'.

    Root cause: Sub-sheet sheet_instances path used "/" instead of the
    hierarchical "/{root_uuid}/{sheet_entry_uuid}" path, causing KiCad
    to show '?' for all ref designators in sub-sheets.
    """

    def test_flat_schematic_no_question_mark_refs(self, tmp_path: Path) -> None:
        """Flat schematic: all refs must be fully annotated."""
        from kicad_pipeline.schematic.builder import build_schematic, write_schematic
        from kicad_pipeline.sexp.parser import parse_file
        from kicad_pipeline.validation.consistency import (
            _get_property,
        )

        req = _make_simple_requirements()
        sch = build_schematic(req)
        sch_path = tmp_path / "test.kicad_sch"
        write_schematic(sch, sch_path)

        tree = parse_file(sch_path)
        assert isinstance(tree, list)

        for child in tree:
            if not isinstance(child, list) or not child or child[0] != "symbol":
                continue
            ref = _get_property(child, "Reference")
            if ref and not ref.startswith("#"):
                assert "?" not in ref, f"Ref designator contains '?': {ref}"

    def test_hierarchical_schematic_no_question_mark_refs(
        self, tmp_path: Path,
    ) -> None:
        """Hierarchical schematic: all refs in all sheets must be annotated."""
        from kicad_pipeline.schematic.builder import (
            build_project_schematics,
            write_hierarchical_schematic,
        )
        from kicad_pipeline.sexp.parser import parse_file
        from kicad_pipeline.validation.consistency import (
            _get_property,
        )

        req = _make_multi_sheet_requirements()
        schematics = build_project_schematics(req, hierarchical=True)
        written = write_hierarchical_schematic(schematics, tmp_path, "regression-test")

        for sch_file in written:
            tree = parse_file(sch_file)
            if not isinstance(tree, list):
                continue
            for child in tree:
                if not isinstance(child, list) or not child or child[0] != "symbol":
                    continue
                ref = _get_property(child, "Reference")
                if ref and not ref.startswith("#"):
                    assert "?" not in ref, (
                        f"Ref designator contains '?' in {sch_file.name}: {ref}"
                    )


# ---------------------------------------------------------------------------
# KI-002: Schematic-PCB component desync
# ---------------------------------------------------------------------------


class TestKI002SchematicPCBSync:
    """Verify that schematic and PCB generated from the same requirements
    contain the same set of component refs.

    Root cause: Independent _enrich_requirements() calls could cause
    divergence between schematic and PCB component lists.
    """

    def test_same_refs_in_sch_and_pcb(self, tmp_path: Path) -> None:
        """All component refs must appear in both schematic and PCB."""
        from kicad_pipeline.pcb.builder import build_pcb, write_pcb
        from kicad_pipeline.schematic.builder import build_schematic, write_schematic
        from kicad_pipeline.validation.consistency import (
            extract_pcb_components,
            extract_schematic_components,
        )

        req = _make_simple_requirements()

        # Build and write schematic
        sch = build_schematic(req)
        sch_path = tmp_path / "test.kicad_sch"
        write_schematic(sch, sch_path)

        # Build and write PCB
        pcb = build_pcb(req)
        pcb_path = tmp_path / "test.kicad_pcb"
        write_pcb(pcb, pcb_path)

        sch_comps = extract_schematic_components(sch_path)
        pcb_comps = extract_pcb_components(pcb_path)

        sch_refs = {c.ref for c in sch_comps}
        pcb_refs = {c.ref for c in pcb_comps}

        # Every non-power schematic ref should be in PCB
        for ref in sch_refs:
            assert ref in pcb_refs, f"{ref} in schematic but missing from PCB"

        # Every PCB ref should be in schematic (except mechanical-only
        # components like mounting holes which are PCB-only)
        for ref in pcb_refs:
            if ref.startswith("H"):
                continue  # Mounting holes are mechanical, not in schematic
            assert ref in sch_refs, f"{ref} in PCB but missing from schematic"


# ---------------------------------------------------------------------------
# KI-003: Footprint mismatch SCH↔PCB
# ---------------------------------------------------------------------------


class TestKI003FootprintConsistency:
    """Verify that schematic and PCB footprints match for all components.

    Root cause: Variant remapping could be applied inconsistently between
    schematic and PCB generation.
    """

    def test_footprints_match_via_consistency_check(self, tmp_path: Path) -> None:
        """check_consistency() must pass for same-requirements SCH + PCB."""
        from kicad_pipeline.pcb.builder import build_pcb, write_pcb
        from kicad_pipeline.schematic.builder import build_schematic, write_schematic
        from kicad_pipeline.validation.consistency import check_consistency

        req = _make_simple_requirements()

        sch = build_schematic(req)
        sch_path = tmp_path / "test.kicad_sch"
        write_schematic(sch, sch_path)

        pcb = build_pcb(req)
        pcb_path = tmp_path / "test.kicad_pcb"
        write_pcb(pcb, pcb_path)

        report = check_consistency(sch_path, pcb_path)

        # No footprint mismatch errors
        fp_errors = [
            v for v in report.errors
            if v.rule == "consistency_footprint_mismatch"
        ]
        assert len(fp_errors) == 0, (
            f"Footprint mismatches: {[v.message for v in fp_errors]}"
        )


# ---------------------------------------------------------------------------
# KI-005: Connectors off-board (pad extent past board edge)
# ---------------------------------------------------------------------------


class TestKI005ConnectorPadExtent:
    """Verify all connector pads are within board boundaries after optimization.

    Root cause: _orient_connectors() and phase 3f2 used centroid-based
    sizing that didn't account for asymmetric origins (pin 1) on connectors
    with large pad spans.  Fixed by using pin_map.pad_extent_in_board_space()
    and origin_to_centroid() for all connector placement and clamping.
    """

    def test_all_connector_pads_within_board(self) -> None:
        """After optimization, every connector pad must be within board bounds."""
        from kicad_pipeline.optimization.placement_optimizer import optimize_placement_ee
        from kicad_pipeline.pcb.builder import build_pcb
        from kicad_pipeline.pcb.pin_map import pad_extent_in_board_space
        from tests.integration.test_placement_visual import _build_requirements

        requirements = _build_requirements()
        pcb = build_pcb(
            requirements, auto_route=False, placement_mode="grouped",
            layer_count=4, preserve_routing=False, skip_inner_zones=True,
        )
        pcb_opt, _ = optimize_placement_ee(requirements, pcb)

        # Board bounds
        pts = pcb_opt.outline.polygon
        bx0 = min(p.x for p in pts)
        by0 = min(p.y for p in pts)
        bx1 = max(p.x for p in pts)
        by1 = max(p.y for p in pts)

        for fp in pcb_opt.footprints:
            if not fp.ref.startswith("J"):
                continue
            px0, py0, px1, py1 = pad_extent_in_board_space(
                fp, fp.position.x, fp.position.y, fp.rotation,
            )
            assert px0 >= bx0 - 0.5, (
                f"{fp.ref} pads extend past left edge: {px0:.1f} < {bx0:.1f}"
            )
            assert py0 >= by0 - 0.5, (
                f"{fp.ref} pads extend past top edge: {py0:.1f} < {by0:.1f}"
            )
            assert px1 <= bx1 + 0.5, (
                f"{fp.ref} pads extend past right edge: {px1:.1f} > {bx1:.1f}"
            )
            assert py1 <= by1 + 0.5, (
                f"{fp.ref} pads extend past bottom edge: {py1:.1f} > {by1:.1f}"
            )


# ---------------------------------------------------------------------------
# KI-020: Mounting hole keepouts must have actual NPTH footprints
# ---------------------------------------------------------------------------


class TestKI020MountingHoleFootprints:
    """Verify that mounting hole keepouts always have matching NPTH footprints."""

    def test_mounting_holes_from_requirements(self) -> None:
        """When requirements specify mounting_hole_positions, PCB must contain
        both keepout zones AND NPTH mounting hole footprints (H1, H2, etc.)."""
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_simple_requirements()
        # Add mounting hole positions via mechanical constraints
        reqs = ProjectRequirements(
            project=reqs.project,
            features=reqs.features,
            components=reqs.components,
            nets=reqs.nets,
            power_budget=reqs.power_budget,
            mechanical=MechanicalConstraints(
                board_width_mm=80.0,
                board_height_mm=60.0,
                mounting_hole_positions=((3.5, 3.5), (76.5, 3.5), (76.5, 56.5), (3.5, 56.5)),
                mounting_hole_diameter_mm=3.2,
            ),
        )
        pcb = build_pcb(reqs, auto_route=False)

        # Must have mounting hole footprints
        mh_refs = {fp.ref for fp in pcb.footprints if fp.ref.startswith("H")}
        assert len(mh_refs) >= 4, (
            f"Expected >=4 mounting hole footprints (H1-H4), got {mh_refs}"
        )

        # Each must be NPTH
        for fp in pcb.footprints:
            if not fp.ref.startswith("H"):
                continue
            assert fp.pads, f"{fp.ref} has no pads"
            assert fp.pads[0].pad_type == "np_thru_hole", (
                f"{fp.ref} should be NPTH, got {fp.pads[0].pad_type}"
            )

    def test_default_corners_get_footprints(self) -> None:
        """Even when no explicit positions are given, the default 4-corner
        keepouts must have matching NPTH footprints."""
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_simple_requirements()
        pcb = build_pcb(reqs, auto_route=False)

        mh_refs = {fp.ref for fp in pcb.footprints if fp.ref.startswith("H")}
        assert len(mh_refs) >= 4, (
            f"Expected >=4 default mounting hole footprints, got {mh_refs}"
        )


# ---------------------------------------------------------------------------
# KI-019: Antenna keepout must not be at hardcoded fallback position
# ---------------------------------------------------------------------------


class TestKI019AntennaKeepoutPosition:
    """Verify antenna keepout follows ESP32 placement, not hardcoded corner."""

    def test_no_orphan_antenna_keepout(self) -> None:
        """When ESP32 position is determined by placement optimizer (not
        fixed_positions), antenna keepout must be at the actual placed
        position, not at a hardcoded board corner."""
        from kicad_pipeline.pcb.builder import build_pcb

        reqs = _make_simple_requirements()
        pcb = build_pcb(reqs, auto_route=False)

        # Find ESP32 position
        esp_fp = None
        for fp in pcb.footprints:
            if "ESP32" in (fp.value or "").upper():
                esp_fp = fp
                break

        if esp_fp is None:
            return  # No ESP32 in this test — skip

        # Get board dimensions from outline
        xs = [p.x for p in pcb.outline.polygon]
        ys = [p.y for p in pcb.outline.polygon]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)

        # Check that no keepout zone is at a hardcoded corner unrelated to ESP32
        for ko in pcb.keepouts:
            if not ko.polygon:
                continue
            # Skip mounting hole keepouts
            if hasattr(ko, "tag") and ko.tag == "mounting_hole":
                continue
            # Compute keepout centroid
            kx = sum(p.x for p in ko.polygon) / len(ko.polygon)
            ky = sum(p.y for p in ko.polygon) / len(ko.polygon)

            # If this is a rectangular keepout near a board corner,
            # it should be within 30mm of the ESP32
            is_corner = (
                (kx < 20.0 or kx > bw - 20.0)
                and (ky < 15.0 or ky > bh - 15.0)
            )

            if is_corner:
                dist = ((kx - esp_fp.position.x) ** 2 + (ky - esp_fp.position.y) ** 2) ** 0.5
                assert dist < 30.0, (
                    f"Orphan keepout at ({kx:.1f}, {ky:.1f}) is {dist:.1f}mm from "
                    f"ESP32 at ({esp_fp.position.x:.1f}, {esp_fp.position.y:.1f}) — "
                    f"likely a hardcoded fallback (KI-019)"
                )


def test_relay_training_nets_match_sanyou_pinout() -> None:
    """KI: relay coil/contact nets were swapped onto the wrong pads.

    The SANYOU SRD footprint pinout is pad1=COM, pad2=Coil-, pad3=NO,
    pad4=NC, pad5=Coil+. The relay training board once wired the coil
    drive to pad 4 (the NC contact) and +5V to pad 1 (the COM blade) —
    an electrically dead board that no placement score could catch.
    Both the Component pin list and the Net list must agree with the
    physical pinout.
    """
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_relay_group")
    reqs = mod._build_requirements()

    net_of_pad = {}
    for net in reqs.nets:
        for conn in net.connections:
            if conn.ref == "K1":
                net_of_pad[conn.pin] = net.name
    assert net_of_pad["1"] == "RELAY_COM1", net_of_pad
    assert net_of_pad["2"] == "RELAY_COIL1", net_of_pad
    assert net_of_pad["3"] == "RELAY_NO1", net_of_pad
    assert net_of_pad["4"] == "RELAY_NC1", net_of_pad
    assert net_of_pad["5"] == "+5V_RELAY", net_of_pad

    k1 = next(c for c in reqs.components if c.ref == "K1")
    for pin in k1.pins:
        assert pin.net == net_of_pad[pin.number], (
            f"K1 pin {pin.number}: Component says {pin.net}, "
            f"Net list says {net_of_pad[pin.number]}"
        )

    # Terminal pin order: COM belongs on the CENTER pin (J pin 2) so it
    # routes down the channel midline without crossing NO/NC. The Net
    # list once disagreed with the Component pins here too.
    j_net_of_pad: dict[str, str] = {}
    for net in reqs.nets:
        for conn in net.connections:
            if conn.ref == "J1":
                j_net_of_pad[conn.pin] = net.name
    assert j_net_of_pad["2"] == "RELAY_COM1", j_net_of_pad
    j1 = next(c for c in reqs.components if c.ref == "J1")
    for pin in j1.pins:
        assert pin.net == j_net_of_pad[pin.number], (
            f"J1 pin {pin.number}: Component says {pin.net}, "
            f"Net list says {j_net_of_pad[pin.number]}"
        )

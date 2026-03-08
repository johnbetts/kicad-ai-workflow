"""Tests for hierarchical schematic support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    Net,
    NetConnection,
    Pin,
    PinType,
    ProjectInfo,
    ProjectRequirements,
)
from kicad_pipeline.models.schematic import (
    HierarchicalLabel,
    Point,
    Schematic,
    Sheet,
    SheetPin,
)
from kicad_pipeline.schematic.hierarchical import (
    _sanitize_filename,
    build_hierarchical_schematic,
    build_root_sheet,
    build_sub_sheet,
    classify_nets,
    partition_requirements,
    should_use_hierarchy,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_component(ref: str, pins: tuple[Pin, ...] = ()) -> Component:
    """Create a minimal component for testing."""
    return Component(ref=ref, value="10k", footprint="R_0805", pins=pins)


def _two_feature_requirements() -> ProjectRequirements:
    """Requirements with 2 features, 4+ components, inter-feature net."""
    return ProjectRequirements(
        project=ProjectInfo(name="Test Project"),
        features=(
            FeatureBlock(
                name="Power",
                description="Power supply",
                components=("U1", "C1"),
                nets=("GND", "+3V3", "PWR_OUT"),
                subcircuits=(),
            ),
            FeatureBlock(
                name="MCU",
                description="Microcontroller",
                components=("U2", "R1"),
                nets=("GND", "+3V3", "PWR_OUT", "SPI_CLK"),
                subcircuits=(),
            ),
        ),
        components=(
            Component(
                ref="U1", value="LDO", footprint="SOT-23-5",
                pins=(
                    Pin(number="1", name="IN", pin_type=PinType.POWER_IN),
                    Pin(number="2", name="GND", pin_type=PinType.PASSIVE),
                    Pin(number="3", name="OUT", pin_type=PinType.POWER_OUT),
                ),
            ),
            Component(
                ref="C1", value="100nF", footprint="C_0805",
                pins=(
                    Pin(number="1", name="~", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="~", pin_type=PinType.PASSIVE),
                ),
            ),
            Component(
                ref="U2", value="MCU", footprint="QFP-48",
                pins=(
                    Pin(number="1", name="VCC", pin_type=PinType.POWER_IN),
                    Pin(number="2", name="GND", pin_type=PinType.PASSIVE),
                    Pin(number="3", name="SPI_CLK", pin_type=PinType.OUTPUT),
                    Pin(number="4", name="PWR_CTRL", pin_type=PinType.INPUT),
                ),
            ),
            Component(
                ref="R1", value="10k", footprint="R_0805",
                pins=(
                    Pin(number="1", name="~", pin_type=PinType.PASSIVE),
                    Pin(number="2", name="~", pin_type=PinType.PASSIVE),
                ),
            ),
        ),
        nets=(
            Net(name="GND", connections=(
                NetConnection(ref="U1", pin="2"),
                NetConnection(ref="C1", pin="2"),
                NetConnection(ref="U2", pin="2"),
            )),
            Net(name="+3V3", connections=(
                NetConnection(ref="U1", pin="3"),
                NetConnection(ref="C1", pin="1"),
                NetConnection(ref="U2", pin="1"),
            )),
            Net(name="PWR_OUT", connections=(
                NetConnection(ref="U1", pin="3"),
                NetConnection(ref="U2", pin="4"),
            )),
            Net(name="SPI_CLK", connections=(
                NetConnection(ref="U2", pin="3"),
                NetConnection(ref="R1", pin="1"),
            )),
        ),
    )


def _single_feature_requirements() -> ProjectRequirements:
    """Requirements with 1 feature, below hierarchy threshold."""
    return ProjectRequirements(
        project=ProjectInfo(name="Simple"),
        features=(
            FeatureBlock(
                name="Main",
                description="Main circuit",
                components=("R1", "R2"),
                nets=("GND",),
                subcircuits=(),
            ),
        ),
        components=(
            _simple_component("R1"),
            _simple_component("R2"),
        ),
        nets=(
            Net(name="GND", connections=(
                NetConnection(ref="R1", pin="1"),
                NetConnection(ref="R2", pin="1"),
            )),
        ),
    )


# ---------------------------------------------------------------------------
# should_use_hierarchy tests
# ---------------------------------------------------------------------------


class TestShouldUseHierarchy:
    """Test auto-detection of hierarchy suitability."""

    def test_two_features_enough_components(self) -> None:
        req = _two_feature_requirements()
        assert should_use_hierarchy(req) is True

    def test_single_feature_returns_false(self) -> None:
        req = _single_feature_requirements()
        assert should_use_hierarchy(req) is False

    def test_two_features_few_components_returns_false(self) -> None:
        req = ProjectRequirements(
            project=ProjectInfo(name="Tiny"),
            features=(
                FeatureBlock(
                    name="A", description="", components=("R1",),
                    nets=(), subcircuits=(),
                ),
                FeatureBlock(
                    name="B", description="", components=("R2",),
                    nets=(), subcircuits=(),
                ),
            ),
            components=(
                _simple_component("R1"),
                _simple_component("R2"),
            ),
            nets=(),
        )
        # 2 features but only 2 components (< 4 minimum)
        assert should_use_hierarchy(req) is False

    def test_many_features_returns_true(self) -> None:
        features = tuple(
            FeatureBlock(
                name=f"F{i}", description="", components=(f"R{i}",),
                nets=(), subcircuits=(),
            )
            for i in range(5)
        )
        components = tuple(_simple_component(f"R{i}") for i in range(5))
        req = ProjectRequirements(
            project=ProjectInfo(name="Big"),
            features=features,
            components=components,
            nets=(),
        )
        assert should_use_hierarchy(req) is True


# ---------------------------------------------------------------------------
# classify_nets tests
# ---------------------------------------------------------------------------


class TestClassifyNets:
    """Test net classification into power/intra/inter-feature."""

    def test_power_nets_identified(self) -> None:
        req = _two_feature_requirements()
        power, intra, inter = classify_nets(req)
        assert "GND" in power
        assert "+3V3" in power

    def test_inter_feature_nets_identified(self) -> None:
        req = _two_feature_requirements()
        power, intra, inter = classify_nets(req)
        assert "PWR_OUT" in inter  # connects U1 (Power) and U2 (MCU)

    def test_intra_feature_nets_identified(self) -> None:
        req = _two_feature_requirements()
        power, intra, inter = classify_nets(req)
        # SPI_CLK connects U2 and R1, both in MCU feature
        assert "SPI_CLK" in intra.get("MCU", frozenset())

    def test_single_feature_no_inter(self) -> None:
        req = _single_feature_requirements()
        power, intra, inter = classify_nets(req)
        assert len(inter) == 0


# ---------------------------------------------------------------------------
# partition_requirements tests
# ---------------------------------------------------------------------------


class TestPartitionRequirements:
    """Test requirements splitting by feature."""

    def test_splits_into_correct_features(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        assert "Power" in parts
        assert "MCU" in parts

    def test_components_assigned_correctly(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        power_refs = {c.ref for c in parts["Power"].components}
        mcu_refs = {c.ref for c in parts["MCU"].components}
        assert power_refs == {"U1", "C1"}
        assert mcu_refs == {"U2", "R1"}

    def test_inter_feature_net_included_in_both(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        power_nets = {n.name for n in parts["Power"].nets}
        mcu_nets = {n.name for n in parts["MCU"].nets}
        # PWR_OUT is inter-feature — should appear in both
        assert "PWR_OUT" in power_nets
        assert "PWR_OUT" in mcu_nets

    def test_inter_feature_net_connections_filtered(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        # In Power partition, PWR_OUT should only have U1's connection
        pwr_out_net = next(
            n for n in parts["Power"].nets if n.name == "PWR_OUT"
        )
        refs = {c.ref for c in pwr_out_net.connections}
        assert "U1" in refs
        assert "U2" not in refs


# ---------------------------------------------------------------------------
# build_sub_sheet tests
# ---------------------------------------------------------------------------


class TestBuildSubSheet:
    """Test sub-sheet construction."""

    def test_sub_sheet_has_hierarchical_labels_for_inter_nets(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        power, _, inter = classify_nets(req)
        feature = req.features[1]  # MCU
        sub_sch = build_sub_sheet(feature, parts["MCU"], inter, power)

        assert isinstance(sub_sch, Schematic)
        # PWR_OUT crosses features — should have a hierarchical label
        h_label_names = {hl.text for hl in sub_sch.hierarchical_labels}
        assert "PWR_OUT" in h_label_names

    def test_sub_sheet_has_symbols(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        power, _, inter = classify_nets(req)
        feature = req.features[1]
        sub_sch = build_sub_sheet(feature, parts["MCU"], inter, power)

        # Should have placed symbols for U2 and R1
        refs = {s.ref for s in sub_sch.symbols}
        assert "U2" in refs
        assert "R1" in refs

    def test_sub_sheet_no_label_for_intra_net(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        power, _, inter = classify_nets(req)
        feature = req.features[1]
        sub_sch = build_sub_sheet(feature, parts["MCU"], inter, power)

        h_label_names = {hl.text for hl in sub_sch.hierarchical_labels}
        # SPI_CLK is intra-MCU — should NOT be a hierarchical label
        assert "SPI_CLK" not in h_label_names


# ---------------------------------------------------------------------------
# build_root_sheet tests
# ---------------------------------------------------------------------------


class TestBuildRootSheet:
    """Test root sheet construction."""

    def test_root_has_sheet_entries(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        power, _, inter = classify_nets(req)

        sub_schems: dict[str, Schematic] = {}
        for feat_name, sub_req in parts.items():
            feature = next(fb for fb in req.features if fb.name == feat_name)
            sub_schems[feat_name] = build_sub_sheet(feature, sub_req, inter, power)

        root = build_root_sheet(sub_schems, inter, "test", req)
        assert len(root.sheets) == 2

    def test_root_has_no_components(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        power, _, inter = classify_nets(req)

        sub_schems: dict[str, Schematic] = {}
        for feat_name, sub_req in parts.items():
            feature = next(fb for fb in req.features if fb.name == feat_name)
            sub_schems[feat_name] = build_sub_sheet(feature, sub_req, inter, power)

        root = build_root_sheet(sub_schems, inter, "test", req)
        assert len(root.symbols) == 0

    def test_root_sheet_pins_match_sub_labels(self) -> None:
        req = _two_feature_requirements()
        parts = partition_requirements(req)
        power, _, inter = classify_nets(req)

        sub_schems: dict[str, Schematic] = {}
        for feat_name, sub_req in parts.items():
            feature = next(fb for fb in req.features if fb.name == feat_name)
            sub_schems[feat_name] = build_sub_sheet(feature, sub_req, inter, power)

        root = build_root_sheet(sub_schems, inter, "test", req)

        # Each sheet should have pins matching its sub-sheet's hierarchical labels
        for sheet in root.sheets:
            feat_name = sheet.sheet_name
            sub_sch = sub_schems[feat_name]
            sub_label_names = {hl.text for hl in sub_sch.hierarchical_labels}
            pin_names = {p.name for p in sheet.pins}
            assert pin_names == sub_label_names


# ---------------------------------------------------------------------------
# S-expression tests
# ---------------------------------------------------------------------------


class TestSexpSerialization:
    """Test S-expression output for new node types."""

    def test_hierarchical_label_sexp(self) -> None:
        from kicad_pipeline.schematic.builder import _hierarchical_label_sexp

        hl = HierarchicalLabel(
            text="SPI_CLK",
            shape="output",
            position=Point(x=10.0, y=20.0),
            rotation=0.0,
            uuid="test-uuid-1",
        )
        sexp = _hierarchical_label_sexp(hl)
        assert sexp[0] == "hierarchical_label"
        assert sexp[1] == "SPI_CLK"
        assert sexp[2] == ["shape", "output"]
        assert sexp[3] == ["at", 10.0, 20.0, 0]

    def test_sheet_pin_sexp(self) -> None:
        from kicad_pipeline.schematic.builder import _sheet_pin_sexp

        pin = SheetPin(
            name="SPI_CLK",
            pin_type="output",
            position=Point(x=5.0, y=15.0),
            rotation=180.0,
            uuid="test-uuid-2",
        )
        sexp = _sheet_pin_sexp(pin)
        assert sexp[0] == "pin"
        assert sexp[1] == "SPI_CLK"
        assert sexp[2] == "output"

    def test_sheet_sexp(self) -> None:
        from kicad_pipeline.schematic.builder import _sheet_sexp

        sheet = Sheet(
            position=Point(x=20.0, y=30.0),
            size_x=25.0,
            size_y=15.0,
            sheet_name="Power",
            sheet_file="power.kicad_sch",
            pins=(),
            uuid="test-uuid-3",
        )
        sexp = _sheet_sexp(sheet)
        assert sexp[0] == "sheet"
        assert sexp[1] == ["at", 20.0, 30.0]
        assert sexp[2] == ["size", 25.0, 15.0]

    def test_schematic_to_sexp_includes_sheets(self) -> None:
        from kicad_pipeline.schematic.builder import schematic_to_sexp

        sch = Schematic(
            lib_symbols=(),
            symbols=(),
            power_symbols=(),
            wires=(),
            junctions=(),
            no_connects=(),
            labels=(),
            global_labels=(),
            sheets=(Sheet(
                position=Point(x=20.0, y=30.0),
                size_x=25.0,
                size_y=15.0,
                sheet_name="Power",
                sheet_file="power.kicad_sch",
                pins=(),
                uuid="sheet-uuid-1",
            ),),
        )
        sexp = schematic_to_sexp(sch)
        # Find the sheet node
        sheet_nodes = [n for n in sexp if isinstance(n, list) and n[0] == "sheet"]
        assert len(sheet_nodes) == 1

    def test_schematic_to_sexp_includes_hierarchical_labels(self) -> None:
        from kicad_pipeline.schematic.builder import schematic_to_sexp

        sch = Schematic(
            lib_symbols=(),
            symbols=(),
            power_symbols=(),
            wires=(),
            junctions=(),
            no_connects=(),
            labels=(),
            global_labels=(),
            hierarchical_labels=(HierarchicalLabel(
                text="SIG1",
                shape="bidirectional",
                position=Point(x=5.0, y=10.0),
                uuid="hl-uuid-1",
            ),),
        )
        sexp = schematic_to_sexp(sch)
        hl_nodes = [n for n in sexp if isinstance(n, list) and n[0] == "hierarchical_label"]
        assert len(hl_nodes) == 1

    def test_sheet_instances_include_sub_sheets(self) -> None:
        from kicad_pipeline.schematic.builder import schematic_to_sexp

        sch = Schematic(
            lib_symbols=(),
            symbols=(),
            power_symbols=(),
            wires=(),
            junctions=(),
            no_connects=(),
            labels=(),
            global_labels=(),
            sheets=(
                Sheet(
                    position=Point(x=20.0, y=30.0),
                    size_x=25.0,
                    size_y=15.0,
                    sheet_name="Power",
                    sheet_file="power.kicad_sch",
                    pins=(),
                    uuid="sheet-uuid-1",
                ),
            ),
        )
        sexp = schematic_to_sexp(sch)
        # Find sheet_instances node
        si_nodes = [n for n in sexp if isinstance(n, list) and n[0] == "sheet_instances"]
        assert len(si_nodes) == 1
        si = si_nodes[0]
        # Should have root path + sub-sheet path
        path_entries = [n for n in si if isinstance(n, list) and n[0] == "path"]
        assert len(path_entries) == 2  # root + 1 sub-sheet


# ---------------------------------------------------------------------------
# build_hierarchical_schematic (integration)
# ---------------------------------------------------------------------------


class TestBuildHierarchicalSchematic:
    """Integration tests for full hierarchical build."""

    def test_returns_root_plus_sub_sheets(self) -> None:
        req = _two_feature_requirements()
        result = build_hierarchical_schematic(req)
        # Should have root + 2 sub-sheets
        assert len(result) == 3

    def test_root_key_is_project_name(self) -> None:
        req = _two_feature_requirements()
        result = build_hierarchical_schematic(req)
        assert "test_project" in result

    def test_sub_sheet_keys_are_sanitized(self) -> None:
        req = _two_feature_requirements()
        result = build_hierarchical_schematic(req)
        assert "power" in result
        assert "mcu" in result


# ---------------------------------------------------------------------------
# write_hierarchical_schematic (file I/O)
# ---------------------------------------------------------------------------


class TestWriteHierarchicalSchematic:
    """Test multi-file output."""

    def test_writes_multiple_files(self, tmp_path: Path) -> None:
        from kicad_pipeline.schematic.builder import write_hierarchical_schematic

        req = _two_feature_requirements()
        result = build_hierarchical_schematic(req)
        written = write_hierarchical_schematic(result, tmp_path, "Test Project")
        assert len(written) >= 3
        for p in written:
            assert p.exists()
            assert p.suffix == ".kicad_sch"


# ---------------------------------------------------------------------------
# build_project_schematics tests
# ---------------------------------------------------------------------------


class TestBuildProjectSchematics:
    """Test the top-level auto-detecting builder."""

    def test_auto_detects_hierarchy(self) -> None:
        from kicad_pipeline.schematic.builder import build_project_schematics

        req = _two_feature_requirements()
        result = build_project_schematics(req)
        assert len(result) > 1  # hierarchical

    def test_force_flat(self) -> None:
        from kicad_pipeline.schematic.builder import build_project_schematics

        req = _two_feature_requirements()
        result = build_project_schematics(req, hierarchical=False)
        assert len(result) == 1

    def test_auto_detects_flat(self) -> None:
        from kicad_pipeline.schematic.builder import build_project_schematics

        req = _single_feature_requirements()
        result = build_project_schematics(req)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_lowercase_and_underscores(self) -> None:
        assert _sanitize_filename("Power Supply") == "power_supply"

    def test_special_chars_replaced(self) -> None:
        assert _sanitize_filename("ADC/Sensors") == "adc_sensors"

    def test_strips_leading_trailing(self) -> None:
        assert _sanitize_filename("  test  ") == "test"

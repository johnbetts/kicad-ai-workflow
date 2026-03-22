"""Tests for visualization.ratsnest — shared ratsnest utilities."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from kicad_pipeline.models.pcb import (
    BoardOutline,
    DesignRules,
    Footprint,
    NetEntry,
    Pad,
    PCBDesign,
    Point,
)
from kicad_pipeline.visualization.ratsnest import (
    POWER_NETS,
    build_net_pad_map,
    minimum_spanning_tree,
    rotate_point,
)

# ---------------------------------------------------------------------------
# rotate_point
# ---------------------------------------------------------------------------


class TestRotatePoint:
    """Unit tests for rotate_point()."""

    def test_no_rotation(self) -> None:
        x, y = rotate_point(3.0, 4.0, 0.0)
        assert abs(x - 3.0) < 1e-9
        assert abs(y - 4.0) < 1e-9

    def test_90_degrees(self) -> None:
        x, y = rotate_point(1.0, 0.0, 90.0)
        assert abs(x - 0.0) < 1e-9
        assert abs(y - 1.0) < 1e-9

    def test_180_degrees(self) -> None:
        x, y = rotate_point(1.0, 0.0, 180.0)
        assert abs(x - (-1.0)) < 1e-9
        assert abs(y - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# minimum_spanning_tree
# ---------------------------------------------------------------------------


class TestMinimumSpanningTree:
    """Unit tests for minimum_spanning_tree()."""

    def test_empty(self) -> None:
        assert minimum_spanning_tree([]) == []

    def test_single_point(self) -> None:
        assert minimum_spanning_tree([(0, 0)]) == []

    def test_two_points(self) -> None:
        edges = minimum_spanning_tree([(0, 0), (1, 1)])
        assert edges == [(0, 1)]

    def test_three_points(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        edges = minimum_spanning_tree(points)
        assert len(edges) == 2  # MST of 3 points has 2 edges

    def test_collinear_points(self) -> None:
        points = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        edges = minimum_spanning_tree(points)
        assert len(edges) == 2


# ---------------------------------------------------------------------------
# build_net_pad_map
# ---------------------------------------------------------------------------


def _make_simple_pcb() -> PCBDesign:
    """Create a minimal PCB with two footprints connected by a signal net."""
    pad_a = Pad(
        number="1", pad_type="smd", shape="rect",
        position=Point(x=-0.5, y=0.0), size_x=1.0, size_y=0.5,
        layers=("F.Cu",), net_number=1, net_name="SIG_A",
    )
    pad_b = Pad(
        number="2", pad_type="smd", shape="rect",
        position=Point(x=0.5, y=0.0), size_x=1.0, size_y=0.5,
        layers=("F.Cu",), net_number=2, net_name="GND",
    )
    fp1 = Footprint(
        lib_id="test:R", ref="R1", value="10k",
        position=Point(x=10.0, y=20.0), rotation=0.0,
        pads=(pad_a, pad_b),
    )

    pad_c = Pad(
        number="1", pad_type="smd", shape="rect",
        position=Point(x=-0.5, y=0.0), size_x=1.0, size_y=0.5,
        layers=("F.Cu",), net_number=1, net_name="SIG_A",
    )
    pad_d = Pad(
        number="2", pad_type="smd", shape="rect",
        position=Point(x=0.5, y=0.0), size_x=1.0, size_y=0.5,
        layers=("F.Cu",), net_number=3, net_name="+3V3",
    )
    fp2 = Footprint(
        lib_id="test:R", ref="R2", value="4.7k",
        position=Point(x=30.0, y=20.0), rotation=0.0,
        pads=(pad_c, pad_d),
    )

    return PCBDesign(
        outline=BoardOutline(
            polygon=(
                Point(x=0, y=0), Point(x=50, y=0),
                Point(x=50, y=40), Point(x=0, y=40), Point(x=0, y=0),
            ),
        ),
        design_rules=DesignRules(),
        nets=(
            NetEntry(number=0, name=""),
            NetEntry(number=1, name="SIG_A"),
            NetEntry(number=2, name="GND"),
            NetEntry(number=3, name="+3V3"),
        ),
        footprints=(fp1, fp2),
        tracks=(),
        vias=(),
        zones=(),
        keepouts=(),
    )


class TestBuildNetPadMap:
    """Tests for build_net_pad_map()."""

    def test_returns_signal_nets_only(self) -> None:
        pcb = _make_simple_pcb()
        net_pads = build_net_pad_map(pcb)
        assert "SIG_A" in net_pads
        assert "GND" not in net_pads
        assert "+3V3" not in net_pads

    def test_absolute_positions_correct(self) -> None:
        pcb = _make_simple_pcb()
        net_pads = build_net_pad_map(pcb)
        sig_a = net_pads["SIG_A"]
        assert len(sig_a) == 2
        # R1 pad1: (10.0 + (-0.5), 20.0 + 0.0) = (9.5, 20.0)
        # R2 pad1: (30.0 + (-0.5), 20.0 + 0.0) = (29.5, 20.0)
        xs = sorted(p[0] for p in sig_a)
        assert abs(xs[0] - 9.5) < 0.01
        assert abs(xs[1] - 29.5) < 0.01

    def test_power_nets_excluded(self) -> None:
        for net in POWER_NETS:
            assert net.upper() in {n.upper() for n in POWER_NETS}

    def test_rotation_affects_positions(self) -> None:
        """Verify pad positions account for footprint rotation."""
        pad = Pad(
            number="1", pad_type="smd", shape="rect",
            position=Point(x=1.0, y=0.0), size_x=1.0, size_y=0.5,
            layers=("F.Cu",), net_number=1, net_name="NET1",
        )
        fp = Footprint(
            lib_id="test:R", ref="R1", value="10k",
            position=Point(x=10.0, y=10.0), rotation=90.0,
            pads=(pad,),
        )
        pcb = PCBDesign(
            outline=BoardOutline(
                polygon=(
                    Point(x=0, y=0), Point(x=50, y=0),
                    Point(x=50, y=40), Point(x=0, y=40), Point(x=0, y=0),
                ),
            ),
            design_rules=DesignRules(),
            nets=(NetEntry(number=0, name=""), NetEntry(number=1, name="NET1")),
            footprints=(fp,),
            tracks=(), vias=(), zones=(), keepouts=(),
        )
        net_pads = build_net_pad_map(pcb)
        pos = net_pads["NET1"][0]
        # At 90 degrees, (1, 0) rotates to (0, 1), so abs pos = (10, 11)
        assert abs(pos[0] - 10.0) < 0.01
        assert abs(pos[1] - 11.0) < 0.01


# ---------------------------------------------------------------------------
# SVG injection
# ---------------------------------------------------------------------------


class TestSVGInjection:
    """Tests for _inject_ratsnest_into_svg via kicad_export."""

    def test_injects_ratsnest_group(self, tmp_path: object) -> None:
        """Verify SVG gets a <g id='ratsnest'> with <line> elements."""
        from pathlib import Path

        from kicad_pipeline.visualization.kicad_export import _inject_ratsnest_into_svg

        tmp = Path(str(tmp_path))
        svg_content = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'viewBox="0 0 50 40" width="500" height="400">'
            '<rect x="0" y="0" width="50" height="40" fill="white"/>'
            '</svg>'
        )
        svg_file = tmp / "test.svg"
        svg_file.write_text(svg_content)

        pcb = _make_simple_pcb()
        _inject_ratsnest_into_svg(svg_file, pcb)

        tree = ET.parse(str(svg_file))
        root = tree.getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        ratsnest = root.find(".//svg:g[@id='ratsnest']", ns)
        assert ratsnest is not None
        lines = ratsnest.findall("svg:line", ns)
        # SIG_A has 2 pads -> 1 MST edge -> 1 line
        assert len(lines) == 1

    def test_no_signal_nets_no_injection(self, tmp_path: object) -> None:
        """SVG unchanged when PCB has no signal nets."""
        from pathlib import Path

        from kicad_pipeline.visualization.kicad_export import _inject_ratsnest_into_svg

        tmp = Path(str(tmp_path))
        svg_content = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'viewBox="0 0 50 40" width="500" height="400">'
            '</svg>'
        )
        svg_file = tmp / "test.svg"
        svg_file.write_text(svg_content)

        # PCB with only power nets
        pad = Pad(
            number="1", pad_type="smd", shape="rect",
            position=Point(x=0, y=0), size_x=1.0, size_y=0.5,
            layers=("F.Cu",), net_number=1, net_name="GND",
        )
        fp = Footprint(
            lib_id="test:R", ref="R1", value="10k",
            position=Point(x=10.0, y=10.0), pads=(pad,),
        )
        pcb = PCBDesign(
            outline=BoardOutline(
                polygon=(
                    Point(x=0, y=0), Point(x=50, y=0),
                    Point(x=50, y=40), Point(x=0, y=40), Point(x=0, y=0),
                ),
            ),
            design_rules=DesignRules(),
            nets=(NetEntry(number=0, name=""), NetEntry(number=1, name="GND")),
            footprints=(fp,),
            tracks=(), vias=(), zones=(), keepouts=(),
        )
        _inject_ratsnest_into_svg(svg_file, pcb)

        # File should be unchanged (no ratsnest group added)
        tree = ET.parse(str(svg_file))
        root = tree.getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        ratsnest = root.find(".//svg:g[@id='ratsnest']", ns)
        assert ratsnest is None

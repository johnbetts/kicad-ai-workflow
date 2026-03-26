"""Tests for placement constraint property serialization in PCB builder."""

from __future__ import annotations

from kicad_pipeline.models.pcb import Footprint, Pad, Point
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
from kicad_pipeline.pcb.builder import _footprint_sexp, build_pcb
from kicad_pipeline.sexp.writer import write

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_footprint(**kwargs: object) -> Footprint:
    """Return a minimal Footprint with sensible defaults."""
    defaults: dict[str, object] = dict(
        lib_id="Capacitor_SMD:C_0402",
        ref="C1",
        value="100nF",
        position=Point(x=10.0, y=10.0),
        pads=(
            Pad(
                number="1",
                pad_type="smd",
                shape="rect",
                position=Point(x=-0.5, y=0.0),
                size_x=0.5,
                size_y=0.5,
                layers=("F.Cu",),
                net_number=1,
                net_name="+3V3",
            ),
            Pad(
                number="2",
                pad_type="smd",
                shape="rect",
                position=Point(x=0.5, y=0.0),
                size_x=0.5,
                size_y=0.5,
                layers=("F.Cu",),
                net_number=2,
                net_name="GND",
            ),
        ),
    )
    defaults.update(kwargs)
    return Footprint(**defaults)  # type: ignore[arg-type]


def _sexp_text(fp: Footprint) -> str:
    """Render a single footprint to an S-expression string."""
    node = _footprint_sexp(fp)
    return write(node)


def _make_requirements_with_component(comp: Component) -> ProjectRequirements:
    """Build minimal ProjectRequirements around a single component."""
    fb = FeatureBlock(
        name="Test",
        description="Test block",
        components=(comp.ref,),
        nets=("+3V3", "GND"),
        subcircuits=(),
    )
    nets = (
        Net(name="+3V3", connections=(NetConnection(ref=comp.ref, pin="1"),)),
        Net(name="GND", connections=(NetConnection(ref=comp.ref, pin="2"),)),
    )
    return ProjectRequirements(
        project=ProjectInfo(name="SerTest", revision="v0.1"),
        features=(fb,),
        components=(comp,),
        nets=nets,
    )


# ---------------------------------------------------------------------------
# Tests: _fp_optional_properties / sexp emission
# ---------------------------------------------------------------------------


def test_custom_properties_in_pcb_sexp() -> None:
    """Footprint with custom_properties emits (property ...) nodes in sexp."""
    fp = _minimal_footprint(
        custom_properties=(("PlacementGroup", "buck_stage"),),
    )
    text = _sexp_text(fp)
    assert '(property "PlacementGroup" "buck_stage"' in text


def test_empty_custom_properties_no_output() -> None:
    """Footprint with empty custom_properties does not emit extra property nodes."""
    fp_no_props = _minimal_footprint()
    fp_with_props = _minimal_footprint(
        custom_properties=(("PlacementGroup", "power"),),
    )
    text_no = _sexp_text(fp_no_props)
    text_with = _sexp_text(fp_with_props)
    assert "PlacementGroup" not in text_no
    # Verify the with-props variant does contain it (sanity check)
    assert "PlacementGroup" in text_with


def test_multiple_custom_properties() -> None:
    """All three custom placement properties appear in the sexp output."""
    fp = _minimal_footprint(
        custom_properties=(
            ("PlacementGroup", "analog_in"),
            ("PlacementNear", "U2:VIN"),
            ("PlacementOrder", "3"),
        ),
    )
    text = _sexp_text(fp)
    assert '(property "PlacementGroup" "analog_in"' in text
    assert '(property "PlacementNear" "U2:VIN"' in text
    assert '(property "PlacementOrder" "3"' in text


# ---------------------------------------------------------------------------
# Tests: build_pcb propagation from Component → Footprint
# ---------------------------------------------------------------------------


def _make_comp_with_placement(**placement_kwargs: object) -> Component:
    """Create a Component with optional placement fields set."""
    return Component(
        ref="C1",
        value="100nF",
        footprint="C_0402",
        pins=(
            Pin(number="1", name="+", pin_type=PinType.PASSIVE, net="+3V3"),
            Pin(number="2", name="-", pin_type=PinType.PASSIVE, net="GND"),
        ),
        **placement_kwargs,  # type: ignore[arg-type]
    )


def test_placement_properties_propagated_from_component() -> None:
    """Component.placement_group is propagated into Footprint.custom_properties."""
    comp = _make_comp_with_placement(placement_group="buck_stage")
    req = _make_requirements_with_component(comp)
    pcb = build_pcb(req)
    c1 = next(fp for fp in pcb.footprints if fp.ref == "C1")
    assert ("PlacementGroup", "buck_stage") in c1.custom_properties


def test_placement_near_propagated_from_component() -> None:
    """Component.placement_near is propagated into Footprint.custom_properties."""
    comp = _make_comp_with_placement(placement_near="U1:VIN", placement_near_max_mm=5.0)
    req = _make_requirements_with_component(comp)
    pcb = build_pcb(req)
    c1 = next(fp for fp in pcb.footprints if fp.ref == "C1")
    assert ("PlacementNear", "U1:VIN") in c1.custom_properties
    assert ("PlacementNearMaxMM", "5.0") in c1.custom_properties


def test_placement_order_propagated_from_component() -> None:
    """Component.placement_order is propagated as a string in custom_properties."""
    comp = _make_comp_with_placement(placement_order=2)
    req = _make_requirements_with_component(comp)
    pcb = build_pcb(req)
    c1 = next(fp for fp in pcb.footprints if fp.ref == "C1")
    assert ("PlacementOrder", "2") in c1.custom_properties


def test_all_placement_properties_appear_in_sexp() -> None:
    """All four placement fields on a Component are serialized into sexp."""
    comp = _make_comp_with_placement(
        placement_group="analog_in",
        placement_near="U2:AIN0",
        placement_order=1,
        placement_near_max_mm=3.5,
    )
    req = _make_requirements_with_component(comp)
    pcb = build_pcb(req)
    c1 = next(fp for fp in pcb.footprints if fp.ref == "C1")
    text = _sexp_text(c1)
    assert '(property "PlacementGroup" "analog_in"' in text
    assert '(property "PlacementNear" "U2:AIN0"' in text
    assert '(property "PlacementOrder" "1"' in text
    assert '(property "PlacementNearMaxMM" "3.5"' in text


def test_component_without_placement_has_no_extra_props() -> None:
    """Component with no placement fields produces no placement custom_properties."""
    comp = _make_comp_with_placement()
    req = _make_requirements_with_component(comp)
    pcb = build_pcb(req)
    c1 = next(fp for fp in pcb.footprints if fp.ref == "C1")
    placement_keys = {k for k, _ in c1.custom_properties}
    assert "PlacementGroup" not in placement_keys
    assert "PlacementNear" not in placement_keys
    assert "PlacementOrder" not in placement_keys
    assert "PlacementNearMaxMM" not in placement_keys

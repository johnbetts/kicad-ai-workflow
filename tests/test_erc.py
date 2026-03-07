"""Tests for kicad_pipeline.schematic.erc."""

from __future__ import annotations

from kicad_pipeline.models.schematic import (
    GlobalLabel,
    Label,
    LibPin,
    LibSymbol,
    Point,
    Schematic,
    SymbolInstance,
    Wire,
)
from kicad_pipeline.schematic.erc import (
    ERCReport,
    ERCSeverity,
    ERCViolation,
    run_erc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_schematic() -> Schematic:
    """Return a schematic with no elements."""
    return Schematic(
        lib_symbols=(),
        symbols=(),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )


def _make_sym(
    ref: str,
    value: str = "10k",
    lib_id: str = "Device:R",
    x: float = 0.0,
    y: float = 0.0,
) -> SymbolInstance:
    """Build a minimal :class:`SymbolInstance` for testing."""
    return SymbolInstance(
        lib_id=lib_id,
        ref=ref,
        value=value,
        footprint="R_0805",
        position=Point(x, y),
    )


def _make_global_label(text: str, x: float = 0.0, y: float = 0.0) -> GlobalLabel:
    return GlobalLabel(
        text=text,
        shape="input",
        position=Point(x, y),
    )


# ---------------------------------------------------------------------------
# test_erc_empty_schematic_passes
# ---------------------------------------------------------------------------


def test_erc_empty_schematic_passes() -> None:
    """An empty schematic with no symbols should pass ERC."""
    report = run_erc(_empty_schematic())
    assert report.passed
    assert len(report.errors) == 0


# ---------------------------------------------------------------------------
# duplicate_ref
# ---------------------------------------------------------------------------


def test_erc_no_duplicates_passes() -> None:
    """Schematic with unique refs passes the duplicate_ref check."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(_make_sym("R1"), _make_sym("R2")),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    dup_errors = [v for v in report.errors if v.rule == "duplicate_ref"]
    assert len(dup_errors) == 0


def test_erc_duplicate_ref_is_error() -> None:
    """Two symbols sharing the same ref → ERROR violation with rule 'duplicate_ref'."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(_make_sym("R1", x=0.0), _make_sym("R1", x=10.0)),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    dup_errors = [v for v in report.errors if v.rule == "duplicate_ref"]
    assert len(dup_errors) >= 1
    assert dup_errors[0].ref == "R1"


# ---------------------------------------------------------------------------
# missing_value
# ---------------------------------------------------------------------------


def test_erc_missing_value_is_error() -> None:
    """A symbol with an empty value field → ERROR violation with rule 'missing_value'."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(_make_sym("R1", value=""),),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    val_errors = [v for v in report.errors if v.rule == "missing_value"]
    assert len(val_errors) == 1
    assert val_errors[0].ref == "R1"


# ---------------------------------------------------------------------------
# ERCReport properties
# ---------------------------------------------------------------------------


def test_erc_report_passed_property() -> None:
    """passed is True when there are only WARNING-severity violations."""
    warning = ERCViolation(
        severity=ERCSeverity.WARNING,
        rule="some_warning",
        message="Just a warning",
    )
    report = ERCReport(violations=(warning,))
    assert report.passed is True


def test_erc_report_errors_filter() -> None:
    """errors property returns only ERROR-severity violations."""
    error = ERCViolation(severity=ERCSeverity.ERROR, rule="r1", message="e")
    warning = ERCViolation(severity=ERCSeverity.WARNING, rule="r2", message="w")
    report = ERCReport(violations=(error, warning))
    assert report.errors == (error,)
    assert len(report.errors) == 1


def test_erc_report_warnings_filter() -> None:
    """warnings property returns only WARNING-severity violations."""
    error = ERCViolation(severity=ERCSeverity.ERROR, rule="r1", message="e")
    warning = ERCViolation(severity=ERCSeverity.WARNING, rule="r2", message="w")
    report = ERCReport(violations=(error, warning))
    assert report.warnings == (warning,)
    assert len(report.warnings) == 1


# ---------------------------------------------------------------------------
# unmatched_global_label
# ---------------------------------------------------------------------------


def test_erc_unmatched_global_label_warning() -> None:
    """A single GlobalLabel with a unique text → WARNING violation."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(_make_global_label("SPI_CLK"),),
    )
    report = run_erc(sch)
    gl_warnings = [v for v in report.warnings if v.rule == "unmatched_global_label"]
    assert len(gl_warnings) == 1
    assert "SPI_CLK" in gl_warnings[0].message


def test_erc_matched_global_labels_pass() -> None:
    """Two GlobalLabels with the same text → no unmatched_global_label violation."""
    sch = Schematic(
        lib_symbols=(),
        symbols=(),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(
            _make_global_label("SPI_CLK", x=0.0),
            _make_global_label("SPI_CLK", x=100.0),
        ),
    )
    report = run_erc(sch)
    gl_violations = [v for v in report.violations if v.rule == "unmatched_global_label"]
    assert len(gl_violations) == 0


# ---------------------------------------------------------------------------
# wire_pin_misaligned
# ---------------------------------------------------------------------------


def _make_lib_pin(
    number: str,
    name: str,
    x: float,
    y: float,
    rotation: float = 180.0,
    length: float = 2.54,
) -> LibPin:
    """Build a minimal LibPin."""
    from kicad_pipeline.models.schematic import FontEffect

    return LibPin(
        number=number,
        name=name,
        pin_type="passive",
        at=Point(x, y),
        rotation=rotation,
        length=length,
        name_effects=FontEffect(),
        number_effects=FontEffect(),
    )


def test_erc_wire_pin_aligned_passes() -> None:
    """Wire endpoints matching pin positions → no wire_pin_misaligned violation."""
    # A resistor with 2 pins at x=+2.54 and x=-2.54 (rotation 0/180).
    # Pin at (2.54, 0) with rotation=0 → tip at (2.54+2.54, 0) = (5.08, 0) rel
    # After Y-negation in placement: connection at symbol_x + 5.08, symbol_y + 0
    # Pin at (-2.54, 0) with rotation=180 → tip at (-2.54-2.54, 0) = (-5.08, 0) rel
    lib_r = LibSymbol(
        lib_id="Device:R",
        pins=(
            _make_lib_pin("1", "~", 2.54, 0.0, rotation=0.0, length=2.54),
            _make_lib_pin("2", "~", -2.54, 0.0, rotation=180.0, length=2.54),
        ),
        shapes=(),
    )
    sym = SymbolInstance(
        lib_id="Device:R",
        ref="R1",
        value="10k",
        footprint="R_0805",
        position=Point(100.0, 50.0),
    )
    # Pin 1 tip: (100 + 5.08, 50) = (105.08, 50)
    # Pin 2 tip: (100 - 5.08, 50) = (94.92, 50)
    # Wire from pin 1 tip to a label, and from pin 2 tip to a label
    wire1 = Wire(start=Point(105.08, 50.0), end=Point(120.0, 50.0))
    wire2 = Wire(start=Point(94.92, 50.0), end=Point(80.0, 50.0))
    lbl1 = Label(text="NET1", position=Point(120.0, 50.0))
    lbl2 = Label(text="NET2", position=Point(80.0, 50.0))

    sch = Schematic(
        lib_symbols=(lib_r,),
        symbols=(sym,),
        power_symbols=(),
        wires=(wire1, wire2),
        junctions=(),
        no_connects=(),
        labels=(lbl1, lbl2),
        global_labels=(),
    )
    report = run_erc(sch)
    misaligned = [v for v in report.violations if v.rule == "wire_pin_misaligned"]
    assert len(misaligned) == 0


def test_erc_wire_pin_misaligned_warns() -> None:
    """Wire endpoint not on any pin/label/junction → WARNING."""
    lib_r = LibSymbol(
        lib_id="Device:R",
        pins=(
            _make_lib_pin("1", "~", 2.54, 0.0, rotation=0.0, length=2.54),
            _make_lib_pin("2", "~", -2.54, 0.0, rotation=180.0, length=2.54),
        ),
        shapes=(),
    )
    sym = SymbolInstance(
        lib_id="Device:R",
        ref="R1",
        value="10k",
        footprint="R_0805",
        position=Point(100.0, 50.0),
    )
    # Wire start matches pin 1 tip, but end is in empty space
    wire = Wire(start=Point(105.08, 50.0), end=Point(200.0, 200.0))

    sch = Schematic(
        lib_symbols=(lib_r,),
        symbols=(sym,),
        power_symbols=(),
        wires=(wire,),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    misaligned = [v for v in report.violations if v.rule == "wire_pin_misaligned"]
    assert len(misaligned) == 1
    assert misaligned[0].severity == ERCSeverity.WARNING
    assert "200.0" in misaligned[0].message


# ---------------------------------------------------------------------------
# symbol_overlap
# ---------------------------------------------------------------------------


def test_erc_symbol_overlap_warns() -> None:
    """Two symbols with overlapping bounding boxes → WARNING."""
    lib_r = LibSymbol(
        lib_id="Device:R",
        pins=(
            _make_lib_pin("1", "~", 2.54, 0.0, rotation=0.0, length=2.54),
            _make_lib_pin("2", "~", -2.54, 0.0, rotation=180.0, length=2.54),
        ),
        shapes=(),
    )
    # Place two resistors at the same position — guaranteed overlap
    sym1 = SymbolInstance(
        lib_id="Device:R",
        ref="R1",
        value="10k",
        footprint="R_0805",
        position=Point(100.0, 50.0),
    )
    sym2 = SymbolInstance(
        lib_id="Device:R",
        ref="R2",
        value="4.7k",
        footprint="R_0805",
        position=Point(100.0, 50.0),
    )

    sch = Schematic(
        lib_symbols=(lib_r,),
        symbols=(sym1, sym2),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    overlaps = [v for v in report.violations if v.rule == "symbol_overlap"]
    assert len(overlaps) == 1
    assert overlaps[0].severity == ERCSeverity.WARNING
    assert "R1" in overlaps[0].message
    assert "R2" in overlaps[0].message


def test_erc_symbol_no_overlap_passes() -> None:
    """Well-spaced symbols → no symbol_overlap violation."""
    lib_r = LibSymbol(
        lib_id="Device:R",
        pins=(
            _make_lib_pin("1", "~", 2.54, 0.0, rotation=0.0, length=2.54),
            _make_lib_pin("2", "~", -2.54, 0.0, rotation=180.0, length=2.54),
        ),
        shapes=(),
    )
    sym1 = SymbolInstance(
        lib_id="Device:R",
        ref="R1",
        value="10k",
        footprint="R_0805",
        position=Point(0.0, 0.0),
    )
    # 50mm apart — well beyond any resistor bbox
    sym2 = SymbolInstance(
        lib_id="Device:R",
        ref="R2",
        value="4.7k",
        footprint="R_0805",
        position=Point(50.0, 50.0),
    )

    sch = Schematic(
        lib_symbols=(lib_r,),
        symbols=(sym1, sym2),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    overlaps = [v for v in report.violations if v.rule == "symbol_overlap"]
    assert len(overlaps) == 0


def test_erc_power_symbol_overlap_ignored() -> None:
    """Power symbols at same position → no symbol_overlap violation."""
    lib_r = LibSymbol(
        lib_id="Device:R",
        pins=(
            _make_lib_pin("1", "~", 2.54, 0.0, rotation=0.0, length=2.54),
            _make_lib_pin("2", "~", -2.54, 0.0, rotation=180.0, length=2.54),
        ),
        shapes=(),
    )
    # Power symbols have # prefix — should be excluded from overlap check
    pwr1 = SymbolInstance(
        lib_id="power:+3.3V",
        ref="#PWR01",
        value="+3.3V",
        footprint="",
        position=Point(100.0, 50.0),
    )
    pwr2 = SymbolInstance(
        lib_id="power:GND",
        ref="#PWR02",
        value="GND",
        footprint="",
        position=Point(100.0, 50.0),
    )

    sch = Schematic(
        lib_symbols=(lib_r,),
        symbols=(pwr1, pwr2),
        power_symbols=(),
        wires=(),
        junctions=(),
        no_connects=(),
        labels=(),
        global_labels=(),
    )
    report = run_erc(sch)
    overlaps = [v for v in report.violations if v.rule == "symbol_overlap"]
    assert len(overlaps) == 0

"""Comprehensive tests for the KiCad S-expression writer and parser.

Tests are intentionally written as plain functions (no class grouping) to keep
pytest output easy to read.  Every test covers exactly one behaviour so that
failures are immediately actionable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from kicad_pipeline.sexp.parser import parse, parse_file
from kicad_pipeline.sexp.writer import SExpNode, atom, needs_quotes, write, write_file

# ===========================================================================
# atom() — scalar formatting
# ===========================================================================


def test_atom_bare_string() -> None:
    """A string without spaces or special chars is written bare (no quotes)."""
    assert atom("wire") == "wire"


def test_atom_quoted_string() -> None:
    """A string containing a space is wrapped in double-quotes."""
    assert atom("kicad-ai-pipeline") == "kicad-ai-pipeline"
    result = atom("A4 landscape")
    assert result == '"A4 landscape"'


def test_atom_number_int() -> None:
    """An integer is written as a bare decimal literal."""
    assert atom(42) == "42"
    assert atom(0) == "0"
    assert atom(-7) == "-7"


def test_atom_number_float() -> None:
    """A float is written as a bare decimal literal."""
    assert atom(3.14) == "3.14"
    assert atom(10.16) == "10.16"
    assert atom(0.0) == "0.0"


def test_atom_bool_true() -> None:
    """Python ``True`` is written as the bare atom ``yes``."""
    assert atom(True) == "yes"


def test_atom_bool_false() -> None:
    """Python ``False`` is written as the bare atom ``no``."""
    assert atom(False) == "no"


# ===========================================================================
# needs_quotes()
# ===========================================================================


def test_needs_quotes_no_space() -> None:
    """A plain identifier does not need quoting."""
    assert needs_quotes("wire") is False


def test_needs_quotes_with_space() -> None:
    """A string with a space needs quoting."""
    assert needs_quotes("A4 landscape") is True


def test_needs_quotes_empty() -> None:
    """An empty string needs quoting."""
    assert needs_quotes("") is True


def test_needs_quotes_with_semicolon() -> None:
    """A string with a semicolon (comment char) needs quoting."""
    assert needs_quotes("foo;bar") is True


# ===========================================================================
# write() — list serialisation
# ===========================================================================


def test_simple_list() -> None:
    """A list with a single atom element serialises to ``(wire)``."""
    result = write(["wire"])
    assert result.strip() == "(wire)"


def test_list_atoms_only_on_one_line() -> None:
    """A list whose elements are all atoms is written on one line."""
    node: SExpNode = ["start", 10.16, 10.16]
    result = write(node)
    assert "\n" not in result.strip()
    assert result.strip() == "(start 10.16 10.16)"


def test_nested_list() -> None:
    """Nested lists produce correctly indented multi-line output."""
    node: SExpNode = [
        "wire",
        ["start", 10.16, 10.16],
        ["end", 20.32, 10.16],
    ]
    result = write(node)
    lines = result.splitlines()
    # First line opens the outer list with its keyword
    assert lines[0].startswith("(wire")
    # Sub-lists are indented relative to the parent
    assert any("(start" in line for line in lines)
    assert any("(end" in line for line in lines)
    # Indented lines have leading spaces
    indented = [ln for ln in lines if "(start" in ln or "(end" in ln]
    assert all(ln.startswith(" ") for ln in indented)


def test_write_bool_in_list() -> None:
    """Booleans inside a list are rendered as yes/no."""
    node: SExpNode = ["hide", True]
    assert write(node).strip() == "(hide yes)"


def test_write_quoted_string_in_list() -> None:
    """Strings with spaces inside a list are rendered with double-quotes."""
    node: SExpNode = ["generator", "kicad-ai-pipeline"]
    result = write(node)
    assert result.strip() == "(generator kicad-ai-pipeline)"

    node2: SExpNode = ["paper", "A4 landscape"]
    result2 = write(node2)
    assert '"A4 landscape"' in result2


# ===========================================================================
# Roundtrip: write → parse → compare
# ===========================================================================


def test_roundtrip_simple() -> None:
    """parse(write(node)) == node for a flat list."""
    node: SExpNode = ["wire", 10, 20]
    assert parse(write(node)) == node


def test_roundtrip_nested() -> None:
    """parse(write(node)) == node for a nested list."""
    node: SExpNode = [
        "wire",
        ["start", 10.16, 10.16],
        ["end", 20.32, 10.16],
        ["stroke", ["width", 0], ["type", "default"]],
    ]
    assert parse(write(node)) == node


def test_roundtrip_booleans() -> None:
    """Booleans survive a write → parse roundtrip."""
    node: SExpNode = ["hide", True, False]
    assert parse(write(node)) == node


def test_roundtrip_quoted_string() -> None:
    """Quoted strings survive a write → parse roundtrip."""
    node: SExpNode = ["paper", "A4 landscape"]
    assert parse(write(node)) == node


# ===========================================================================
# Parser — individual feature tests
# ===========================================================================


def test_parser_handles_quotes() -> None:
    """Parser correctly decodes a double-quoted string with spaces."""
    result = parse('(paper "A4 landscape")')
    assert result == ["paper", "A4 landscape"]


def test_parser_handles_escape_sequences() -> None:
    """Parser correctly decodes ``\\`` and ``\"`` escape sequences."""
    result = parse(r'(label "foo\"bar")')
    assert result == ["label", 'foo"bar']

    result2 = parse(r'(path "C:\\Users\\name")')
    assert result2 == ["path", "C:\\Users\\name"]


def test_parser_handles_numbers() -> None:
    """Parser produces int and float values, not strings."""
    result = parse("(at 10 20.5 180)")
    assert result == ["at", 10, 20.5, 180]
    assert isinstance(result, list)
    assert isinstance(result[1], int)
    assert isinstance(result[2], float)
    assert isinstance(result[3], int)


def test_parser_handles_booleans() -> None:
    """``yes`` parses to ``True`` and ``no`` parses to ``False``."""
    result = parse("(hide yes)")
    assert result == ["hide", True]

    result2 = parse("(mirror no)")
    assert result2 == ["mirror", False]


def test_parser_strips_comments() -> None:
    """Semicolon comments are stripped and do not appear in the result."""
    text = """\
; This is a comment
(wire  ; inline comment
  (start 1 2)
)
"""
    result = parse(text)
    assert result == ["wire", ["start", 1, 2]]


def test_parser_handles_empty_list() -> None:
    """An empty list ``()`` parses to an empty Python list."""
    result = parse("()")
    assert result == []


def test_kicad_wire_fragment() -> None:
    """Parse a realistic KiCad wire S-expression fragment."""
    text = """\
(wire (start 10.16 10.16) (end 20.32 10.16)
  (stroke (width 0) (type default))
  (uuid "550e8400-e29b-41d4-a716-446655440000")
)
"""
    result = parse(text)
    assert isinstance(result, list)
    assert result[0] == "wire"

    # Find sub-lists by keyword
    def find(keyword: str, nodes: list[SExpNode]) -> list[SExpNode] | None:
        for n in nodes:
            if isinstance(n, list) and n and n[0] == keyword:
                return n
        return None

    assert isinstance(result, list)
    start = find("start", result)
    assert start == ["start", 10.16, 10.16]

    end = find("end", result)
    assert end == ["end", 20.32, 10.16]

    stroke = find("stroke", result)
    assert stroke is not None
    assert isinstance(stroke, list)
    width = find("width", stroke)
    assert width == ["width", 0]

    uuid_node = find("uuid", result)
    assert uuid_node is not None
    assert uuid_node[1] == "550e8400-e29b-41d4-a716-446655440000"


# ===========================================================================
# File I/O
# ===========================================================================


def test_write_file_and_parse_file(tmp_path: Path) -> None:
    """Roundtrip through the filesystem: write_file → parse_file."""
    node: SExpNode = [
        "kicad_sch",
        ["version", 20231120],
        ["generator", "kicad-ai-pipeline"],
        ["paper", "A4"],
        ["wire", ["start", 10.16, 10.16], ["end", 20.32, 10.16]],
    ]
    dest = tmp_path / "test.kicad_sch"
    write_file(node, dest)

    assert dest.exists()
    recovered = parse_file(dest)
    assert recovered == node


def test_write_file_creates_utf8(tmp_path: Path) -> None:
    """write_file produces a UTF-8 encoded file."""
    node: SExpNode = ["label", "réseau"]
    dest = tmp_path / "utf8.kicad_sch"
    write_file(node, dest)
    raw = dest.read_bytes()
    # é is 0xc3 0xa9 in UTF-8
    assert b"\xc3\xa9" in raw


# ===========================================================================
# Error handling
# ===========================================================================


def test_parse_raises_on_unterminated_list() -> None:
    """SExpParseError is raised for a list missing its closing paren."""
    from kicad_pipeline.exceptions import SExpParseError

    with pytest.raises(SExpParseError):
        parse("(wire (start 1 2)")


def test_parse_raises_on_empty_input() -> None:
    """SExpParseError is raised for completely empty input."""
    from kicad_pipeline.exceptions import SExpParseError

    with pytest.raises(SExpParseError):
        parse("")


def test_parse_raises_on_unexpected_close_paren() -> None:
    """SExpParseError is raised for a stray closing parenthesis."""
    from kicad_pipeline.exceptions import SExpParseError

    with pytest.raises(SExpParseError):
        parse("(wire)) ")

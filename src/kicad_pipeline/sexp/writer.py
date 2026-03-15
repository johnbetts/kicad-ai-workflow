"""KiCad S-expression writer.

Converts the in-memory ``SExpNode`` tree into the textual S-expression format
used by KiCad's ``.kicad_sch`` and ``.kicad_pcb`` files.

KiCad 9 formatting rules
-------------------------
* The first element in each list (the keyword/tag) is written bare.
* All other string values are double-quoted.
* Numbers (``int`` / ``float``): written as bare literals.
* Booleans: ``True`` -> ``yes``, ``False`` -> ``no`` (bare).
* Lists: serialised as ``(keyword "value1" "value2" ...)``.
* Indentation: if a list contains only atoms, it is written on one line.
  If any element is itself a list, sub-lists each occupy their own indented
  line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import SExpWriteError

if TYPE_CHECKING:
    from typing import TypeAlias

# Re-exported so callers can do ``from kicad_pipeline.sexp.writer import SExpWriteError``
__all__ = [
    "SExpNode",
    "SExpWriteError",
    "atom",
    "keyword_atom",
    "needs_quotes",
    "write",
    "write_file",
]

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

# A node is either a scalar atom or a (possibly nested) list of nodes.
SExpNode: TypeAlias = str | int | float | bool | list["SExpNode"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDENT_STEP: int = 2
"""Number of spaces per indentation level."""

_BARE_PATTERN: re.Pattern[str] = re.compile(r"^[^\s()\";]+$")
"""Regex that matches strings that are safe to write without quotes."""

# KiCad S-expression enum/keyword values that should remain bare (unquoted)
# even when they appear as non-first elements of a list.
_BARE_KEYWORDS: frozenset[str] = frozenset({
    # Pin types
    "input", "output", "bidirectional", "tri_state", "passive",
    "free", "unspecified", "power_in", "power_out", "open_collector",
    "open_emitter", "no_connect",
    # Pin styles
    "line", "inverted", "clock", "inverted_clock", "input_low",
    "clock_low", "output_low", "edge_clock_high", "non_logic",
    # Stroke types
    "default", "solid", "dash", "dot", "dash_dot", "dash_dot_dot",
    # Fill types
    "none", "outline", "background", "color",
    # Layer types
    "signal", "power", "mixed", "jumper", "user",
    # Footprint attributes
    "smd", "through_hole", "board_only", "exclude_from_pos_files",
    "exclude_from_bom", "allow_soldermask_bridges",
    # Boolean-like
    "yes", "no",
    # Pad shapes/types
    "circle", "rect", "oval", "trapezoid", "roundrect", "custom",
    "thru_hole", "np_thru_hole", "connect",
    # Via types
    "blind_buried", "micro",
    # Zone fill types ("solid" already in stroke types)
    "hatch",
    # Text justification
    "left", "right", "top", "bottom", "mirror",
    # Zone hatch styles
    "edge", "full",
    # Keepout rules
    "not_allowed", "allowed",
    # Symbol scope (KiCad 10: power symbol visibility)
    "global", "local",
    # Misc
    "hide",
})

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def needs_quotes(s: str) -> bool:
    """Return ``True`` if *s* must be wrapped in double-quotes.

    A string needs quoting when it is empty, or when it contains whitespace,
    parentheses, double-quotes, or semicolons — characters that would
    otherwise confuse a KiCad S-expression parser.

    Args:
        s: The raw string value to test.

    Returns:
        ``True`` if double-quoting is required, ``False`` if the string can
        be written bare.
    """
    if not s:
        return True
    return _BARE_PATTERN.match(s) is None


def _quote_string(s: str) -> str:
    """Wrap a string in double-quotes, escaping as needed."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def atom(value: str | int | float | bool) -> str:
    """Format a single scalar value as a KiCad S-expression atom.

    KiCad 9 quotes most string values. Exceptions:

    * Known enum/keyword values (``passive``, ``line``, ``default``, etc.)
      are written bare.
    * Booleans (``yes``/``no``) are always bare.
    * Numbers are bare.

    Args:
        value: The scalar value to format.

    Returns:
        The formatted atom string.
    """
    # bool must be tested before int (bool is a subclass of int)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Preserve enough precision; strip trailing zeros after the decimal
        # point but always keep at least one digit (e.g. "1.0" not "1.").
        formatted = f"{value:g}"
        # Ensure there is a decimal point so it is distinguishable from int
        # when round-tripped through the parser.
        if "." not in formatted and "e" not in formatted:
            formatted += ".0"
        return formatted
    # str — bare if it's a known KiCad keyword, otherwise quoted
    if value in _BARE_KEYWORDS:
        return value
    return _quote_string(value)


def keyword_atom(value: str | int | float | bool) -> str:
    """Format a keyword (first element of a list) as a bare atom.

    Keywords are the first element in an S-expression list, such as
    ``kicad_sch``, ``version``, ``wire``, ``symbol``, etc. These are
    always written unquoted.

    Args:
        value: The keyword value.

    Returns:
        The formatted keyword string (never quoted).
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return atom(value)
    # str — bare for keywords
    return str(value)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_sublist(nodes: list[SExpNode]) -> bool:
    """Return ``True`` if any element of *nodes* is itself a list."""
    return any(isinstance(n, list) for n in nodes)


def _write_node(node: SExpNode, indent: int, *, is_keyword: bool = False) -> str:
    """Recursively render *node* to a string.

    Args:
        node: The node to render.
        indent: Current indentation level (in number of levels, not spaces).
        is_keyword: If True, strings are written bare (used for the first
            element of a list).

    Returns:
        The rendered string, without a trailing newline.
    """
    if isinstance(node, list):
        return _write_list(node, indent)
    if is_keyword:
        return keyword_atom(node)
    return atom(node)


def _write_list(nodes: list[SExpNode], indent: int) -> str:
    """Render a list node (parenthesised expression).

    The first element (keyword/tag) is written bare; all subsequent string
    values are quoted, matching KiCad 9 output conventions.

    If all elements are atoms (no sub-lists), writes them on one line:
    ``(keyword "value1" "value2")``.

    If any element is a list, sub-lists each occupy their own indented line.

    Args:
        nodes: The list of child nodes.
        indent: Current indentation level.

    Returns:
        The rendered string.
    """
    if not nodes:
        return "()"

    if not _has_sublist(nodes):
        # All atoms — single line
        # First element is keyword (bare), rest are values (quoted if string)
        parts: list[str] = []
        for i, n in enumerate(nodes):
            parts.append(_write_node(n, indent, is_keyword=(i == 0)))
        return "(" + " ".join(parts) + ")"

    # Has at least one sub-list — multi-line layout
    child_indent = indent + 1
    prefix = " " * (_INDENT_STEP * child_indent)

    lines: list[str] = []
    # Collect leading atoms (before first sub-list) to sit on the first line
    first_line_parts: list[str] = []
    remaining: list[SExpNode] = list(nodes)
    atom_index = 0

    while remaining and not isinstance(remaining[0], list):
        first_line_parts.append(
            _write_node(remaining.pop(0), child_indent, is_keyword=(atom_index == 0))
        )
        atom_index += 1

    header = "(" + " ".join(first_line_parts)

    for child in remaining:
        rendered = _write_node(child, child_indent)
        lines.append(prefix + rendered)

    closing_indent = " " * (_INDENT_STEP * indent)
    body = "\n".join(lines)
    return f"{header}\n{body}\n{closing_indent})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write(node: SExpNode, indent: int = 0) -> str:
    """Render *node* to a KiCad S-expression string.

    Args:
        node: The root node to serialise.
        indent: Starting indentation level (default ``0``).

    Returns:
        The S-expression string, terminated with a newline when *node* is a
        list.

    Raises:
        SExpWriteError: If the node tree contains unserialisable values.
    """
    result = _write_node(node, indent)
    # Top-level lists get a trailing newline for file compatibility
    if isinstance(node, list):
        return result + "\n"
    return result


def write_file(node: SExpNode, path: str | Path) -> None:
    """Write *node* as a KiCad S-expression file at *path*.

    The file is written with UTF-8 encoding and Unix line endings.

    Args:
        node: The root node to serialise.
        path: Destination file path (string or :class:`pathlib.Path`).

    Raises:
        SExpWriteError: If the node tree contains unserialisable values.
        OSError: If the file cannot be written.
    """
    text = write(node)
    Path(path).write_text(text, encoding="utf-8")

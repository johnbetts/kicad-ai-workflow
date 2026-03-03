"""KiCad S-expression writer.

Converts the in-memory ``SExpNode`` tree into the textual S-expression format
used by KiCad's ``.kicad_sch`` and ``.kicad_pcb`` files.

KiCad formatting rules
----------------------
* Bare strings (no spaces, no special chars): written without quotes.
* Strings with spaces or special characters: wrapped in double-quotes with
  ``\\`` and ``"`` escaped.
* Numbers (``int`` / ``float``): written as bare literals.
* Booleans: ``True`` → ``yes``, ``False`` → ``no``.
* Lists: serialised as ``(element0 element1 …)``.
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


def atom(value: str | int | float | bool) -> str:
    """Format a single scalar value as a KiCad S-expression atom.

    Formatting rules:

    * ``bool`` → ``"yes"`` or ``"no"`` (checked *before* ``int`` because
      ``bool`` is a subclass of ``int`` in Python).
    * ``int`` / ``float`` → decimal literal, e.g. ``42`` or ``3.14``.
    * ``str`` → bare if safe, otherwise double-quoted with ``\\`` and ``"``
      escaped.

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
    # str
    if needs_quotes(value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_sublist(nodes: list[SExpNode]) -> bool:
    """Return ``True`` if any element of *nodes* is itself a list."""
    return any(isinstance(n, list) for n in nodes)


def _write_node(node: SExpNode, indent: int) -> str:
    """Recursively render *node* to a string.

    Args:
        node: The node to render.
        indent: Current indentation level (in number of levels, not spaces).

    Returns:
        The rendered string, without a trailing newline.
    """
    if isinstance(node, list):
        return _write_list(node, indent)
    # bool must be tested before int (bool is a subclass of int)
    return atom(node)


def _write_list(nodes: list[SExpNode], indent: int) -> str:
    """Render a list node (parenthesised expression).

    If the list is empty, returns ``"()"``.

    If all elements are atoms (no sub-lists), writes them on one line:
    ``(elem0 elem1 elem2)``.

    If any element is a list, the *first* element (typically the keyword) is
    written on the same line as the opening paren, and subsequent sub-lists
    each occupy their own indented line:

    .. code-block:: text

       (wire
         (start 10.16 10.16)
         (end 20.32 10.16)
       )

    Atoms that appear after the first element (before any sub-list) are
    written inline with the keyword.

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
        parts = [_write_node(n, indent) for n in nodes]
        return "(" + " ".join(parts) + ")"

    # Has at least one sub-list — multi-line layout
    child_indent = indent + 1
    prefix = " " * (_INDENT_STEP * child_indent)

    lines: list[str] = []
    # Collect leading atoms (before first sub-list) to sit on the first line
    first_line_parts: list[str] = []
    remaining: list[SExpNode] = list(nodes)

    while remaining and not isinstance(remaining[0], list):
        first_line_parts.append(_write_node(remaining.pop(0), child_indent))

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

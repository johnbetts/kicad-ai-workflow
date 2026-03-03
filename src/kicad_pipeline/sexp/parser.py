"""KiCad S-expression parser.

Converts the textual S-expression format used by KiCad's ``.kicad_sch`` and
``.kicad_pcb`` files back into the in-memory ``SExpNode`` tree.

Supported syntax
----------------
* Parenthesised lists: ``(elem0 elem1 …)``
* Double-quoted strings with ``\\`` and ``"`` escape sequences.
* Bare atoms: unquoted strings, integers, floats, and the booleans ``yes`` /
  ``no``.
* Line comments starting with ``;`` — stripped during tokenisation.

The parser is intentionally permissive: it does not validate KiCad-specific
semantics, only structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import SExpParseError

if TYPE_CHECKING:
    from kicad_pipeline.sexp.writer import SExpNode

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[tuple[str, int]]:
    """Break *text* into a flat list of ``(token, position)`` pairs.

    Token kinds emitted:
    * ``"("`` — open paren
    * ``")"`` — close paren
    * ``"STRING:<value>"`` — double-quoted string (value already unescaped)
    * ``"ATOM:<value>"`` — bare atom

    Comments (``; … \\n``) are silently dropped.

    Args:
        text: Raw S-expression source text.

    Returns:
        Ordered list of ``(token_string, char_position)`` pairs.

    Raises:
        SExpParseError: On unterminated string literals or other lex errors.
    """
    tokens: list[tuple[str, int]] = []
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]

        # Whitespace
        if ch in " \t\r\n":
            i += 1
            continue

        # Comment — skip to end of line
        if ch == ";":
            while i < length and text[i] != "\n":
                i += 1
            continue

        # Open / close paren
        if ch == "(":
            tokens.append(("(", i))
            i += 1
            continue
        if ch == ")":
            tokens.append((")", i))
            i += 1
            continue

        # Double-quoted string
        if ch == '"':
            start = i
            i += 1  # skip opening quote
            buf: list[str] = []
            while i < length:
                c = text[i]
                if c == "\\":
                    i += 1
                    if i >= length:
                        raise SExpParseError(
                            "Unterminated escape sequence in string literal",
                            position=start,
                        )
                    esc = text[i]
                    if esc == "n":
                        buf.append("\n")
                    elif esc == "t":
                        buf.append("\t")
                    elif esc == "r":
                        buf.append("\r")
                    elif esc in ('"', "\\"):
                        buf.append(esc)
                    else:
                        # Pass unknown escapes through verbatim
                        buf.append("\\")
                        buf.append(esc)
                    i += 1
                elif c == '"':
                    i += 1  # skip closing quote
                    break
                else:
                    buf.append(c)
                    i += 1
            else:
                raise SExpParseError(
                    "Unterminated string literal",
                    position=start,
                )
            tokens.append(("STRING:" + "".join(buf), start))
            continue

        # Bare atom — everything up to the next whitespace / paren / quote / semicolon
        start = i
        while i < length and text[i] not in " \t\r\n();\"\\":
            i += 1
        atom_text = text[start:i]
        if atom_text:
            tokens.append(("ATOM:" + atom_text, start))

    return tokens


# ---------------------------------------------------------------------------
# Atom value coercion
# ---------------------------------------------------------------------------


def _coerce_atom(raw: str) -> SExpNode:
    """Convert a bare atom string to its most specific Python type.

    Conversion order:

    1. ``"yes"`` → ``True``, ``"no"`` → ``False``
    2. Integer literal → ``int``
    3. Float literal → ``float``
    4. Anything else → ``str``

    Args:
        raw: The unquoted atom text from the source.

    Returns:
        The coerced Python value.
    """
    if raw == "yes":
        return True
    if raw == "no":
        return False
    # Try integer first (no decimal point, no exponent)
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------


def _parse_tokens(
    tokens: list[tuple[str, int]],
    pos: int,
) -> tuple[SExpNode, int]:
    """Parse a single node starting at token index *pos*.

    Args:
        tokens: Token list from :func:`_tokenise`.
        pos: Index of the token to start parsing at.

    Returns:
        A ``(node, next_pos)`` pair where *next_pos* is the index of the first
        unconsumed token.

    Raises:
        SExpParseError: On unexpected tokens or structural errors.
    """
    if pos >= len(tokens):
        raise SExpParseError("Unexpected end of input", position=None)

    token, char_pos = tokens[pos]

    if token == "(":
        # Parse list
        pos += 1
        children: list[SExpNode] = []
        while pos < len(tokens) and tokens[pos][0] != ")":
            child, pos = _parse_tokens(tokens, pos)
            children.append(child)
        if pos >= len(tokens):
            raise SExpParseError(
                "Unterminated list — missing closing parenthesis",
                position=char_pos,
            )
        pos += 1  # consume ")"
        return children, pos

    if token == ")":
        raise SExpParseError(
            "Unexpected closing parenthesis",
            position=char_pos,
        )

    if token.startswith("STRING:"):
        return token[len("STRING:"):], pos + 1

    if token.startswith("ATOM:"):
        raw = token[len("ATOM:"):]
        return _coerce_atom(raw), pos + 1

    raise SExpParseError(f"Unknown token kind: {token!r}", position=char_pos)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(text: str) -> SExpNode:
    """Parse an S-expression string into an ``SExpNode`` tree.

    If the text contains exactly one top-level expression, that expression is
    returned directly.  If it contains multiple top-level expressions (unusual
    for KiCad files but valid S-expression syntax), they are wrapped in a
    list.

    Args:
        text: Raw S-expression source text (may include comments).

    Returns:
        The root ``SExpNode``.

    Raises:
        SExpParseError: If the text is not valid S-expression syntax.
    """
    tokens = _tokenise(text)
    if not tokens:
        raise SExpParseError("Empty input — no S-expression found")

    roots: list[SExpNode] = []
    pos = 0
    while pos < len(tokens):
        node, pos = _parse_tokens(tokens, pos)
        roots.append(node)

    if len(roots) == 1:
        return roots[0]
    return roots


def parse_file(path: str | Path) -> SExpNode:
    """Parse a KiCad S-expression file into an ``SExpNode`` tree.

    Args:
        path: Path to the file (string or :class:`pathlib.Path`).

    Returns:
        The root ``SExpNode``.

    Raises:
        SExpParseError: If the file content is not valid S-expression syntax.
        OSError: If the file cannot be read.
    """
    text = Path(path).read_text(encoding="utf-8")
    return parse(text)

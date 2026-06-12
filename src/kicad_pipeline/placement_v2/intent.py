"""Draft a :class:`~kicad_pipeline.models.requirements.BoardIntent`.

Council verdict 2026-06-11: **inference proposes, declaration
governs.** This module is the "inference proposes" half: it derives a
draft board intent from the netlist and part classes; the human edits
and confirms it ONCE, the confirmed intent is persisted in
requirements (``ProjectRequirements.board_intent``), and Gate A
enforces it on every build. Misfires here are cheap — a draft is a
proposal, never silently applied.

Derivation rules (the gap-analysis edge-cohort defaults):

* Every TerminalBlock-class connector joins the **field_wiring**
  cohort; field wiring claims ONE edge (``north`` by default — the
  human's reference design runs terminals along the top).
* Every other connector (USB, RJ45, SD, pin headers) and every
  part-rule edge-pinned module (ESP32 antenna) joins the **io**
  cohort on the OPPOSITE edge (``south``).
* Pin-assignment freedom is never guessed: ``pins_interchangeable``
  stays ``None`` (undeclared) in drafts — declaring a pinout a fixed
  contract is the human's call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kicad_pipeline.models.requirements import BoardIntent, ConnectorIntent
from kicad_pipeline.placement_v2.part_rules import ref_alpha_prefix

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kicad_pipeline.models.requirements import Component, ProjectRequirements

#: Footprint tokens marking field-wiring (screwdriver-and-cable) parts.
_FIELD_WIRING_TOKENS = ("terminalblock", "screw")
#: Footprint tokens marking connectors at all (same set the compiler uses).
_CONNECTOR_FP_TOKENS = ("usb", "rj45", "terminalblock", "pinheader", "conn")
#: Default cohort edges; the human may move either cohort wholesale —
#: which physical edge is free is board-specific, opposition is not.
FIELD_WIRING_EDGE = "north"
IO_EDGE = "south"


def _is_connector(comp: Component) -> bool:
    fp = comp.footprint.lower()
    return ref_alpha_prefix(comp.ref) == "J" or any(
        t in fp for t in _CONNECTOR_FP_TOKENS
    )


def draft_board_intent(
    requirements: ProjectRequirements,
    extra_io_refs: Iterable[str] = (),
) -> BoardIntent:
    """Derive a draft intent: field-wiring vs IO edge cohorts.

    Args:
        requirements: The project requirements (components + nets).
        extra_io_refs: Non-connector refs that still claim an edge and
            belong with the IO cohort — typically part-rule edge-pinned
            modules (the ESP32's antenna end). The caller resolves
            those from part rules; this module stays rule-file-free.

    Returns:
        A draft :class:`BoardIntent` for human confirmation. Every
        existing hand-authored lock is the regression target for what
        this draft must derive (council: locks = compiler tests).
    """
    entries: dict[str, ConnectorIntent] = {}
    for comp in requirements.components:
        if not _is_connector(comp):
            continue
        fp = comp.footprint.lower()
        field_wiring = any(t in fp for t in _FIELD_WIRING_TOKENS)
        entries[comp.ref] = ConnectorIntent(
            ref=comp.ref,
            edge=FIELD_WIRING_EDGE if field_wiring else IO_EDGE,
        )
    for ref in extra_io_refs:
        entries.setdefault(ref, ConnectorIntent(ref=ref, edge=IO_EDGE))
    return BoardIntent(connectors=tuple(
        entries[ref] for ref in sorted(entries)
    ))


__all__ = ["FIELD_WIRING_EDGE", "IO_EDGE", "draft_board_intent"]

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

from kicad_pipeline.models.requirements import (
    BoardIntent,
    ConnectorIntent,
    IsolationRegionIntent,
)
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


#: Minimum components on an isolated rail for it to draft a region —
#: a buck's switch node also looks "private" but feeds 1-2 parts.
_REGION_MIN_MEMBERS = 3
#: Value tokens marking a FERRITE bead (vs a power inductor, whose
#: henry-valued L bridges a regulator's switch node, not a domain).
_FERRITE_VALUE_TOKENS = ("ferrite", "bead", "blm")


def draft_isolation_regions(
    requirements: ProjectRequirements,
) -> tuple[IsolationRegionIntent, ...]:
    """Derive draft isolation regions from ferrite-separated rail subtrees.

    A boundary ferrite is an L-prefix FERRITE-valued part bridging a
    MAIN rail (a source-fed or canonical power/ground net) to a
    PRIVATE rail fed only through it (L3: +5V -> RELAY_5V; L4:
    GND -> AGND). The private rail's components seed a region;
    detected subcircuits close over the seeds (a relay on RELAY_5V
    pulls its whole driver cell in); rails whose member sets overlap
    merge into one region (RELAY_5V+RELAY_GND, AGND+AVCC — supply
    and return ferrites bound the same domain). Buck inductors
    self-exclude twice over: henry values are not ferrite tokens, and
    a switch node carries the regulator's OUTPUT pin.

    A DRAFT for human confirmation — filter ferrites in odd topologies
    can misfire, which is exactly why silent auto-zoning was rejected
    (council 2026-06-11).
    """
    from kicad_pipeline.models.requirements import PinType
    from kicad_pipeline.optimization.functional_grouper import detect_subcircuits

    by_ref = {c.ref: c for c in requirements.components}
    net_conns: dict[str, list[tuple[str, str]]] = {}
    for net in requirements.nets:
        net_conns[net.name] = [(c.ref, c.pin) for c in net.connections]

    def _is_main_rail(name: str) -> bool:
        # Source-fed, or canonically named (+5V, GND). VCC/VDD tokens
        # are deliberately NOT main markers: AVCC is exactly the kind
        # of isolated rail this function must classify as private.
        upper = name.upper()
        if name.startswith("+") or upper == "GND" or upper.startswith("GND"):
            return True
        for ref, pin_no in net_conns.get(name, ()):
            comp = by_ref.get(ref)
            pin = comp.get_pin(pin_no) if comp is not None else None
            if pin is not None and pin.pin_type in (
                PinType.POWER_OUT, PinType.OUTPUT,
            ):
                return True
        return False

    # rail name -> (members, boundary ferrites)
    rails: dict[str, tuple[set[str], set[str]]] = {}
    for comp in requirements.components:
        if ref_alpha_prefix(comp.ref) != "L":
            continue
        if not any(t in comp.value.lower() for t in _FERRITE_VALUE_TOKENS):
            continue
        nets = [n for n, conns in net_conns.items()
                if any(r == comp.ref for r, _ in conns)]
        if len(nets) != 2:
            continue
        mains = [n for n in nets if _is_main_rail(n)]
        if len(mains) != 1:
            continue  # rail-to-rail or floating bridge: not a boundary
        private = nets[0] if nets[1] == mains[0] else nets[1]
        members = {r for r, _ in net_conns[private] if r != comp.ref}
        if len(members) < _REGION_MIN_MEMBERS:
            continue
        prior = rails.get(private, (set(), set()))
        rails[private] = (prior[0] | members, prior[1] | {comp.ref})

    if not rails:
        return ()

    # Closure 1 — subcircuits: a detected cell with one foot in a
    # region belongs to it whole (relay driver Q/R/D follow their K) —
    # but a CONNECTOR only joins through direct rail membership or
    # closure 3: an ADC channel cell spans the divider AND the harness
    # terminal it senses, and pulling J1 into the analog region via
    # that cell would claim the relay harness for analog (2026-06-12).
    # Closure 2 — passive chains: a 2-pin signal net from a member to a
    # passive/LED/switch-class part pulls it in (the relay indicator
    # LEDs D18-D21 and the opto chain R25/R32 are region parts on the
    # reference board but sit on private signal nets, not the rail).
    # Closure 3 — associated connectors: a connector whose signal-net
    # partners ALL live in the region is its connector (J3/J4 in the
    # analog zone, J1 over the relay bank — spec section 3).
    subs = [
        {r for r in s.refs if not _is_connector(by_ref[r])}
        for s in detect_subcircuits(requirements)
        if all(r in by_ref for r in s.refs)
    ]
    chain_prefixes = ("R", "C", "D", "L", "LED", "SW", "Q")
    two_pin_signal = [
        (conns[0][0], conns[1][0])
        for n, conns in net_conns.items()
        if len(conns) == 2 and not _is_main_rail(n)
        and conns[0][0] != conns[1][0]
    ]
    for name in rails:
        members, boundary = rails[name]
        changed = True
        while changed:
            changed = False
            for sub in subs:
                if members & sub and not sub <= members:
                    members |= sub
                    changed = True
            for a, b in two_pin_signal:
                for inside, outside in ((a, b), (b, a)):
                    chained = by_ref.get(outside)
                    if (inside in members and outside not in members
                            and chained is not None
                            and not _is_connector(chained)
                            and ref_alpha_prefix(outside) in chain_prefixes):
                        members.add(outside)
                        changed = True
        rails[name] = (members, boundary)

    # Merge rails with overlapping membership (supply + return pair).
    merged: list[tuple[set[str], set[str], set[str]]] = []  # names, members, ferrites
    for name in sorted(rails):
        members, ferrites = rails[name]
        members = set(members)
        ferrites = set(ferrites)
        for entry in merged:
            if entry[1] & members:
                entry[0].add(name)
                entry[1].update(members)
                entry[2].update(ferrites)
                break
        else:
            merged.append(({name}, set(members), ferrites))

    # Closure 3, AFTER the merge — a connector's partners may straddle
    # the supply and return rails of one domain (J4's ladder spans
    # AGND- and AVCC-seeded parts), so subset tests only make sense on
    # the merged member sets.
    assoc_partners: dict[str, set[str]] = {}
    for comp in requirements.components:
        if not _is_connector(comp):
            continue
        partners: set[str] = set()
        for n, conns in net_conns.items():
            if _is_main_rail(n):
                continue
            refs_here = [r for r, _ in conns]
            if comp.ref in refs_here:
                partners.update(r for r in refs_here if r != comp.ref)
        if partners:
            assoc_partners[comp.ref] = partners
    for _names, mem, _fer in merged:
        for ref, partners in assoc_partners.items():
            if ref not in mem and partners and partners <= mem:
                mem.add(ref)

    # Exclusive occupancy: a ref claimed by an earlier region (by name)
    # is dropped from later ones.
    merged.sort(key=lambda e: "+".join(sorted(e[0])))
    claimed: set[str] = set()
    out: list[IsolationRegionIntent] = []
    for names, members, ferrites in merged:
        refs = tuple(sorted((members - claimed) - ferrites))
        if len(refs) < _REGION_MIN_MEMBERS:
            continue
        claimed |= set(refs)
        out.append(IsolationRegionIntent(
            name="+".join(sorted(names)),
            refs=refs,
            boundary_refs=tuple(sorted(ferrites)),
        ))
    return tuple(out)


__all__ = [
    "FIELD_WIRING_EDGE",
    "IO_EDGE",
    "draft_board_intent",
    "draft_isolation_regions",
]

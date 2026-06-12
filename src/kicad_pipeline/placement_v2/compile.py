"""Constraint compiler — turns requirements into a :class:`ConstraintSet`.

Compiles placement intent from three sources, applied in priority order
so that higher-priority sources override lower ones on key collisions:

1. **Netlist topology** (lowest) — decoupling/relay-driver/ESD pin
   attachments, sequences from ``placement_group``/``placement_order``
   and repeated arrays, connector edge pins, board containment.
2. **Part rules** — per-part-class JSON rules (keepouts, isolation
   domains, edge pins). See :mod:`kicad_pipeline.placement_v2.part_rules`.
3. **Board intent** — the typed, requirements-resident ``BoardIntent``
   (human-confirmed connector edges and pin-freedom declarations;
   drafted by :mod:`kicad_pipeline.placement_v2.intent`).
4. **Human feedback locks** (highest) — persisted live-iteration
   corrections layered on top of the confirmed intent.

Override keys: edge pins collide on ``ref``; sequences on their exact
``refs`` tuple; a feedback pin-attach replaces ANY existing attach with
the same source pad (so "C1 belongs to U2, not U1" actually wins).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from kicad_pipeline.models.requirements import PinType
from kicad_pipeline.placement_v2.ir import (
    AttachBundle,
    Axis,
    BoardContain,
    CellKeepout,
    ConnectorFanout,
    ConstraintSet,
    ConstraintSource,
    Edge,
    EdgePin,
    FanoutLine,
    GroupAssoc,
    IsolationGap,
    IsolationRegion,
    PadRef,
    PinAttach,
    SequenceAlong,
)
from kicad_pipeline.placement_v2.part_rules import (
    apply_part_rules,
    load_feedback_locks,
    load_part_rules,
    ref_alpha_prefix,
)

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.requirements import Component, Net, ProjectRequirements

logger = logging.getLogger(__name__)

_POWER_NET_RE = re.compile(r"^\+?\d+(\.\d+)?V\d*$", re.IGNORECASE)
_POWER_NAME_TOKENS = ("VCC", "VDD")
_CONNECTOR_FP_TOKENS = ("usb", "rj45", "terminalblock", "pinheader", "conn")
_REF_NUM_RE = re.compile(r"(\d+)$")

_DECOUPLING_MAX_MM = 5.0
_DECOUPLING_IDEAL_MM = 2.0
_RELAY_DRIVER_MAX_MM = 10.0
_FLYBACK_MAX_MM = 8.0
_BASE_RESISTOR_MAX_MM = 6.0
_ESD_MAX_MM = 8.0
_ESD_MAX_PINS = 6  # connector-support parts are small (ESD diodes/arrays)
_NEAR_DEFAULT_MAX_MM = 5.0
_BOARD_MARGIN_MM = 0.5
_ESD_NET_MAX_CONNECTIONS = 3
_CONNECTOR_ARRAY_MIN = 3


def _ref_sort_key(ref: str) -> tuple[str, int]:
    """Sort refs naturally: K2 before K10."""
    m = _REF_NUM_RE.search(ref)
    return (ref_alpha_prefix(ref), int(m.group(1)) if m else 0)


class _Index:
    """Pre-computed netlist lookups shared by the compilation passes."""

    def __init__(self, requirements: ProjectRequirements) -> None:
        self.components: tuple[Component, ...] = requirements.components
        self.by_ref: dict[str, Component] = {c.ref: c for c in requirements.components}
        self.nets: tuple[Net, ...] = requirements.nets
        # (ref, pin) -> net names; ref -> {net -> pin numbers on it}
        self.ref_net_pins: dict[str, dict[str, list[str]]] = {}
        for net in requirements.nets:
            for conn in net.connections:
                self.ref_net_pins.setdefault(conn.ref, {}).setdefault(net.name, []).append(
                    conn.pin
                )
        self.power_nets: frozenset[str] = frozenset(
            n.name for n in requirements.nets if self._is_power_net(n)
        )
        # Declared pin-assignment freedom (board_intent): True/False
        # overrides the footprint-class heuristic; absent refs use it.
        self.pins_interchangeable: dict[str, bool] = {}
        self.intent_refs: frozenset[str] = frozenset()
        if requirements.board_intent is not None:
            self.intent_refs = frozenset(
                c.ref for c in requirements.board_intent.connectors
            )
            for c in requirements.board_intent.connectors:
                if c.pins_interchangeable is not None:
                    self.pins_interchangeable[c.ref] = c.pins_interchangeable
        self.gnd_nets: frozenset[str] = frozenset(
            n.name for n in requirements.nets if "GND" in n.name.upper()
        )
        # ref -> FeatureBlock name (functional grouping signal)
        self.feature_of: dict[str, str] = {}
        for feature in requirements.features:
            for ref in feature.components:
                self.feature_of.setdefault(ref, feature.name)

    def _is_power_net(self, net: Net) -> bool:
        upper = net.name.upper()
        if net.name.startswith("+"):
            return True  # KiCad rail convention: +5V_RELAY, +3V3_A, ...
        if _POWER_NET_RE.match(net.name) or any(t in upper for t in _POWER_NAME_TOKENS):
            return True
        # A net feeding an IC POWER_IN pin is a power net even if oddly named.
        for conn in net.connections:
            comp = self.by_ref.get(conn.ref)
            if comp is None or ref_alpha_prefix(comp.ref) != "U":
                continue
            pin = comp.get_pin(conn.pin)
            if pin is not None and pin.pin_type is PinType.POWER_IN:
                return True
        return False

    def is_signal_net(self, name: str) -> bool:
        """True when the net is neither power nor ground."""
        return name not in self.power_nets and name not in self.gnd_nets

    def nets_of(self, ref: str) -> dict[str, list[str]]:
        """Mapping net name -> pin numbers of *ref* on that net."""
        return self.ref_net_pins.get(ref, {})

    def refs_with_prefix(self, prefix: str) -> list[Component]:
        """Components whose alphabetic ref prefix equals *prefix*, ref-sorted."""
        return sorted(
            (c for c in self.components if ref_alpha_prefix(c.ref) == prefix),
            key=lambda c: _ref_sort_key(c.ref),
        )


def _decoupling_attaches(idx: _Index) -> list[PinAttach]:
    """Caps straddling a power net and GND attach to the IC on that net."""
    out: list[PinAttach] = []
    for cap in idx.refs_with_prefix("C"):
        nets = idx.nets_of(cap.ref)
        # GND itself is in power_nets (IC ground pins are POWER_IN
        # typed): a true decoupling cap straddles a NON-GND rail and
        # ground — without the exclusion, crystal load caps (XTAL+GND)
        # matched and got attached to the IC's ground pad.
        power = sorted(
            n for n in nets if n in idx.power_nets and n not in idx.gnd_nets
        )
        gnd = sorted(n for n in nets if n in idx.gnd_nets)
        if not power or not gnd:
            continue
        net_name = power[0]
        # Loads worth decoupling: ICs and LED-class modules (a WS2812
        # is an IC with a supply pin — binding only to U refs attached
        # C34 to the regulator 20mm away instead of the LED it serves;
        # proximity audit 2026-06-12).
        ics = [
            u
            for u in (*idx.refs_with_prefix("U"), *idx.refs_with_prefix("LED"))
            if net_name in idx.nets_of(u.ref)
        ]
        if not ics:
            continue
        preferred = [
            u
            for u in ics
            if cap.placement_group is not None and u.placement_group == cap.placement_group
        ]
        if not preferred:
            # A shared rail (+3V3) feeds several ICs: the cap belongs
            # to the IC in ITS OWN FeatureBlock. Blind ics[0] once
            # bound every rail cap to the regulator U2, 50-80mm away
            # (nl-s-3c, 2026-06-11).
            cap_feature = idx.feature_of.get(cap.ref)
            preferred = [
                u for u in ics
                if cap_feature is not None
                and idx.feature_of.get(u.ref) == cap_feature
            ]
        if not preferred and len(ics) > 1:
            continue  # ambiguous: a wrong hard bound is worse than none
        ic = preferred[0] if preferred else ics[0]
        out.append(
            PinAttach(
                src=PadRef(cap.ref, nets[net_name][0]),
                dst=PadRef(ic.ref, idx.nets_of(ic.ref)[net_name][0]),
                net=net_name,
                max_mm=_DECOUPLING_MAX_MM,
                ideal_mm=_DECOUPLING_IDEAL_MM,
            )
        )
    return out


def _placement_near_attaches(idx: _Index) -> list[PinAttach]:
    """Explicit ``placement_near="U1:VIN"`` hints become pin attachments."""
    out: list[PinAttach] = []
    for comp in idx.components:
        if comp.placement_near is None:
            continue
        target_ref, sep, target_pin_id = comp.placement_near.partition(":")
        target = idx.by_ref.get(target_ref)
        if not sep or target is None:
            logger.warning(
                "%s.placement_near=%r is not a resolvable 'REF:PIN'; skipping",
                comp.ref,
                comp.placement_near,
            )
            continue
        pin = target.get_pin(target_pin_id)
        if pin is None:  # fall back to pin-name lookup
            pin = next((p for p in target.pins if p.name == target_pin_id), None)
        target_pin_num = pin.number if pin is not None else target_pin_id
        target_nets = idx.nets_of(target_ref)
        shared_nets = [n for n, pins in target_nets.items() if target_pin_num in pins]
        if not shared_nets and pin is not None and pin.net is not None:
            shared_nets = [pin.net]
        src_pin, net_name = "1", shared_nets[0] if shared_nets else ""
        for n, pins in idx.nets_of(comp.ref).items():
            if n in shared_nets:
                src_pin, net_name = pins[0], n
                break
        max_mm = comp.placement_near_max_mm or _NEAR_DEFAULT_MAX_MM
        out.append(
            PinAttach(
                src=PadRef(comp.ref, src_pin),
                dst=PadRef(target_ref, target_pin_num),
                net=net_name,
                max_mm=max_mm,
                ideal_mm=max_mm / 2.0,
            )
        )
    return out


def _relay_driver_attaches(idx: _Index) -> list[PinAttach]:
    """Q drives K coil; flyback D across the coil; base R into Q."""
    out: list[PinAttach] = []
    for q in idx.refs_with_prefix("Q"):
        q_nets = idx.nets_of(q.ref)
        coil_net: str | None = None
        for net_name in sorted(q_nets):
            if not idx.is_signal_net(net_name):
                continue
            for k in idx.refs_with_prefix("K"):
                k_pins = idx.nets_of(k.ref).get(net_name)
                if not k_pins:
                    continue
                coil_net = net_name
                k_pad = PadRef(k.ref, k_pins[0])
                out.append(
                    PinAttach(
                        src=PadRef(q.ref, q_nets[net_name][0]),
                        dst=k_pad,
                        net=net_name,
                        max_mm=_RELAY_DRIVER_MAX_MM,
                        ideal_mm=_RELAY_DRIVER_MAX_MM / 2.0,
                    )
                )
                for d in idx.refs_with_prefix("D"):
                    d_pins = idx.nets_of(d.ref).get(net_name)
                    if d_pins:
                        out.append(
                            PinAttach(
                                src=PadRef(d.ref, d_pins[0]),
                                dst=k_pad,
                                net=net_name,
                                max_mm=_FLYBACK_MAX_MM,
                                ideal_mm=_FLYBACK_MAX_MM / 2.0,
                            )
                        )
                break
            if coil_net is not None:
                break
        # Base resistor: R sharing a non-power, non-coil net with Q.
        for net_name in sorted(q_nets):
            if not idx.is_signal_net(net_name) or net_name == coil_net:
                continue
            for r in idx.refs_with_prefix("R"):
                r_pins = idx.nets_of(r.ref).get(net_name)
                if r_pins:
                    out.append(
                        PinAttach(
                            src=PadRef(r.ref, r_pins[0]),
                            dst=PadRef(q.ref, q_nets[net_name][0]),
                            net=net_name,
                            max_mm=_BASE_RESISTOR_MAX_MM,
                            ideal_mm=_BASE_RESISTOR_MAX_MM / 2.0,
                        )
                    )
    return out


def _connector_support_attaches(idx: _Index) -> list[PinAttach]:
    """ESD diodes / protection ICs on short connector nets attach to the J pad."""
    out: list[PinAttach] = []
    for net in idx.nets:
        if not idx.is_signal_net(net.name) or len(net.connections) > _ESD_NET_MAX_CONNECTIONS:
            continue
        j_conns = [c for c in net.connections if ref_alpha_prefix(c.ref) == "J"]
        # Only SMALL parts are connector support (ESD diodes/arrays):
        # without the pin-count cap, the ESP32 itself was attached to
        # the USB connector with an 8mm bound it can never satisfy.
        supports = [
            c for c in net.connections
            if ref_alpha_prefix(c.ref) in ("D", "U")
            and (comp := idx.by_ref.get(c.ref)) is not None
            and len(comp.pins) <= _ESD_MAX_PINS
        ]
        if not j_conns or not supports:
            continue
        j_pad = PadRef(j_conns[0].ref, j_conns[0].pin)
        out.extend(
            PinAttach(
                src=PadRef(s.ref, s.pin),
                dst=j_pad,
                net=net.name,
                max_mm=_ESD_MAX_MM,
                ideal_mm=_ESD_MAX_MM / 2.0,
            )
            for s in supports
        )
    return out


#: Partner preference for chain completion: actively-driving parts first.
_CHAIN_PARTNER_RANK = {"Q": 0, "U": 1, "K": 2, "R": 3, "D": 4, "C": 5, "L": 6}
_CHAIN_MAX_MM = 8.0


def _chain_completion_attaches(
    idx: _Index, attached_srcs: frozenset[str],
) -> list[PinAttach]:
    """Attach leftover 2-3 pad passives through their most private net.

    Indicator chains (relay LED + series resistor), snubbers, and other
    support parts that no specific rule caught would otherwise fall to
    the cell's orphan shelf and break the column layout the reference
    board demonstrates. Each unattached R/C/D/L component is linked to
    the best partner (driver Q first, then U/K/...) on its smallest
    signal net, so the generator's chain inheritance can stack it.
    """
    conn_count = {n.name: len(n.connections) for n in idx.nets}
    out: list[PinAttach] = []
    new_dst: dict[str, str] = {}  # src ref -> dst ref emitted by THIS pass
    for comp in sorted(idx.components, key=lambda c: _ref_sort_key(c.ref)):
        if ref_alpha_prefix(comp.ref) not in ("R", "C", "D", "L"):
            continue
        if comp.ref in attached_srcs:
            continue
        fp_lower = comp.footprint.lower()
        if any(t in fp_lower for t in _CONNECTOR_FP_TOKENS):
            continue
        for net_name in sorted(
            (n for n in idx.nets_of(comp.ref) if idx.is_signal_net(n)),
            key=lambda n: (conn_count.get(n, 0), n),
        ):
            partners = sorted(
                (
                    (other, pins)
                    for net in idx.nets
                    if net.name == net_name
                    for other, pins in (
                        (c.ref, c.pin) for c in net.connections
                    )
                    if other != comp.ref and other in idx.by_ref
                    # No 2-cycles: a partner already attached TO this
                    # component would leave both unreachable from the
                    # anchor (the generator places hosts before tails).
                    and new_dst.get(other) != comp.ref
                    and not any(
                        t in idx.by_ref[other].footprint.lower()
                        for t in _CONNECTOR_FP_TOKENS
                    )
                    # Same FeatureBlock only: a cross-group monitoring
                    # tap (ADC divider sensing a relay contact) is a
                    # measurement net, not a placement chain — the 8mm
                    # bound put R18 53mm in the red on nl-s-3c
                    # (2026-06-11).
                    and idx.feature_of.get(other) == idx.feature_of.get(comp.ref)
                ),
                # Prefer the most LOCAL partner — the one touching the
                # fewest nets (a crystal over the MCU, a transistor
                # over the relay): small neighbors keep the chain
                # inside one cell, where its bound is enforceable.
                key=lambda rp: (
                    len(idx.nets_of(rp[0])),
                    _CHAIN_PARTNER_RANK.get(ref_alpha_prefix(rp[0]), 9),
                    _ref_sort_key(rp[0]),
                ),
            )
            if not partners:
                continue
            partner_ref, partner_pin = partners[0]
            src_pin = idx.nets_of(comp.ref)[net_name][0]
            out.append(PinAttach(
                src=PadRef(comp.ref, src_pin),
                dst=PadRef(partner_ref, partner_pin),
                net=net_name,
                max_mm=_CHAIN_MAX_MM,
                ideal_mm=_CHAIN_MAX_MM / 2.0,
            ))
            new_dst[comp.ref] = partner_ref
            break
    return out


def _group_sequences(
    idx: _Index, attached_srcs: frozenset[str] = frozenset(),
) -> list[SequenceAlong]:
    """``placement_group`` + ``placement_order`` become horizontal sequences.

    Refs that already have a pin attachment are EXCLUDED: a decoupling
    cap hugs its regulator pad exactly; slotting it at sequence pitch
    instead breaks the attachment bound (power-chain cells halted on
    this). The sequence keeps the signal-flow backbone (ICs,
    inductors); attached passives ride along with their hosts.
    """
    groups: dict[str, list[Component]] = {}
    for comp in idx.components:
        if comp.placement_group is not None and comp.placement_order is not None:
            if comp.ref in attached_srcs:
                continue
            groups.setdefault(comp.placement_group, []).append(comp)
    out: list[SequenceAlong] = []
    for name in sorted(groups):
        members = sorted(groups[name], key=lambda c: (c.placement_order or 0, c.ref))
        if len(members) >= 2:
            out.append(SequenceAlong(axis=Axis.HORIZONTAL, refs=tuple(m.ref for m in members)))
    return out


def _array_sequences(idx: _Index) -> list[SequenceAlong]:
    """Repeated identical relays (n>=2) and connectors (n>=3) form arrays."""
    out: list[SequenceAlong] = []
    for prefix, minimum in (("K", 2), ("J", _CONNECTOR_ARRAY_MIN)):
        kinds: dict[tuple[str, str], list[str]] = {}
        for comp in idx.refs_with_prefix(prefix):
            kinds.setdefault((comp.value, comp.footprint), []).append(comp.ref)
        for key in sorted(kinds):
            refs = sorted(kinds[key], key=_ref_sort_key)
            if len(refs) >= minimum:
                out.append(SequenceAlong(
                    axis=Axis.HORIZONTAL, refs=tuple(refs), aligned=True,
                ))
    return out


def _is_connector(comp: Component) -> bool:
    fp = comp.footprint.lower()
    return ref_alpha_prefix(comp.ref) == "J" or any(
        t in fp for t in _CONNECTOR_FP_TOKENS
    )


def _connector_group_assocs(idx: _Index) -> list[GroupAssoc]:
    """Connector -> parent FeatureBlock when its signal partners agree.

    The connector-as-subgroup directive (board owner, 2026-06-11
    evening addendum): an edge connector whose SIGNAL nets terminate
    overwhelmingly in ONE FeatureBlock is that group's edge subgroup
    (J14's SPI/TFT nets all end at U3 -> MCU). Majority means strictly
    more than half of the signal-net partner connections; power and
    ground nets carry no ownership. The connector itself does not vote
    (J14 lives in its own 'Display' block; its partners decide).
    """
    out: list[GroupAssoc] = []
    for comp in idx.components:
        if not _is_connector(comp):
            continue
        votes: dict[str, list[str]] = {}
        total = 0
        for net in idx.nets:
            if not idx.is_signal_net(net.name):
                continue
            pins_here = [c for c in net.connections if c.ref == comp.ref]
            if not pins_here:
                continue
            for conn in net.connections:
                if conn.ref == comp.ref:
                    continue
                block = idx.feature_of.get(conn.ref)
                if block is None:
                    continue
                total += 1
                votes.setdefault(block, []).append(conn.ref)
        if not votes:
            continue
        block, partners = max(
            votes.items(), key=lambda kv: (len(kv[1]), kv[0]),
        )
        if len(partners) * 2 <= total:
            continue  # no strict majority: ambiguous, compile nothing
        out.append(GroupAssoc(
            ref=comp.ref,
            group=block,
            partner_refs=tuple(sorted(set(partners), key=_ref_sort_key)),
        ))
    return out


def _connector_edge_pins(idx: _Index) -> list[EdgePin]:
    """Every connector (J* ref or connector-family footprint) pins to an edge."""
    out: list[EdgePin] = []
    for comp in idx.components:
        fp = comp.footprint.lower()
        if ref_alpha_prefix(comp.ref) == "J" or any(t in fp for t in _CONNECTOR_FP_TOKENS):
            out.append(EdgePin(ref=comp.ref, edge=None, face_out=True))
    return out


#: Footprint tokens of connectors whose PIN ASSIGNMENT is a free choice
#: in requirements (screw terminals, generic headers) — crossing attach
#: lines there mean the requirements chose the wrong pin order. Fixed-
#: pinout pairs (RJ45 <-> PHY, USB <-> MCU) are excluded: their pin
#: orders are device facts and a straight-line crossing may be routed
#: legitimately with a layer swap.
_FREE_PIN_CONNECTOR_TOKENS = ("terminalblock", "pinheader", "conn_01x", "screw")


def _has_free_pin_order(comp: Component, idx: _Index | None = None) -> bool:
    """Pin assignment is a free requirements choice for this part.

    A board_intent declaration wins (``pins_interchangeable=False``
    marks a FIXED external contract — J1 harness, J14 display ribbon);
    undeclared parts fall back to the footprint-class heuristic.
    """
    if idx is not None and comp.ref in idx.pins_interchangeable:
        return idx.pins_interchangeable[comp.ref]
    fp = comp.footprint.lower()
    return any(t in fp for t in _FREE_PIN_CONNECTOR_TOKENS)


def _attach_bundles(idx: _Index) -> list[AttachBundle]:
    """Pairs of parts joined by >= 2 two-pin signal nets must not cross.

    The relay NO/NC defect class (Gate C 2026-06-11 item 2): a relay's
    contact pads and its screw terminal's pins are joined by parallel
    two-pin nets; if the terminal pin order does not mirror the relay's
    physical pad order, the straight connections cross and force an
    avoidable crossover trace. Compiled for every such pair where at
    least one side has a freely-assignable pin order (see
    :data:`_FREE_PIN_CONNECTOR_TOKENS`), then verified geometrically at
    Gate A (segment intersections == 0).
    """
    pair_nets: dict[tuple[str, str], list[tuple[str, PadRef, PadRef]]] = {}
    for net in idx.nets:
        if len(net.connections) != 2 or not idx.is_signal_net(net.name):
            continue
        a, b = net.connections
        if a.ref == b.ref:
            continue
        comp_a, comp_b = idx.by_ref.get(a.ref), idx.by_ref.get(b.ref)
        if comp_a is None or comp_b is None:
            continue
        if not (_has_free_pin_order(comp_a, idx) or _has_free_pin_order(comp_b, idx)):
            continue
        ref_a, ref_b = sorted((a.ref, b.ref))
        pad_a = PadRef(a.ref, a.pin) if a.ref == ref_a else PadRef(b.ref, b.pin)
        pad_b = PadRef(b.ref, b.pin) if a.ref == ref_a else PadRef(a.ref, a.pin)
        pair_nets.setdefault((ref_a, ref_b), []).append((net.name, pad_a, pad_b))
    out: list[AttachBundle] = []
    for (ref_a, ref_b), entries in sorted(pair_nets.items()):
        if len(entries) < 2:
            continue
        entries.sort()
        out.append(AttachBundle(
            ref_a=ref_a,
            ref_b=ref_b,
            pad_pairs=tuple((pa, pb) for _, pa, pb in entries),
            nets=tuple(name for name, _, _ in entries),
        ))
    return out


def _connector_fanouts(idx: _Index) -> list[ConnectorFanout]:
    """Every free-pin connector's ratsnest lines must not cross.

    For each connected pad of the connector, the line runs to the
    nearest same-net pad (resolved by the verifier from the artifact).
    Global nets (GND) are INCLUDED — the analog terminals' AIN/GND X
    (human finding 2026-06-11) was invisible to AttachBundle precisely
    because GND has many connections.
    """
    out: list[ConnectorFanout] = []
    for comp in idx.components:
        # Free-pin connectors AND connectors whose pin freedom the
        # human DECLARED (either way): a FIXED-pinout harness terminal
        # still wants crossing-free fanout — components move even when
        # pins cannot (the J1 relay-row reversal was driven by exactly
        # this check). Edge-only intent entries do NOT qualify: naming
        # the ESP32's edge must not hang a 40-pin fanout web on it
        # (nl-s-3c went 23 -> 153 violations doing that, 2026-06-12).
        if not (_has_free_pin_order(comp, idx)
                or comp.ref in idx.pins_interchangeable):
            continue
        lines: list[FanoutLine] = []
        for net_name, pins in sorted(idx.nets_of(comp.ref).items()):
            net = next(n for n in idx.nets if n.name == net_name)
            for pin in pins:
                candidates = tuple(
                    PadRef(c.ref, c.pin) for c in net.connections
                    if (c.ref, c.pin) != (comp.ref, pin)
                )
                if candidates:
                    lines.append(FanoutLine(
                        net=net_name, src=PadRef(comp.ref, pin),
                        candidates=candidates,
                    ))
        if len(lines) >= 2:
            out.append(ConnectorFanout(ref=comp.ref, lines=tuple(lines)))
    return out


#: Passive ref prefixes whose 2-pin link to a connector pad roots the
#: subcircuit chain at that pad (divider entry at its screw terminal).
_CHAIN_ENTRY_PREFIXES = frozenset({"R", "C", "D", "L"})
_CONNECTOR_CHAIN_MAX_MM = 8.0
_CONNECTOR_CHAIN_IDEAL_MM = 3.0


def _connector_chain_attaches(idx: _Index) -> list[PinAttach]:
    """Root each channel chain at its connector pin.

    A 2-pin signal net joining a free-pin connector pad to a passive
    (divider entry resistor) becomes a PinAttach so the band generator
    places the passive AT the connector pin's coordinate instead of
    shelving the whole channel as unordered orphans — the root cause of
    the analog AIN/GND crossings (human finding 2026-06-11).

    Only for connectors carrying exactly ONE such chain: a per-channel
    screw terminal anchors its own channel cell, but a shared harness
    connector (J3 with three channels) cannot have every channel within
    the bound — there the connector_fanout crossing check is the
    invariant, not a hard distance (found on nl-s-3c, 2026-06-11).
    """
    candidates: list[PinAttach] = []
    for net in idx.nets:
        if len(net.connections) != 2 or not idx.is_signal_net(net.name):
            continue
        a, b = net.connections
        comp_a, comp_b = idx.by_ref.get(a.ref), idx.by_ref.get(b.ref)
        if comp_a is None or comp_b is None:
            continue
        conn, part = (a, b) if _has_free_pin_order(comp_a, idx) else (b, a)
        conn_comp = idx.by_ref[conn.ref]
        part_comp = idx.by_ref[part.ref]
        if not _has_free_pin_order(conn_comp, idx) or _has_free_pin_order(part_comp, idx):
            continue
        if ref_alpha_prefix(part.ref) not in _CHAIN_ENTRY_PREFIXES:
            continue
        if idx.feature_of.get(conn.ref) != idx.feature_of.get(part.ref):
            # A harness connector in another FeatureBlock feeds this
            # channel from across the board — the fanout crossing
            # check governs there, not a hard distance.
            continue
        candidates.append(PinAttach(
            src=PadRef(part.ref, part.pin),
            dst=PadRef(conn.ref, conn.pin),
            net=net.name,
            max_mm=_CONNECTOR_CHAIN_MAX_MM,
            ideal_mm=_CONNECTOR_CHAIN_IDEAL_MM,
        ))
    chains_per_connector: dict[str, int] = {}
    for pa in candidates:
        chains_per_connector[pa.dst.ref] = chains_per_connector.get(pa.dst.ref, 0) + 1
    return [pa for pa in candidates if chains_per_connector[pa.dst.ref] == 1]


def _with_calibrated_opening(prior: EdgePin | None, ep: EdgePin) -> EdgePin:
    """Carry a prior CALIBRATED opening into an overriding edge pin.

    A human edge declaration pins the EDGE; the part-rule calibrated
    opening is orthogonal measured data and must survive the merge — a
    wholesale replace once dropped the RJ45/ESP32 openings, so the
    floorplan aimed them by the courtyard-bulge proxy and faced them
    the wrong way (nl-s-3c, 2026-06-11).
    """
    if prior is not None and prior.opening is not None and ep.opening is None:
        return EdgePin(
            ref=ep.ref, edge=ep.edge, face_out=ep.face_out,
            max_edge_distance_mm=ep.max_edge_distance_mm,
            opening=prior.opening, source=ep.source,
        )
    return ep


def compile_constraints(
    requirements: ProjectRequirements,
    *,
    part_rules_path: Path | None = None,
    feedback_locks_path: Path | None = None,
) -> ConstraintSet:
    """Compile the full placement :class:`ConstraintSet` for a board.

    Args:
        requirements: The validated project requirements (components + nets).
        part_rules_path: Optional part rules JSON (``data/part_rules.json``).
            ``None`` skips part rules entirely. Malformed content raises
            :class:`~kicad_pipeline.exceptions.ConfigurationError`.
        feedback_locks_path: Optional human feedback locks JSON. A missing
            file yields no locks; malformed content raises ``ValueError``.

    Returns:
        The merged constraint set with ``BoardContain(margin_mm=0.5)``.
    """
    idx = _Index(requirements)

    attaches: dict[tuple[str, str, str, str], PinAttach] = {}
    for pa in (
        _decoupling_attaches(idx)
        + _placement_near_attaches(idx)
        + _relay_driver_attaches(idx)
        + _connector_support_attaches(idx)
        + _connector_chain_attaches(idx)
    ):
        attaches.setdefault((pa.src.ref, pa.src.pin, pa.dst.ref, pa.dst.pin), pa)
    attached_srcs = frozenset(k[0] for k in attaches)
    for pa in _chain_completion_attaches(idx, attached_srcs):
        attaches.setdefault((pa.src.ref, pa.src.pin, pa.dst.ref, pa.dst.pin), pa)
    all_attached = frozenset(k[0] for k in attaches)
    sequences: dict[tuple[str, ...], SequenceAlong] = {
        s.refs: s
        for s in _group_sequences(idx, all_attached) + _array_sequences(idx)
    }
    edge_pins: dict[str, EdgePin] = {e.ref: e for e in _connector_edge_pins(idx)}
    keepouts: tuple[CellKeepout, ...] = ()
    isolation: tuple[IsolationGap, ...] = ()

    if part_rules_path is not None:
        compiled = apply_part_rules(load_part_rules(part_rules_path), idx.components)
        keepouts = compiled.keepouts
        isolation = compiled.isolation
        for ep in compiled.edge_pins:  # part rules override netlist edge pins
            edge_pins[ep.ref] = ep

    isolation_regions: tuple[IsolationRegion, ...] = ()
    if requirements.board_intent is not None:
        # Typed board intent (requirements-resident, human-confirmed):
        # same authority as feedback locks; locks loaded AFTER remain
        # the live-iteration override channel on top of it.
        isolation_regions = tuple(
            IsolationRegion(
                name=r.name, refs=r.refs, boundary_refs=r.boundary_refs,
                min_gap_mm=r.min_gap_mm,
            )
            for r in requirements.board_intent.isolation_regions
        )
        for ci in requirements.board_intent.connectors:
            if ci.edge is None:
                continue
            ep = EdgePin(
                ref=ci.ref, edge=Edge(ci.edge), face_out=True,
                source=ConstraintSource.HUMAN_FEEDBACK,
            )
            edge_pins[ci.ref] = _with_calibrated_opening(
                edge_pins.get(ci.ref), ep,
            )

    if feedback_locks_path is not None:
        locks = load_feedback_locks(feedback_locks_path)
        for pa in locks.pin_attach:
            # A human lock replaces ANY attach with the same source pad.
            stale = [k for k in attaches if (k[0], k[1]) == (pa.src.ref, pa.src.pin)]
            for k in stale:
                del attaches[k]
            attaches[(pa.src.ref, pa.src.pin, pa.dst.ref, pa.dst.pin)] = pa
        for seq in locks.sequences:
            sequences[seq.refs] = seq
        for ep in locks.edge_pins:
            edge_pins[ep.ref] = _with_calibrated_opening(
                edge_pins.get(ep.ref), ep,
            )

    result = ConstraintSet(
        pin_attach=tuple(attaches.values()),
        sequences=tuple(sequences.values()),
        edge_pins=tuple(edge_pins.values()),
        keepouts=keepouts,
        isolation=isolation,
        bundles=tuple(_attach_bundles(idx)),
        fanouts=tuple(_connector_fanouts(idx)),
        group_assocs=tuple(_connector_group_assocs(idx)),
        isolation_regions=isolation_regions,
        contain=BoardContain(margin_mm=_BOARD_MARGIN_MM, source=ConstraintSource.NETLIST),
    )
    logger.info(
        "compiled %d constraints (%d attach, %d seq, %d edge, %d keepout, "
        "%d isolation, %d bundle, %d fanout, %d assoc, %d region)",
        result.count(),
        len(result.pin_attach),
        len(result.sequences),
        len(result.edge_pins),
        len(result.keepouts),
        len(result.isolation),
        len(result.bundles),
        len(result.fanouts),
        len(result.group_assocs),
        len(result.isolation_regions),
    )
    return result

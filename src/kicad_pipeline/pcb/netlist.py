"""Netlist extraction and net-number assignment for the kicad-ai-pipeline.

Converts a :class:`~kicad_pipeline.models.requirements.ProjectRequirements`
object into a :class:`Netlist` that maps every electrical net to the pads it
connects, and provides helpers to stamp net numbers back onto placed
:class:`~kicad_pipeline.models.pcb.Footprint` objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import PCBError
from kicad_pipeline.models.pcb import Footprint, NetEntry, Pad

if TYPE_CHECKING:
    from kicad_pipeline.models.requirements import ProjectRequirements

_log = logging.getLogger(__name__)

# Net number reserved for unconnected pads
_NET_UNCONNECTED: int = 0
# GND net is always assigned net number 1 by convention
_NET_GND_NUMBER: int = 1
_GND_NAMES: frozenset[str] = frozenset({"GND", "AGND", "DGND", "PGND", "GND_A", "GND_D"})


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetlistEntry:
    """Maps a single net to all the pads it connects.

    Attributes:
        net: The :class:`~kicad_pipeline.models.pcb.NetEntry` (number + name).
        pad_refs: Tuple of ``(ref, pad_number)`` pairs connected to this net.
    """

    net: NetEntry
    pad_refs: tuple[tuple[str, str], ...]  # ((ref, pad_number), ...)


@dataclass(frozen=True)
class Netlist:
    """Complete netlist: all nets and their pad connections.

    Attributes:
        entries: Tuple of :class:`NetlistEntry` objects, one per net.
    """

    entries: tuple[NetlistEntry, ...]

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def net_for_pad(self, ref: str, pad_number: str) -> NetEntry | None:
        """Return the :class:`NetEntry` for a given ref + pad number, or ``None``.

        Args:
            ref: Component reference designator (e.g. ``"R1"``).
            pad_number: Pad number string (e.g. ``"1"``, ``"A2"``).

        Returns:
            The matching :class:`NetEntry`, or ``None`` if not found.
        """
        for entry in self.entries:
            for r, p in entry.pad_refs:
                if r == ref and p == pad_number:
                    return entry.net
        return None

    def pads_for_net(self, net_name: str) -> tuple[tuple[str, str], ...]:
        """Return all ``(ref, pad_number)`` tuples for the named net.

        Args:
            net_name: Net name string (e.g. ``"GND"``, ``"+3V3"``).

        Returns:
            Tuple of ``(ref, pad_number)`` pairs, empty if name not found.
        """
        for entry in self.entries:
            if entry.net.name == net_name:
                return entry.pad_refs
        return ()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_netlist(requirements: ProjectRequirements) -> Netlist:
    """Build a :class:`Netlist` from :class:`ProjectRequirements`.

    Net numbering rules:

    - Net 0 is reserved for unconnected pads (never assigned here).
    - GND (and common GND aliases) is always net 1.
    - All other nets are numbered from 2 upward in the order they appear in
      ``requirements.nets``.

    Args:
        requirements: Validated project requirements object.

    Returns:
        Populated :class:`Netlist`.

    Raises:
        PCBError: When a net connection references an unknown component.
    """
    nets = list(requirements.nets)

    # Separate GND nets from the rest
    gnd_nets = [n for n in nets if n.name in _GND_NAMES]
    other_nets = [n for n in nets if n.name not in _GND_NAMES]

    # Assign net numbers: GND variants first (all share number 1 if more than
    # one GND alias exists), then remaining nets from 2.
    ordered = gnd_nets + other_nets

    entries: list[NetlistEntry] = []
    next_number = _NET_GND_NUMBER  # start at 1

    for net in ordered:
        if net.name in _GND_NAMES:
            net_number = _NET_GND_NUMBER
        else:
            # GND(s) already consumed slot 1; subsequent nets from 2
            if next_number == _NET_GND_NUMBER:
                next_number = _NET_GND_NUMBER + 1
            net_number = next_number
            next_number += 1

        net_entry = NetEntry(number=net_number, name=net.name)
        pad_refs: list[tuple[str, str]] = []

        for conn in net.connections:
            # Validate the referenced component exists
            comp = requirements.get_component(conn.ref)
            if comp is None:
                raise PCBError(
                    f"Net '{net.name}' references unknown component '{conn.ref}'"
                )
            pad_refs.append((conn.ref, conn.pin))
            _log.debug(
                "netlist: net=%s(%d) ← %s.%s", net.name, net_number, conn.ref, conn.pin
            )

        entries.append(NetlistEntry(net=net_entry, pad_refs=tuple(pad_refs)))

    _log.info("build_netlist: built %d net entries", len(entries))
    return Netlist(entries=tuple(entries))


# ---------------------------------------------------------------------------
# Net-number assignment
# ---------------------------------------------------------------------------


def assign_net_numbers_to_footprints(
    footprints: list[Footprint],
    netlist: Netlist,
) -> list[Footprint]:
    """Return a new list of :class:`Footprint` objects with pad net numbers stamped in.

    For each pad in each footprint, the netlist is queried for
    ``(footprint.ref, pad.number)``.  If a match is found the pad's
    ``net_number`` and ``net_name`` are set; otherwise ``net_number`` is set
    to ``0`` (unconnected) and ``net_name`` to ``None``.

    Because :class:`Pad` and :class:`Footprint` are frozen dataclasses, new
    instances are created via :func:`dataclasses.replace`.

    Args:
        footprints: List of footprints to process.
        netlist: Netlist produced by :func:`build_netlist`.

    Returns:
        New list of :class:`Footprint` objects with updated pads.
    """
    result: list[Footprint] = []
    for fp in footprints:
        updated_pads: list[Pad] = []
        for pad in fp.pads:
            net_entry = netlist.net_for_pad(fp.ref, pad.number)
            if net_entry is not None:
                updated_pad = replace(
                    pad,
                    net_number=net_entry.number,
                    net_name=net_entry.name,
                )
            else:
                updated_pad = replace(
                    pad,
                    net_number=_NET_UNCONNECTED,
                    net_name=None,
                )
            updated_pads.append(updated_pad)
            _log.debug(
                "assign_net: %s.%s → net %s",
                fp.ref,
                pad.number,
                net_entry.number if net_entry else "0 (unconnected)",
            )

        result.append(replace(fp, pads=tuple(updated_pads)))

    _log.info(
        "assign_net_numbers_to_footprints: processed %d footprints", len(footprints)
    )
    return result

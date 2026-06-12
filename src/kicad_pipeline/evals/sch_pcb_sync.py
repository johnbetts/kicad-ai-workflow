"""Written-file schematic <-> PCB sync gate (the "Update PCB from
Schematic must be a no-op" invariant).

The in-memory ``check_schematic_pcb_sync`` only compares ref SETS
between the built PCB and requirements; it never reads the WRITTEN
``.kicad_sch``, so a schematic KiCad cannot even load still "passed
sync" (found 2026-06-12: every generated schematic failed KiCad 10's
parser on numeric pin types + missing ``embedded_fonts``).

This gate goes end to end through KiCad's OWN parser:

1. ``kicad-cli sch export netlist`` on the written schematic — if the
   file does not load, the gate fails outright.
2. The exported netlist (refs, footprint assignments, net name per
   ref.pin) is diffed against the WRITTEN ``.kicad_pcb`` (footprint
   lib_ids and pad net assignments).

When all three agree, KiCad's "Update PCB from Schematic" has nothing
to change: same components, same footprints, same connectivity.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kicad_pipeline.sexp.parser import parse_file

if TYPE_CHECKING:
    from kicad_pipeline.sexp.writer import SExpNode

log = logging.getLogger(__name__)

#: Known kicad-cli locations, first match wins (PATH is checked first).
_KICAD_CLI_CANDIDATES = (
    "kicad-cli",
    "/Applications/KiCad 10/KiCad.app/Contents/MacOS/kicad-cli",
    "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
)


@dataclass(frozen=True)
class SyncReport:
    """Result of the written-file sync check."""

    passed: bool
    issues: tuple[str, ...]
    components_checked: int
    nets_checked: int

    @property
    def detail(self) -> str:
        """Single-line summary for ledgers and logs."""
        if self.passed:
            return (
                f"OK ({self.components_checked} components, "
                f"{self.nets_checked} pin-net assignments)"
            )
        return "; ".join(self.issues[:10])


def find_kicad_cli() -> str | None:
    """Locate kicad-cli, or None when KiCad is not installed."""
    for cand in _KICAD_CLI_CANDIDATES:
        path = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if path:
            return path
    return None


def _children(node: SExpNode, name: str) -> list[list[SExpNode]]:
    if not isinstance(node, list):
        return []
    return [
        c for c in node
        if isinstance(c, list) and c and c[0] == name
    ]


def _atom(node: SExpNode, name: str) -> str | None:
    if not isinstance(node, list):
        return None
    for c in node:
        if isinstance(c, list) and c and c[0] == name and len(c) > 1:
            return str(c[1])
    return None


def _parse_netlist(path: Path) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """kicadsexpr netlist -> ({ref: footprint}, {(ref, pin): net name})."""
    root = parse_file(path)
    footprints: dict[str, str] = {}
    pin_nets: dict[tuple[str, str], str] = {}
    for comps in _children(root, "components"):
        for comp in _children(comps, "comp"):
            ref = _atom(comp, "ref")
            fp = _atom(comp, "footprint")
            if ref is not None:
                footprints[ref] = fp or ""
    for nets in _children(root, "nets"):
        for net in _children(nets, "net"):
            name = _atom(net, "name") or ""
            for node in _children(net, "node"):
                ref = _atom(node, "ref")
                pin = _atom(node, "pin")
                if ref is not None and pin is not None:
                    pin_nets[(ref, pin)] = name
    return footprints, pin_nets


def _parse_pcb(path: Path) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Written .kicad_pcb -> ({ref: lib_id}, {(ref, pad): net name})."""
    root = parse_file(path)
    footprints: dict[str, str] = {}
    pad_nets: dict[tuple[str, str], str] = {}
    for fp in _children(root, "footprint"):
        lib_id = str(fp[1]) if len(fp) > 1 else ""
        ref = None
        for prop in _children(fp, "property"):
            if len(prop) > 2 and str(prop[1]) == "Reference":
                ref = str(prop[2])
                break
        if ref is None:
            continue
        footprints[ref] = lib_id
        for pad in _children(fp, "pad"):
            number = str(pad[1]) if len(pad) > 1 else ""
            for net in _children(pad, "net"):
                if len(net) > 2:
                    pad_nets[(ref, number)] = str(net[2])
    return footprints, pad_nets


def check_written_sync(
    sch_path: Path,
    pcb_path: Path,
    kicad_cli: str | None = None,
) -> SyncReport:
    """Run the end-to-end written-file sync check.

    Args:
        sch_path: The written ``.kicad_sch``.
        pcb_path: The written ``.kicad_pcb``.
        kicad_cli: Explicit kicad-cli path; auto-detected when None.

    Returns:
        A :class:`SyncReport`. A missing kicad-cli FAILS the check
        loudly — silently skipping would report sync that was never
        verified.
    """
    cli = kicad_cli or find_kicad_cli()
    if cli is None:
        return SyncReport(
            passed=False,
            issues=("kicad-cli not found: written-file sync cannot be verified",),
            components_checked=0, nets_checked=0,
        )

    with tempfile.TemporaryDirectory(prefix="sch_sync_") as td:
        netlist_path = Path(td) / "netlist.net"
        try:
            proc = subprocess.run(
                [cli, "sch", "export", "netlist", "--format", "kicadsexpr",
                 "-o", str(netlist_path), str(sch_path)],
                capture_output=True, text=True, timeout=300,
            )
            failed = proc.returncode != 0 or not netlist_path.exists()
            msg = (proc.stderr or proc.stdout or "").strip() or "no output"
        except (OSError, subprocess.TimeoutExpired) as exc:
            failed = True
            msg = str(exc)
        if failed:
            return SyncReport(
                passed=False,
                issues=(
                    f"KiCad cannot load the written schematic ({msg}) — "
                    f"'Update PCB from Schematic' would fail outright",
                ),
                components_checked=0, nets_checked=0,
            )
        sch_fps, sch_pin_nets = _parse_netlist(netlist_path)

    pcb_fps, pcb_pad_nets = _parse_pcb(pcb_path)
    # Mounting holes are board-only by design.
    pcb_fps = {r: f for r, f in pcb_fps.items() if not r.startswith("H")}

    issues: list[str] = []

    missing = sorted(set(sch_fps) - set(pcb_fps))
    if missing:
        issues.append(f"in schematic but not PCB: {', '.join(missing[:8])}")
    orphans = sorted(set(pcb_fps) - set(sch_fps))
    if orphans:
        issues.append(f"in PCB but not schematic: {', '.join(orphans[:8])}")

    for ref in sorted(set(sch_fps) & set(pcb_fps)):
        if sch_fps[ref] and sch_fps[ref] != pcb_fps[ref]:
            issues.append(
                f"{ref} footprint differs: schematic={sch_fps[ref]!r} "
                f"pcb={pcb_fps[ref]!r} (update would replace the footprint)"
            )

    common_refs = set(sch_fps) & set(pcb_fps)
    checked_nets = 0
    for (ref, pin), net in sorted(sch_pin_nets.items()):
        if ref not in common_refs:
            continue
        pcb_net = pcb_pad_nets.get((ref, pin))
        if pcb_net is None:
            # Unconnected schematic pins carry KiCad auto-names; a pad
            # with no net entry on the PCB matches "no connection".
            if net.startswith("unconnected-") or not net:
                continue
            issues.append(f"{ref}.{pin}: net {net!r} in schematic, no net on PCB pad")
            continue
        checked_nets += 1
        if net != pcb_net and not net.startswith("unconnected-"):
            issues.append(
                f"{ref}.{pin}: net differs schematic={net!r} pcb={pcb_net!r}"
            )

    return SyncReport(
        passed=not issues,
        issues=tuple(issues),
        components_checked=len(common_refs),
        nets_checked=checked_nets,
    )


__all__ = ["SyncReport", "check_written_sync", "find_kicad_cli"]

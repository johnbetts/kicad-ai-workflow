"""Project-local footprint library generation.

Generates a ``.pretty`` directory containing ``.kicad_mod`` files for every
unique footprint in the project, plus an ``fp-lib-table`` that registers it
with KiCad.  This enables KiCad's "Update PCB from Schematic" workflow by
making all footprint ``lib_id`` references resolvable through the library
system.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kicad_pipeline.constants import (
    KICAD_GENERATOR,
    KICAD_GENERATOR_VERSION,
    KICAD_PCB_VERSION,
)
from kicad_pipeline.pcb.footprints import footprint_for_component
from kicad_pipeline.sexp.writer import SExpNode, write

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.requirements import ProjectRequirements

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def footprint_name_from_lib_id(lib_id: str) -> str:
    """Extract the footprint name from a ``lib_id``.

    ``"Resistor_SMD:R_0805_2012Metric"`` → ``"R_0805_2012Metric"``
    ``"kicad-ai:R_0805"`` → ``"R_0805"``
    ``"R_0805"`` → ``"R_0805"``

    Args:
        lib_id: A KiCad footprint library identifier.

    Returns:
        The bare footprint name (after the colon, or the whole string).
    """
    return lib_id.split(":")[-1] if ":" in lib_id else lib_id


def remap_footprint_lib_ids(
    requirements: ProjectRequirements,
    project_name: str,
) -> dict[str, str]:
    """Build a mapping from current lib_ids to project-local lib_ids.

    Covers all prefix styles: ``Resistor_SMD:X``, ``kicad-ai:X``,
    ``easyeda2kicad:X``, ``jlcpcb:X``, and bare names.

    Args:
        requirements: Project requirements with all components.
        project_name: Project name used as the library prefix.

    Returns:
        Dict mapping old lib_id → new ``{project_name}:{footprint_name}``.
    """
    mapping: dict[str, str] = {}
    for comp in requirements.components:
        old_id = comp.footprint
        fp_name = footprint_name_from_lib_id(old_id)
        new_id = f"{project_name}:{fp_name}"
        if old_id not in mapping:
            mapping[old_id] = new_id
    return mapping


def footprint_to_kicad_mod(fp_sexp: list[SExpNode], footprint_name: str) -> str:
    """Convert a footprint S-expression to standalone ``.kicad_mod`` content.

    Strips instance-specific data:

    - Removes ``(net ...)`` from pads
    - Removes board-relative ``(at x y rot)`` from the top-level footprint
    - Replaces instance Reference with ``REF**``
    - Replaces instance Value with *footprint_name*
    - Adds ``(version ...)``, ``(generator ...)``, ``(generator_version ...)``

    Args:
        fp_sexp: S-expression list from ``_footprint_sexp()``.
        footprint_name: Name for this footprint (used as the first element
            and the Value property text).

    Returns:
        S-expression string for a standalone ``.kicad_mod`` file.
    """
    # Deep-copy to avoid mutating the original
    import copy

    node = copy.deepcopy(fp_sexp)

    # Replace lib_id (element 1) with bare footprint name
    node[1] = footprint_name

    # Remove (at x y rot) from the top-level footprint
    node[:] = [
        child
        for child in node
        if not (isinstance(child, list) and len(child) >= 1 and child[0] == "at")
    ]

    # Insert version/generator after the footprint name (element 1)
    insert_idx = 2
    headers: list[SExpNode] = [
        ["version", KICAD_PCB_VERSION],
        ["generator", KICAD_GENERATOR],
        ["generator_version", KICAD_GENERATOR_VERSION],
    ]
    for item in headers:
        node.insert(insert_idx, item)
        insert_idx += 1

    # Property name -> replacement value mapping
    _property_overrides: dict[str, str] = {
        "Reference": "REF**",
        "Value": footprint_name,
        "Footprint": "",
    }

    # Walk children to strip nets from pads and fix properties
    for i, child in enumerate(node):
        if not isinstance(child, list) or not child:
            continue

        tag = child[0]

        # Strip (net N "name") and instance UUIDs from pad nodes
        if tag == "pad":
            node[i] = [
                elem
                for elem in child
                if not (isinstance(elem, list) and len(elem) >= 1 and elem[0] in ("net", "uuid"))
            ]

        # Fix properties using dispatch dict
        if tag == "property" and len(child) >= 3:
            override = _property_overrides.get(str(child[1]))
            if override is not None:
                child[2] = override

    # Remove top-level uuid (instance UUID)
    node[:] = [
        child
        for child in node
        if not (isinstance(child, list) and len(child) >= 1 and child[0] == "uuid")
    ]

    # Add embedded_fonts at the end
    node.append(["embedded_fonts", False])

    return write(node)


def build_footprint_library(
    requirements: ProjectRequirements,
    project_dir: Path,
    project_name: str,
) -> Path:
    """Generate a project-local ``.pretty`` footprint library.

    For every unique footprint in *requirements*:

    1. Generate the footprint via :func:`footprint_for_component`.
    2. Serialise it to S-expression via the PCB builder.
    3. Strip instance data and write as a standalone ``.kicad_mod`` file.

    Args:
        requirements: Project requirements with all components.
        project_dir: Path to the project root directory.
        project_name: Project name (used as library name).

    Returns:
        Path to the generated ``.pretty`` directory.
    """
    from kicad_pipeline.pcb.builder import _footprint_sexp

    pretty_dir = project_dir / f"{project_name}.pretty"
    pretty_dir.mkdir(parents=True, exist_ok=True)

    # Group by footprint ID to deduplicate
    seen: set[str] = set()
    for comp in requirements.components:
        fp_name = footprint_name_from_lib_id(comp.footprint)
        if fp_name in seen:
            continue
        seen.add(fp_name)

        fp = footprint_for_component(
            comp.ref,
            comp.value,
            comp.footprint,
            comp.lcsc,
        )
        fp_sexp = _footprint_sexp(fp)
        assert isinstance(fp_sexp, list)  # always a list node
        kicad_mod = footprint_to_kicad_mod(fp_sexp, fp_name)
        mod_path = pretty_dir / f"{fp_name}.kicad_mod"
        mod_path.write_text(kicad_mod, encoding="utf-8")
        log.debug("Wrote footprint: %s", mod_path)

    log.info(
        "Footprint library: %d footprints in %s",
        len(seen),
        pretty_dir,
    )
    return pretty_dir


def write_fp_lib_table(
    project_dir: Path,
    project_name: str,
) -> Path:
    """Write ``fp-lib-table`` registering the project-local library.

    Args:
        project_dir: Path to the project root directory.
        project_name: Library name (matches ``.pretty`` directory).

    Returns:
        Path to the generated ``fp-lib-table`` file.
    """
    table: list[SExpNode] = [
        "fp_lib_table",
        ["version", 7],
        [
            "lib",
            ["name", project_name],
            ["type", "KiCad"],
            ["uri", f"${{KIPRJMOD}}/{project_name}.pretty"],
            ["options", ""],
            ["descr", "Project-local footprints generated by kicad-ai-pipeline"],
        ],
    ]
    fp_lib_path = project_dir / "fp-lib-table"
    fp_lib_path.write_text(write(table), encoding="utf-8")
    log.info("fp-lib-table written: %s", fp_lib_path)
    return fp_lib_path


# ---------------------------------------------------------------------------
# Schematic ↔ PCB path synchronisation
# ---------------------------------------------------------------------------


def _extract_ref_uuid_map(sch_path: Path) -> dict[str, str]:
    """Extract a ref → symbol UUID mapping from a KiCad schematic file.

    Parses the schematic to find each ``(symbol ...)`` block's reference
    designator and its UUID.  Only top-level symbols are included (not
    lib_symbol definitions or power symbols).

    Args:
        sch_path: Path to a ``.kicad_sch`` file.

    Returns:
        Dict mapping reference designator (e.g. ``"C20"``) to its symbol
        UUID string.
    """
    import re

    text = sch_path.read_text(encoding="utf-8")

    ref_uuid: dict[str, str] = {}

    # Match top-level (symbol ...) blocks that have a (lib_id ...) and
    # Reference property.  Each symbol ends with (uuid "...") just before
    # its closing paren.
    #
    # Strategy: find all (property "Reference" "XX" ...) and then find
    # the nearest (uuid "...") that closes the same symbol block.

    # Split into symbol blocks by finding "(symbol\n" or "(symbol " at
    # top indentation (2 spaces).
    symbol_starts = [m.start() for m in re.finditer(r"(?m)^  \(symbol\b", text)]
    for start in symbol_starts:
        # Find the end of this symbol block by matching parens
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        block = text[start:end]

        # Skip lib_symbol definitions (they have (symbol "lib_name" inside)
        if "(lib_id" not in block:
            continue

        # Extract Reference
        ref_match = re.search(
            r'\(property\s+"Reference"\s+"([^"]+)"', block
        )
        if not ref_match:
            continue
        ref = ref_match.group(1)

        # Extract the LAST uuid in the block (that's the symbol's own UUID)
        uuid_matches = list(re.finditer(r'\(uuid\s+"([^"]+)"\)', block))
        if not uuid_matches:
            continue
        sym_uuid = uuid_matches[-1].group(1)

        ref_uuid[ref] = sym_uuid

    return ref_uuid


def sync_pcb_to_schematic(pcb_path: Path, sch_path: Path) -> int:
    """Patch a PCB file to add ``(path ...)`` entries linking to schematic symbols.

    KiCad's "Update PCB from Schematic" uses the ``(path "/symbol_uuid")``
    field inside each PCB footprint to correlate it with the corresponding
    schematic symbol.  Without this field, KiCad cannot match existing
    footprints and tries to re-add them, failing with "footprint not found".

    This function reads the schematic to build a ref→UUID mapping, then
    patches the PCB file in-place to insert ``(path ...)`` entries.

    Args:
        pcb_path: Path to the ``.kicad_pcb`` file.
        sch_path: Path to the ``.kicad_sch`` file.

    Returns:
        Number of footprints patched.
    """
    import re

    if not sch_path.exists():
        log.warning("sync_pcb_to_schematic: schematic not found at %s", sch_path)
        return 0

    ref_uuid = _extract_ref_uuid_map(sch_path)
    if not ref_uuid:
        log.warning("sync_pcb_to_schematic: no symbols found in schematic")
        return 0

    pcb_text = pcb_path.read_text(encoding="utf-8")
    patched = 0

    # For each footprint in the PCB, find its Reference property and
    # insert a (path "/uuid") before the closing paren of the footprint.
    # We work line-by-line to preserve formatting.
    lines = pcb_text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        result.append(line)

        # Detect the start of a footprint block at indent level 2
        if re.match(r'^  \(footprint\s', line):
            # Scan the footprint block to find its Reference and where it ends
            depth = line.count("(") - line.count(")")
            j = i + 1
            ref_found: str | None = None
            has_path = False
            while j < len(lines) and depth > 0:
                fp_line = lines[j]
                depth += fp_line.count("(") - fp_line.count(")")

                # Look for Reference property
                ref_match = re.search(
                    r'\(property\s+"Reference"\s+"([^"]+)"', fp_line
                )
                if ref_match:
                    ref_found = ref_match.group(1)

                # Check if path already exists
                if re.search(r'\(path\s+"', fp_line):
                    has_path = True

                if depth == 0 and ref_found and not has_path:
                    # This is the closing line of the footprint.
                    # Insert (path ...) before it.
                    sym_uuid = ref_uuid.get(ref_found)
                    if sym_uuid:
                        result.append(f'    (path "/{sym_uuid}")')
                        patched += 1

                result.append(fp_line)
                j += 1

            i = j
            continue

        i += 1

    if patched > 0:
        pcb_path.write_text("\n".join(result), encoding="utf-8")
        log.info(
            "sync_pcb_to_schematic: patched %d footprints with schematic paths",
            patched,
        )

    return patched

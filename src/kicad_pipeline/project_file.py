"""Generate KiCad 9 project files (.kicad_pro).

The .kicad_pro file is a JSON file that ties together the schematic,
PCB, and project settings. KiCad uses filename conventions to link
files: the .kicad_pro, .kicad_sch, and .kicad_pcb must share the
same base name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path  # noqa: TC003 — used at runtime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import NetClass

log = logging.getLogger(__name__)


def build_project_file(
    project_name: str,
    root_uuid: str = "",
    netclasses: tuple[NetClass, ...] | None = None,
    drc_exclusions: tuple[str, ...] | None = None,
    layer_count: int = 2,
) -> dict[str, Any]:
    """Build a minimal KiCad 9 project file structure.

    Args:
        project_name: Project name (used for filename field).
        root_uuid: UUID of the root schematic sheet. If empty,
            KiCad will assign one on first open.
        netclasses: Optional netclass definitions to include in
            the project file's net_settings section.
        drc_exclusions: Optional list of DRC exclusion strings
            (e.g. intra-footprint clearance exclusions).
        layer_count: Number of copper layers (2 or 4).

    Returns:
        A dictionary suitable for JSON serialisation as a .kicad_pro file.
    """
    sheets: list[list[str]] = []
    if root_uuid:
        sheets.append([root_uuid, "Root"])

    data: dict[str, Any] = {
        "board": {
            "3dviewports": [],
            "design_settings": {
                "defaults": {
                    "apply_defaults_to_fp_fields": False,
                    "apply_defaults_to_fp_shapes": False,
                    "apply_defaults_to_fp_text": False,
                    "board_outline_line_width": 0.15,
                    "copper_line_width": 0.2,
                    "copper_text_italic": False,
                    "copper_text_size_h": 1.0,
                    "copper_text_size_v": 1.0,
                    "copper_text_thickness": 0.3,
                    "copper_text_upright": False,
                    "courtyard_line_width": 0.05,
                    "dimension_precision": 4,
                    "dimension_units": 3,
                    "fab_line_width": 0.1,
                    "fab_text_italic": False,
                    "fab_text_size_h": 1.0,
                    "fab_text_size_v": 1.0,
                    "fab_text_thickness": 0.15,
                    "fab_text_upright": False,
                    "other_line_width": 0.1,
                    "other_text_italic": False,
                    "other_text_size_h": 1.0,
                    "other_text_size_v": 1.0,
                    "other_text_thickness": 0.15,
                    "other_text_upright": False,
                    "silk_line_width": 0.15,
                    "silk_text_italic": False,
                    "silk_text_size_h": 1.0,
                    "silk_text_size_v": 1.0,
                    "silk_text_thickness": 0.15,
                    "silk_text_upright": False,
                    "zones": {
                        "45_degree_only": False,
                        "min_clearance": 0.2,
                    },
                },
                "diff_pair_dimensions": [],
                "drc_exclusions": [],
                "meta": {
                    "filename": "board_design_settings.json",
                    "version": 2,
                },
                "rule_severities": {
                    "annular_width": "error",
                    "clearance": "error",
                    "copper_edge_clearance": "error",
                    "courtyards_overlap": "warning",
                    "drill_out_of_range": "error",
                    "duplicate_footprints": "warning",
                    "hole_clearance": "error",
                    "hole_near_hole": "error",
                    "invalid_outline": "error",
                    "isolated_copper": "warning",
                    "item_on_disabled_layer": "error",
                    "lib_footprint_issues": "ignore",
                    "lib_footprint_mismatch": "ignore",
                    "missing_courtyard": "ignore",
                    "missing_footprint": "warning",
                    "shorting_items": "error",
                    "silk_overlap": "warning",
                    "solder_mask_bridge": "warning",
                    "track_dangling": "warning",
                    "track_width": "error",
                    "unconnected_items": "error",
                    "unresolved_variable": "error",
                    "via_dangling": "warning",
                    "zones_intersect": "error",
                },
                "rules": {
                    "allow_blind_buried_vias": False,
                    "allow_microvias": False,
                    "max_error": 0.005,
                    "min_clearance": 0.0,
                    "min_connection": 0.0,
                    "min_copper_edge_clearance": 0.3,
                    "min_hole_clearance": 0.0,
                    "min_hole_to_hole": 0.2,
                    "min_microvia_diameter": 0.508,
                    "min_microvia_drill": 0.127,
                    "min_resolved_spokes": 2,
                    "min_silk_clearance": 0.0,
                    "min_text_height": 0.8,
                    "min_text_thickness": 0.08,
                    "min_through_hole_diameter": 0.3,
                    "min_track_width": 0.127,
                    "min_via_annular_width": 0.05,
                    "min_via_diameter": 0.6,
                    "solder_mask_clearance": 0.05,
                    "solder_mask_min_width": 0.1,
                    "solder_mask_to_copper_clearance": 0.05,
                    "use_height_for_length_calcs": True,
                },
                "track_widths": [0.0, 0.2, 0.25, 0.4, 0.5],
                "via_dimensions": [
                    {"diameter": 0.0, "drill": 0.0},
                    {"diameter": 0.6, "drill": 0.3},
                    {"diameter": 0.8, "drill": 0.508},
                ],
                "zones_allow_external_fillets": False,
                "zones_use_no_outline": True,
            },
            "layer_pairs": [],
            "layer_presets": [],
            "viewports": [],
        },
        "boards": [],
        "cvpcb": {"equivalence_files": []},
        "libraries": {
            "pinned_footprint_libs": [],
            "pinned_symbol_libs": [],
        },
        "meta": {
            "filename": f"{project_name}.kicad_pro",
            "version": 3,
        },
        "net_settings": {
            "classes": [
                {
                    "bus_width": 12,
                    "clearance": 0.2,
                    "diff_pair_gap": 0.25,
                    "diff_pair_via_gap": 0.25,
                    "diff_pair_width": 0.2,
                    "line_style": 0,
                    "microvia_diameter": 0.508,
                    "microvia_drill": 0.127,
                    "name": "Default",
                    "pcb_color": "rgba(0, 0, 0, 0.000)",
                    "priority": 2147483647,
                    "schematic_color": "rgba(0, 0, 0, 0.000)",
                    "track_width": 0.25,
                    "via_diameter": 0.8,
                    "via_drill": 0.508,
                    "wire_width": 6,
                }
            ],
            "meta": {"version": 4},
            "net_colors": None,
            "netclass_assignments": None,
            "netclass_patterns": [],
        },
        "pcbnew": {
            "last_paths": {
                "gencad": "",
                "idf": "",
                "netlist": "",
                "plot": "",
                "pos_files": "",
                "specctra_dsn": "",
                "step": "",
                "svg": "",
                "vrml": "",
            },
            "page_layout_descr_file": "",
        },
        "schematic": {
            "annotate_start_num": 0,
            "bom_export_filename": "${PROJECTNAME}.csv",
            "bom_fmt_presets": [],
            "bom_fmt_settings": {
                "field_delimiter": ",",
                "keep_line_breaks": False,
                "keep_tabs": False,
                "name": "CSV",
                "ref_delimiter": ",",
                "ref_range_delimiter": "",
                "string_delimiter": "\"",
            },
            "bom_presets": [],
            "bom_settings": {
                "exclude_dnp": False,
                "fields_ordered": [
                    {
                        "group_by": False,
                        "label": "Reference",
                        "name": "Reference",
                        "show": True,
                    },
                    {
                        "group_by": True,
                        "label": "Value",
                        "name": "Value",
                        "show": True,
                    },
                    {
                        "group_by": False,
                        "label": "Datasheet",
                        "name": "Datasheet",
                        "show": True,
                    },
                    {
                        "group_by": False,
                        "label": "Footprint",
                        "name": "Footprint",
                        "show": True,
                    },
                    {
                        "group_by": False,
                        "label": "Qty",
                        "name": "${QUANTITY}",
                        "show": True,
                    },
                ],
                "filter_string": "",
                "group_symbols": True,
                "include_excluded_from_bom": False,
                "name": "Grouped By Value",
                "sort_asc": True,
                "sort_field": "Reference",
            },
            "connection_grid_size": 50.0,
            "drawing": {
                "dashed_lines_dash_length_ratio": 12.0,
                "dashed_lines_gap_length_ratio": 3.0,
                "default_bus_thickness": 12.0,
                "default_junction_size": 40.0,
                "default_line_thickness": 6.0,
                "default_text_size": 50.0,
                "default_wire_thickness": 6.0,
                "field_names": [],
                "intersheets_ref_own_page": False,
                "intersheets_ref_prefix": "",
                "intersheets_ref_short": False,
                "intersheets_ref_show": False,
                "intersheets_ref_suffix": "",
                "junction_size_choice": 3,
                "label_size_ratio": 0.3,
                "overbar_offset_ratio": 1.23,
                "pin_symbol_size": 25.0,
                "text_offset_ratio": 0.3,
            },
            "legacy_lib_dir": "",
            "legacy_lib_list": [],
            "meta": {"version": 1},
            "net_format_name": "",
            "page_layout_descr_file": "",
            "plot_directory": "",
            "spice_adjust_passive_values": False,
            "spice_current_sheet_as_root": False,
            "spice_external_command": "spice \"%I\"",
            "spice_model_current_sheet_as_root": True,
            "spice_save_all_currents": False,
            "spice_save_all_voltages": False,
            "subpart_first_id": 65,
            "subpart_id_separator": 0,
        },
        "sheets": sheets,
        "text_variables": {},
    }

    # Inject DRC exclusions
    if drc_exclusions:
        data["board"]["design_settings"]["drc_exclusions"] = list(drc_exclusions)

    # Inject additional netclass definitions and assignments
    if netclasses:
        classes = data["net_settings"]["classes"]
        assignments: dict[str, str] = {}
        for nc in netclasses:
            if nc.name == "Default":
                # Update the existing Default class values
                classes[0]["track_width"] = nc.trace_width_mm
                classes[0]["clearance"] = nc.clearance_mm
                classes[0]["via_diameter"] = nc.via_diameter_mm
                classes[0]["via_drill"] = nc.via_drill_mm
                continue
            classes.append({
                "bus_width": 12,
                "clearance": nc.clearance_mm,
                "diff_pair_gap": nc.diff_pair_gap_mm,
                "diff_pair_via_gap": 0.25,
                "diff_pair_width": nc.diff_pair_width_mm,
                "line_style": 0,
                "microvia_diameter": 0.508,
                "microvia_drill": 0.127,
                "name": nc.name,
                "pcb_color": "rgba(0, 0, 0, 0.000)",
                "priority": 2147483647,
                "schematic_color": "rgba(0, 0, 0, 0.000)",
                "track_width": nc.trace_width_mm,
                "via_diameter": nc.via_diameter_mm,
                "via_drill": nc.via_drill_mm,
                "wire_width": 6,
            })
            for net_name in nc.nets:
                assignments[net_name] = nc.name

        if assignments:
            data["net_settings"]["netclass_assignments"] = assignments

    # Inject layer definitions for 4-layer stackup
    if layer_count >= 4:
        data["board"]["design_settings"]["layers"] = {
            "F.Cu": {"name": "F.Cu", "type": 0},
            "In1.Cu": {"name": "In1.Cu", "type": 1},
            "In2.Cu": {"name": "In2.Cu", "type": 1},
            "B.Cu": {"name": "B.Cu", "type": 0},
        }

    return data


def write_project_file(
    project_name: str,
    directory: Path,
    root_uuid: str = "",
    netclasses: tuple[NetClass, ...] | None = None,
    drc_exclusions: tuple[str, ...] | None = None,
    layer_count: int = 2,
) -> Path:
    """Write a .kicad_pro file to the given directory.

    Args:
        project_name: Base name for the project file.
        directory: Directory to write the file into.
        root_uuid: UUID of the root schematic sheet.
        netclasses: Optional netclass definitions to include.
        drc_exclusions: Optional DRC exclusion strings.
        layer_count: Number of copper layers (2 or 4).

    Returns:
        Path to the written .kicad_pro file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    pro_path = directory / f"{project_name}.kicad_pro"
    data = build_project_file(
        project_name, root_uuid, netclasses=netclasses, drc_exclusions=drc_exclusions,
        layer_count=layer_count,
    )
    pro_path.write_text(json.dumps(data, indent=2) + "\n")
    log.info("Project file written: %s", pro_path)
    return pro_path

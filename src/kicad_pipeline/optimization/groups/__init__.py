"""Component group placement optimization modules.

Contains specialized placement logic for different component types:
- MCU placement and power decoupling
- Power supply organization
- ADC channel layout
- Ethernet interface placement
- Connector positioning
- General group utilities
"""

from .helpers import (
    _build_exclusion_grid,
    _build_net_to_group_refs,
    _build_ref_net_adjacency,
    _caps_on_nets,
    _clamp,
    _clamp_to_bounds,
    _classify_eth_caps,
    _classify_refs_by_prefix,
    _collect_feature_refs,
    _expand_one_hop,
    _find_zone_rect,
    _is_power_or_bus_net,
    _is_small_passive,
    _nets_connected_to_prefix,
    _nets_connected_to_ref,
    _place_component_with_grid,
    _place_connector_at_edge,
    _place_refs_in_column_grid,
    _push_component_outside_courtyard,
    _push_non_group_away_from_ic,
    _should_add_ref,
)
"""Design review module — generates actionable checklists from requirements and PCB data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import Component, ProjectRequirements

# Patterns for identifying power nets.
_POWER_NET_PREFIXES: tuple[str, ...] = ("+", "V")
_GND_NET_NAMES: frozenset[str] = frozenset({"GND", "AGND", "DGND", "GNDA", "GNDD"})
_WIFI_KEYWORDS: frozenset[str] = frozenset({"ESP32", "WROOM", "BLE", "WIFI", "WI-FI"})
_REGULATOR_KEYWORDS: frozenset[str] = frozenset({
    "LDO", "BUCK", "BOOST", "REGULATOR", "AMS1117", "LP5907", "MCP1700",
    "AP2112", "XC6206", "TPS7A", "TPS54", "LM1117", "NCP1117", "RT9080",
})


@dataclass(frozen=True)
class ReviewItem:
    """A single design review finding."""

    category: str  # "antenna", "relay", "power", "thermal", "mechanical"
    severity: str  # "required", "recommended", "optional"
    title: str
    description: str
    affected_refs: tuple[str, ...]


@dataclass(frozen=True)
class ComponentGroup:
    """A logical grouping of components (feature block or subcircuit).

    Attributes:
        name: Group name (e.g. "Power Supply", "MCU").
        description: Brief description of the group's function.
        refs: Component reference designators in this group.
        subgroups: Named subgroups within this group (e.g. decoupling caps
            near a specific IC, or a regulator subcircuit).
    """

    name: str
    description: str
    refs: tuple[str, ...]
    subgroups: tuple[ComponentGroup, ...] = ()


@dataclass(frozen=True)
class BoardSummary:
    """High-level board characteristics extracted from the design."""

    board_size_mm: tuple[float, float]
    component_count: int
    layer_count: int
    unique_nets: int
    power_nets: tuple[str, ...]
    has_wifi: bool
    has_relays: bool
    has_adc: bool


@dataclass(frozen=True)
class DesignReview:
    """Complete design review containing summary and actionable items."""

    board_summary: BoardSummary
    items: tuple[ReviewItem, ...]
    component_groups: tuple[ComponentGroup, ...] = ()


def _is_power_net(name: str) -> bool:
    """Return True if ``name`` looks like a power or ground net."""
    upper = name.upper()
    if upper in _GND_NET_NAMES:
        return True
    return any(upper.startswith(prefix) for prefix in _POWER_NET_PREFIXES)


def _has_wifi_component(requirements: ProjectRequirements) -> tuple[bool, tuple[str, ...]]:
    """Check if any component value contains a WiFi-related keyword.

    Returns:
        A tuple of (has_wifi, affected_refs).
    """
    refs: list[str] = []
    for comp in requirements.components:
        upper_value = comp.value.upper()
        upper_fp = comp.footprint.upper()
        for kw in _WIFI_KEYWORDS:
            if kw in upper_value or kw in upper_fp:
                refs.append(comp.ref)
                break
    return (len(refs) > 0, tuple(refs))


def _find_relay_refs(requirements: ProjectRequirements) -> tuple[str, ...]:
    """Return refs for relay components (K* designators)."""
    return tuple(comp.ref for comp in requirements.components if comp.ref.startswith("K"))


def _find_power_nets(requirements: ProjectRequirements) -> tuple[str, ...]:
    """Return sorted tuple of power-related net names from the design."""
    power: set[str] = set()
    for net in requirements.nets:
        if _is_power_net(net.name):
            power.add(net.name)
    return tuple(sorted(power))


def _find_regulator_refs(requirements: ProjectRequirements) -> tuple[str, ...]:
    """Return refs for voltage regulators (U* with power-related values)."""
    refs: list[str] = []
    for comp in requirements.components:
        if not comp.ref.startswith("U"):
            continue
        upper_value = comp.value.upper()
        upper_desc = (comp.description or "").upper()
        for kw in _REGULATOR_KEYWORDS:
            if kw in upper_value or kw in upper_desc:
                refs.append(comp.ref)
                break
    return tuple(refs)


_REGULATOR_DESC_KEYWORDS: frozenset[str] = frozenset({
    "BUCK", "BOOST", "LDO", "REGULATOR", "CONVERTER", "SWITCHING",
    "LINEAR REG", "VOLTAGE REG", "POWER SUPPLY",
})

# Maximum number of ICs sharing a power rail before we treat it as a "bus"
# and refuse to pair caps via rail matching alone.
_MAX_IC_COUNT_FOR_RAIL_MATCH = 2


def _match_cap_by_description(
    desc: str,
    ic_refs: list[str],
    comp_map: dict[str, object],
) -> tuple[str, float]:
    """Priority 1: Match cap to IC whose value name appears in cap description."""
    for ic in ic_refs:
        ic_comp = comp_map.get(ic)
        if ic_comp is None:
            continue
        ic_value = ic_comp.value.upper()
        if ic_value and ic_value in desc:
            return ic, 1000.0
    return "", 0.0


def _match_cap_by_regulator(
    desc: str,
    cap_rails: set[str],
    cap_feature: str,
    ic_refs: list[str],
    regulator_set: set[str],
    ref_power_nets: dict[str, set[str]],
    ref_to_feature: dict[str, str],
) -> tuple[str, float]:
    """Priority 2: Match cap with regulator keywords to a regulator on same rail."""
    if not any(kw in desc for kw in _REGULATOR_DESC_KEYWORDS):
        return "", 0.0
    best_ic = ""
    best_score = 0.0
    for ic in ic_refs:
        if ic not in regulator_set:
            continue
        ic_nets = ref_power_nets.get(ic, set())
        if not (cap_rails & ic_nets):
            continue
        ic_feature = ref_to_feature.get(ic, "")
        feature_bonus = 10.0 if (cap_feature and cap_feature == ic_feature) else 0.0
        score = 500.0 + feature_bonus
        if score > best_score:
            best_score = score
            best_ic = ic
    return best_ic, best_score


def _match_cap_by_rail(
    cap_rails: set[str],
    cap_feature: str,
    ic_refs: list[str],
    comp_map: dict[str, object],
    ref_power_nets: dict[str, set[str]],
    net_ic_count: dict[str, int],
    ref_to_feature: dict[str, str],
) -> tuple[str, float]:
    """Priority 3: Match cap to IC sharing a specific (low-user-count) rail."""
    best_ic = ""
    best_score = 0.0
    for ic in ic_refs:
        if comp_map.get(ic) is None:
            continue
        ic_nets = ref_power_nets.get(ic, set())
        shared_rails = cap_rails & ic_nets
        if not shared_rails:
            continue
        specific_rails = {
            r for r in shared_rails
            if net_ic_count.get(r, 0) <= _MAX_IC_COUNT_FOR_RAIL_MATCH
        }
        if not specific_rails:
            continue
        score = sum(
            1.0 / max(net_ic_count.get(r, 1), 1)
            for r in specific_rails
        )
        ic_feature = ref_to_feature.get(ic, "")
        if cap_feature and cap_feature == ic_feature:
            score += 5.0
        if score > best_score:
            best_score = score
            best_ic = ic
    return best_ic, best_score


def _build_power_net_maps(
    requirements: ProjectRequirements,
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Build ref-to-power-nets mapping and net-to-IC-count mapping."""
    ref_power_nets: dict[str, set[str]] = {}
    net_ic_count: dict[str, int] = {}
    for net in requirements.nets:
        if not _is_power_net(net.name):
            continue
        ic_count = 0
        for conn in net.connections:
            ref_power_nets.setdefault(conn.ref, set()).add(net.name)
            if conn.ref.startswith("U"):
                ic_count += 1
        net_ic_count[net.name] = ic_count
    return ref_power_nets, net_ic_count


def _find_decoupling_caps(
    cap_refs: list[str],
    comp_map: dict[str, Component],
) -> list[str]:
    """Filter caps to those that look like decoupling (both pins on power nets)."""
    result: list[str] = []
    for cref in cap_refs:
        comp = comp_map.get(cref)
        if comp is None or len(comp.pins) != 2:
            continue
        pin_nets = [p.net for p in comp.pins if p.net]
        if len(pin_nets) == 2 and all(_is_power_net(n) for n in pin_nets):
            result.append(cref)
    return result


def _best_ic_for_cap(
    cap: str,
    comp_map: dict[str, Component],
    ic_refs: list[str],
    ref_power_nets: dict[str, set[str]],
    net_ic_count: dict[str, int],
    regulator_set: set[str],
    ref_to_feature: dict[str, str],
) -> str | None:
    """Find the best IC match for a decoupling cap using tiered matching."""
    cap_comp = comp_map.get(cap)
    cap_nets = ref_power_nets.get(cap, set())
    cap_rails = {n for n in cap_nets if n.upper() not in _GND_NET_NAMES}
    desc = (cap_comp.description or "").upper() if cap_comp else ""
    cap_feature = ref_to_feature.get(cap, "")

    best_ic, best_score = _match_cap_by_description(desc, ic_refs, comp_map)

    if best_score < 1000.0:
        ic, score = _match_cap_by_regulator(
            desc, cap_rails, cap_feature, ic_refs,
            regulator_set, ref_power_nets, ref_to_feature,
        )
        if score > best_score:
            best_ic, best_score = ic, score

    if best_score < 500.0:
        ic, score = _match_cap_by_rail(
            cap_rails, cap_feature, ic_refs, comp_map,
            ref_power_nets, net_ic_count, ref_to_feature,
        )
        if score > best_score:
            best_ic = ic

    return best_ic


def _find_ic_decoupling_pairs(
    requirements: ProjectRequirements,
) -> list[tuple[str, str]]:
    """Find (IC_ref, cap_ref) pairs for decoupling cap placement checks.

    Pairing strategy (in priority order):
    1. **Description match**: cap description mentions an IC value name
       (e.g. "ESP32 decoupling" → U3 whose value is ESP32-S3-WROOM-1).
    2. **Regulator description**: cap description contains regulator keywords
       (e.g. "3.3V buck output cap") → paired with the regulator in the same
       feature block or sharing the cap's power rail.
    3. **Specific rail match**: cap shares a non-GND power rail with an IC,
       and that rail has at most 2 IC users (specific rail, not a bus).
    4. **Skip**: caps on busy shared rails (≥3 ICs) or bulk/section caps
       are not paired to avoid misleading recommendations.
    """
    comp_map = {c.ref: c for c in requirements.components}
    ref_power_nets, net_ic_count = _build_power_net_maps(requirements)

    ic_refs = [c.ref for c in requirements.components if c.ref.startswith("U")]
    cap_refs = [c.ref for c in requirements.components if c.ref.startswith("C")]

    regulator_set: set[str] = set(_find_regulator_refs(requirements))

    ref_to_feature: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            ref_to_feature[ref] = fb.name

    decoupling_caps = _find_decoupling_caps(cap_refs, comp_map)

    pairs: list[tuple[str, str]] = []
    seen_caps: set[str] = set()

    for cap in decoupling_caps:
        best_ic = _best_ic_for_cap(
            cap, comp_map, ic_refs, ref_power_nets, net_ic_count,
            regulator_set, ref_to_feature,
        )
        if best_ic and cap not in seen_caps:
            pairs.append((best_ic, cap))
            seen_caps.add(cap)

    return pairs


_ADC_VALUE_KEYWORDS: frozenset[str] = frozenset({"ADC", "ADS1", "MCP3"})
_ADC_PIN_FUNCTIONS: frozenset[str] = frozenset({"adc", "analog_in"})


def _has_adc_component(requirements: ProjectRequirements) -> bool:
    """Return True if any component has ADC-related pins or values."""
    for comp in requirements.components:
        has_adc_pin = any(
            pin.function is not None and pin.function.value in _ADC_PIN_FUNCTIONS
            for pin in comp.pins
        )
        if has_adc_pin:
            return True
        upper_value = comp.value.upper()
        has_adc_value = any(kw in upper_value for kw in _ADC_VALUE_KEYWORDS)
        if has_adc_value:
            return True
    return False


def _check_partial_connectivity(
    comp: object,
    connectable_pins: list[object],
    connected_pins: set[tuple[str, str]],
) -> list[ReviewItem]:
    """Check for partially unconnected signal pins on a component.

    Returns a list containing at most one ReviewItem if floating signal
    pins are detected.
    """
    unconnected = [
        p.number for p in connectable_pins
        if (comp.ref, p.number) not in connected_pins
    ]
    # Build a set of power pin numbers for fast lookup
    power_pin_numbers = {
        p.number for p in connectable_pins
        if p.pin_type.value in ("power_in", "power_out")
    }
    signal_unconnected = [
        p_num for p_num in unconnected
        if p_num not in power_pin_numbers
    ]
    if not signal_unconnected:
        return []
    pin_list = ", ".join(signal_unconnected[:5])
    extra = len(signal_unconnected) - 5
    suffix = f" +{extra} more" if extra > 0 else ""
    return [ReviewItem(
        category="connectivity",
        severity="recommended",
        title="Partially unconnected component",
        description=(
            f"{comp.ref} ({comp.value}) has {len(signal_unconnected)} "
            f"unconnected signal pins: {pin_list}{suffix}"
        ),
        affected_refs=(comp.ref,),
    )]


def _check_connectivity(
    requirements: ProjectRequirements,
) -> list[ReviewItem]:
    """Check for unconnected components and dead-end nets.

    Detects:
    - Components where ALL non-NC pins lack net connections (fully unconnected).
    - Components not listed in any feature block (orphaned).
    - Signal nets with only one connection (dead-end, excluding power nets).

    Args:
        requirements: Project requirements to validate.

    Returns:
        List of :class:`ReviewItem` findings.
    """
    items: list[ReviewItem] = []

    # Build set of (ref, pin) pairs that appear in any net
    connected_pins: set[tuple[str, str]] = set()
    for net in requirements.nets:
        for conn in net.connections:
            connected_pins.add((conn.ref, conn.pin))

    # --- Check for fully unconnected components ---
    for comp in requirements.components:
        if not comp.pins:
            continue
        # Count non-NC pins that have net connections
        connectable_pins = [
            p for p in comp.pins
            if p.pin_type.value != "no_connect"
        ]
        if not connectable_pins:
            continue
        connected_count = sum(
            1 for p in connectable_pins
            if (comp.ref, p.number) in connected_pins
        )
        is_fully_unconnected = connected_count == 0
        is_partially_connected = 0 < connected_count < len(connectable_pins)

        if is_fully_unconnected:
            items.append(ReviewItem(
                category="connectivity",
                severity="required",
                title="Unconnected component",
                description=(
                    f"{comp.ref} ({comp.value}) has no net connections — "
                    f"all {len(connectable_pins)} connectable pins are floating"
                ),
                affected_refs=(comp.ref,),
            ))
        elif is_partially_connected:
            items.extend(
                _check_partial_connectivity(comp, connectable_pins, connected_pins)
            )

    # --- Check for orphaned components (not in any feature) ---
    if requirements.features:
        featured_refs: set[str] = set()
        for fb in requirements.features:
            featured_refs.update(fb.components)
        for comp in requirements.components:
            if comp.ref not in featured_refs:
                items.append(ReviewItem(
                    category="connectivity",
                    severity="recommended",
                    title="Orphaned component",
                    description=(
                        f"{comp.ref} ({comp.value}) is not assigned to any feature block"
                    ),
                    affected_refs=(comp.ref,),
                ))

    # --- Check for dead-end signal nets ---
    for net in requirements.nets:
        if _is_power_net(net.name):
            continue  # Power nets with one connection are normal (power symbols)
        if len(net.connections) == 1:
            conn = net.connections[0]
            items.append(ReviewItem(
                category="connectivity",
                severity="required",
                title="Dead-end net",
                description=(
                    f"Net '{net.name}' has only one connection ({conn.ref}.{conn.pin}) "
                    f"— signal goes nowhere"
                ),
                affected_refs=(conn.ref,),
            ))

    return items


def _subcircuit_design_notes(
    requirements: ProjectRequirements,
) -> list[ReviewItem]:
    """Generate design notes based on detected subcircuit types.

    Subcircuit types are detected from ``FeatureBlock.subcircuits`` metadata
    and from component patterns in the netlist.

    Args:
        requirements: Project requirements to analyze.

    Returns:
        List of :class:`ReviewItem` design notes for subcircuit-specific concerns.
    """
    items: list[ReviewItem] = []
    detected_types: set[str] = set()

    # Collect declared subcircuit types from feature blocks
    for fb in requirements.features:
        detected_types.update(fb.subcircuits)

    # Infer subcircuit types from components
    relay_refs = _find_relay_refs(requirements)
    if relay_refs:
        detected_types.add("relay_driver")

    has_ldo = bool(_find_regulator_refs(requirements))
    if has_ldo:
        detected_types.add("ldo_regulator")

    has_adc = _has_adc_component(requirements)
    if has_adc:
        detected_types.add("voltage_divider_adc")

    # Check for USB-C connectors
    usb_c_refs = tuple(
        c.ref for c in requirements.components
        if "USB_C" in c.footprint.upper() or "USB-C" in c.value.upper()
    )
    if usb_c_refs:
        detected_types.add("usb_c_input")

    # Generate notes per subcircuit type
    if "relay_driver" in detected_types:
        items.append(ReviewItem(
            category="subcircuit",
            severity="recommended",
            title="Relay driver trace width",
            description=(
                "Relay coil traces should be >=0.5mm for coil current; "
                "flyback diode must be adjacent to relay coil pins; "
                "consider board slots between relay contacts and logic (>=10mm isolation)"
            ),
            affected_refs=relay_refs,
        ))

    if "ldo_regulator" in detected_types:
        reg_refs = _find_regulator_refs(requirements)
        items.append(ReviewItem(
            category="subcircuit",
            severity="recommended",
            title="LDO regulator layout",
            description=(
                "Add thermal vias under thermal pad; "
                "input/output caps must be within 5mm; "
                "verify dropout voltage vs input range"
            ),
            affected_refs=reg_refs,
        ))

    if "voltage_divider_adc" in detected_types:
        adc_refs = tuple(
            c.ref for c in requirements.components
            if any(
                p.function is not None and p.function.value in _ADC_PIN_FUNCTIONS
                for p in c.pins
            ) or any(kw in c.value.upper() for kw in _ADC_VALUE_KEYWORDS)
        )
        items.append(ReviewItem(
            category="subcircuit",
            severity="optional",
            title="ADC voltage divider routing",
            description=(
                "Keep traces short from divider output to ADC input; "
                "consider guard ring for high-impedance inputs"
            ),
            affected_refs=adc_refs,
        ))

    if "usb_c_input" in detected_types:
        items.append(ReviewItem(
            category="subcircuit",
            severity="recommended",
            title="USB-C layout notes",
            description=(
                "CC resistor tolerance must be 1%; "
                "add ESD protection on VBUS/D+/D-; "
                "maintain impedance control on D+/D- differential pair"
            ),
            affected_refs=usb_c_refs,
        ))

    return items


def _find_ic_decoupling_subgroups(
    fb_refs: set[str],
    comp_map: dict[str, object],
    ic_to_caps: dict[str, list[str]],
) -> list[ComponentGroup]:
    """Find IC + decoupling cap subgroups within a feature block."""
    subgroups: list[ComponentGroup] = []
    for ref in sorted(fb_refs):
        comp = comp_map.get(ref)
        if comp is None or not ref.startswith("U"):
            continue
        caps = ic_to_caps.get(ref, [])
        caps_in_feature = [c for c in caps if c in fb_refs]
        if caps_in_feature:
            subgroups.append(ComponentGroup(
                name=f"{ref} ({comp.value})",
                description=f"{comp.value} + decoupling",
                refs=(ref, *sorted(caps_in_feature)),
            ))
    return subgroups


def _find_relay_subgroups(
    fb_components: tuple[str, ...],
    ref_nets: dict[str, set[str]],
) -> list[ComponentGroup]:
    """Find relay driver subgroups (relay + flyback diode + driver)."""
    relay_refs = [r for r in fb_components if r.startswith("K")]
    if not relay_refs:
        return []
    relay_associated: set[str] = set(relay_refs)
    for relay_ref in relay_refs:
        relay_nets = ref_nets.get(relay_ref, set())
        for ref in fb_components:
            if ref in relay_associated:
                continue
            is_support_component = ref.startswith(("D", "Q"))
            shares_net = bool(ref_nets.get(ref, set()) & relay_nets)
            if is_support_component and shares_net:
                relay_associated.add(ref)
    if len(relay_associated) > len(relay_refs):
        return [ComponentGroup(
            name="Relay Driver Circuit",
            description="Relays with flyback diodes and drivers",
            refs=tuple(sorted(relay_associated)),
        )]
    return []


def _find_regulator_subgroups(
    fb_components: tuple[str, ...],
    comp_map: dict[str, object],
    ref_nets: dict[str, set[str]],
    cap_assigned: set[str],
) -> list[ComponentGroup]:
    """Find regulator + input/output passive subgroups."""
    subgroups: list[ComponentGroup] = []
    for ref in sorted(fb_components):
        comp = comp_map.get(ref)
        if comp is None or not ref.startswith("U"):
            continue
        upper_value = comp.value.upper()
        upper_desc = (comp.description or "").upper()
        is_regulator = any(
            kw in upper_value or kw in upper_desc
            for kw in _REGULATOR_KEYWORDS
        )
        if not is_regulator:
            continue
        reg_nets = ref_nets.get(ref, set())
        reg_power = {n for n in reg_nets if _is_power_net(n)}
        associated: list[str] = [ref]
        for cref in sorted(fb_components):
            if cref == ref or cref in cap_assigned:
                continue
            is_passive = cref.startswith(("C", "L"))
            shares_power_net = bool(ref_nets.get(cref, set()) & reg_power)
            if is_passive and shares_power_net:
                associated.append(cref)
        if len(associated) > 1:
            subgroups.append(ComponentGroup(
                name=f"{ref} Regulator ({comp.value})",
                description=f"{comp.value} with input/output passives",
                refs=tuple(associated),
            ))
    return subgroups


def _build_component_groups(
    requirements: ProjectRequirements,
) -> tuple[ComponentGroup, ...]:
    """Build component groups from feature blocks and netlist relationships.

    Each feature block becomes a top-level group. Within each group,
    components are further organized into subgroups:
    - ICs and their associated decoupling caps
    - Voltage regulators and their input/output passives
    - Connectors and their associated protection components

    Args:
        requirements: Project requirements with features, components, and nets.

    Returns:
        Tuple of :class:`ComponentGroup` describing the design's logical structure.
    """
    comp_map = {c.ref: c for c in requirements.components}
    groups: list[ComponentGroup] = []

    # Build ref->nets mapping for subgroup detection
    ref_nets: dict[str, set[str]] = {}
    for net in requirements.nets:
        for conn in net.connections:
            ref_nets.setdefault(conn.ref, set()).add(net.name)

    # Get decoupling pairs for subgroup assignment
    decoupling_pairs = _find_ic_decoupling_pairs(requirements)
    ic_to_caps: dict[str, list[str]] = {}
    cap_assigned: set[str] = set()
    for ic_ref, cap_ref in decoupling_pairs:
        ic_to_caps.setdefault(ic_ref, []).append(cap_ref)
        cap_assigned.add(cap_ref)

    for fb in requirements.features:
        fb_refs = set(fb.components)
        subgroups: list[ComponentGroup] = []

        subgroups.extend(
            _find_ic_decoupling_subgroups(fb_refs, comp_map, ic_to_caps)
        )
        subgroups.extend(
            _find_relay_subgroups(fb.components, ref_nets)
        )
        subgroups.extend(
            _find_regulator_subgroups(fb.components, comp_map, ref_nets, cap_assigned)
        )

        groups.append(ComponentGroup(
            name=fb.name,
            description=fb.description,
            refs=tuple(sorted(fb.components)),
            subgroups=tuple(subgroups),
        ))

    return tuple(groups)


def _build_board_summary(
    requirements: ProjectRequirements,
    pcb_design: PCBDesign | None,
) -> BoardSummary:
    """Build a board summary from requirements and optional PCB data."""
    has_wifi, _wifi_refs = _has_wifi_component(requirements)
    relay_refs = _find_relay_refs(requirements)
    power_nets = _find_power_nets(requirements)

    # Board size from mechanical constraints or PCB outline.
    if requirements.mechanical is not None:
        board_size = (
            requirements.mechanical.board_width_mm,
            requirements.mechanical.board_height_mm,
        )
    elif pcb_design is not None:
        xs = [p.x for p in pcb_design.outline.polygon]
        ys = [p.y for p in pcb_design.outline.polygon]
        board_size = (max(xs) - min(xs), max(ys) - min(ys))
    else:
        board_size = (0.0, 0.0)

    # Layer count from PCB or default.
    layer_count = 2
    if pcb_design is not None:
        layer_count = pcb_design.design_rules.layer_count

    # Unique nets.
    unique_nets = len(requirements.nets)
    if pcb_design is not None:
        unique_nets = len(pcb_design.nets)

    return BoardSummary(
        board_size_mm=board_size,
        component_count=len(requirements.components),
        layer_count=layer_count,
        unique_nets=unique_nets,
        power_nets=power_nets,
        has_wifi=has_wifi,
        has_relays=len(relay_refs) > 0,
        has_adc=_has_adc_component(requirements),
    )


def generate_design_review(
    requirements: ProjectRequirements,
    pcb_design: PCBDesign | None = None,
) -> DesignReview:
    """Analyze a design and generate actionable review items.

    Args:
        requirements: The project requirements describing components, nets, etc.
        pcb_design: Optional PCB layout for additional context.

    Returns:
        A DesignReview with board summary and categorized review items.
    """
    items: list[ReviewItem] = []

    # --- Connectivity validation (required — catch design errors early) ---
    items.extend(_check_connectivity(requirements))

    # --- Antenna edge clearance (manual verification needed) ---
    # NOTE: Antenna keepout zone is auto-generated by _make_antenna_keepout()
    # in pcb/builder.py, so we only remind about edge clearance verification.
    has_wifi, wifi_refs = _has_wifi_component(requirements)
    if has_wifi:
        items.append(ReviewItem(
            category="antenna",
            severity="required",
            title="Antenna edge clearance",
            description=(
                "Verify WiFi antenna extends past board edge or has clearance "
                "(keepout zone is auto-generated)"
            ),
            affected_refs=wifi_refs,
        ))

    # --- Relay isolation (manual — requires board cutouts) ---
    relay_refs = _find_relay_refs(requirements)
    if relay_refs:
        items.append(ReviewItem(
            category="relay",
            severity="required",
            title="Relay board slots",
            description=(
                "Add board slots between relay contacts and logic circuits"
            ),
            affected_refs=relay_refs,
        ))

    # NOTE: The following items are handled automatically by the framework:
    # - High-current trace widths → netclasses.py classify_nets() assigns
    #   wider traces to power nets automatically.
    # - Decoupling cap placement → constraints.py creates NEAR constraints
    #   placing caps within 3mm of their IC's VCC pins.
    # - Antenna keepout zone → builder.py _make_antenna_keepout() auto-creates.
    # - Relay trace width → netclasses.py applies wider power traces.
    # - Thermal relief → zones.py applies thermal relief to all zone connections.
    #
    # These are NOT listed as recommendations because they already happen.

    # --- Thermal vias for regulators (manual — add vias under thermal pad) ---
    regulator_refs = _find_regulator_refs(requirements)
    if regulator_refs:
        items.append(ReviewItem(
            category="thermal",
            severity="recommended",
            title="Regulator thermal vias",
            description=(
                "Add thermal vias under regulator thermal pad"
            ),
            affected_refs=regulator_refs,
        ))

    # --- Zone fill reminder (always — requires KiCad GUI action) ---
    items.append(ReviewItem(
        category="mechanical",
        severity="required",
        title="Zone fill",
        description=(
            "Run zone fill (Edit \u2192 Fill All Zones / press B) before final DRC"
        ),
        affected_refs=(),
    ))

    # --- Subcircuit-specific design notes ---
    items.extend(_subcircuit_design_notes(requirements))

    # --- Board context notes ---
    has_board_context = requirements.board_context is not None
    if has_board_context:
        ctx = requirements.board_context
        has_system_integration = bool(ctx.target_system)
        has_shared_grounds = bool(ctx.shared_grounds)
        if has_system_integration:
            items.append(ReviewItem(
                category="context",
                severity="required",
                title="System integration",
                description=(
                    f"Board connects to {ctx.target_system} — "
                    f"verify connector pinout matches harness"
                ),
                affected_refs=(),
            ))
        if has_shared_grounds:
            items.append(ReviewItem(
                category="context",
                severity="recommended",
                title="Shared ground return",
                description=(
                    "Sensors share ground return — consider star-ground "
                    "topology to minimize noise coupling"
                ),
                affected_refs=(),
            ))
        for note in ctx.notes:
            items.append(ReviewItem(
                category="context",
                severity="optional",
                title="Design note",
                description=note,
                affected_refs=(),
            ))

    summary = _build_board_summary(requirements, pcb_design)
    component_groups = _build_component_groups(requirements)

    return DesignReview(
        board_summary=summary,
        items=tuple(items),
        component_groups=component_groups,
    )


def _format_board_summary(s: BoardSummary) -> list[str]:
    """Format board summary section as Markdown lines."""
    lines = [
        "## Board Summary",
        f"- Size: {s.board_size_mm[0]}x{s.board_size_mm[1]}mm",
        f"- Components: {s.component_count}",
        f"- Nets: {s.unique_nets}",
        f"- Layers: {s.layer_count}",
    ]
    if s.power_nets:
        lines.append(f"- Power nets: {', '.join(s.power_nets)}")
    specials = [
        label
        for flag, label in (
            (s.has_wifi, "WiFi"), (s.has_relays, "Relays"), (s.has_adc, "ADC"),
        )
        if flag
    ]
    if specials:
        lines.append(f"- Special: {', '.join(specials)}")
    lines.append("")
    return lines


def _format_severity_section(
    heading: str,
    items: list[ReviewItem],
) -> list[str]:
    """Format a severity-level section as a Markdown checklist."""
    if not items:
        return []
    lines = [heading]
    for item in items:
        ref_str = f" (affects: {', '.join(item.affected_refs)})" if item.affected_refs else ""
        lines.append(f"- [ ] **{item.title}**: {item.description}{ref_str}")
    lines.append("")
    return lines


def format_design_review(
    review: DesignReview,
    project_name: str = "",
) -> str:
    """Format a DesignReview as a Markdown checklist.

    Args:
        review: The design review to format.
        project_name: Optional project name for the heading.

    Returns:
        A Markdown string with categorized checklists.
    """
    lines: list[str] = []
    heading = f"# Design Review: {project_name}" if project_name else "# Design Review"
    lines.append(heading)
    lines.append("")

    lines.extend(_format_board_summary(review.board_summary))

    # Component Groups
    if review.component_groups:
        lines.append("## Component Groups")
        for group in review.component_groups:
            lines.append(f"### {group.name}")
            if group.description:
                lines.append(f"_{group.description}_")
            lines.append(f"- Components: {', '.join(group.refs)}")
            for sub in group.subgroups:
                lines.append(f"  - **{sub.name}**: {', '.join(sub.refs)}")
                if sub.description:
                    lines.append(f"    _{sub.description}_")
            lines.append("")

    # Partition items by severity
    severity_sections = [
        ("## Required Actions", "required"),
        ("## Recommended", "recommended"),
        ("## Optional", "optional"),
    ]
    for section_heading, severity in severity_sections:
        items = [i for i in review.items if i.severity == severity]
        lines.extend(_format_severity_section(section_heading, items))

    return "\n".join(lines)

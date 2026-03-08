"""Design review module — generates actionable checklists from requirements and PCB data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicad_pipeline.models.pcb import PCBDesign
    from kicad_pipeline.models.requirements import ProjectRequirements

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

    # Build ref -> set of power nets, and net -> set of IC refs
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

    ic_refs = [c.ref for c in requirements.components if c.ref.startswith("U")]
    cap_refs = [c.ref for c in requirements.components if c.ref.startswith("C")]

    # Identify which ICs are regulators
    regulator_set: set[str] = set(_find_regulator_refs(requirements))

    # Build ref-to-feature mapping for feature-local matching
    ref_to_feature: dict[str, str] = {}
    for fb in requirements.features:
        for ref in fb.components:
            ref_to_feature[ref] = fb.name

    # Filter to caps that look like decoupling (both pins on power nets)
    decoupling_caps: list[str] = []
    for cref in cap_refs:
        comp = comp_map.get(cref)
        if comp is None or len(comp.pins) != 2:
            continue
        pin_nets = [p.net for p in comp.pins if p.net]
        if len(pin_nets) == 2 and all(_is_power_net(n) for n in pin_nets):
            decoupling_caps.append(cref)

    pairs: list[tuple[str, str]] = []
    seen_caps: set[str] = set()

    for cap in decoupling_caps:
        cap_comp = comp_map.get(cap)
        cap_nets = ref_power_nets.get(cap, set())
        cap_rails = {n for n in cap_nets if n.upper() not in _GND_NET_NAMES}
        desc = (cap_comp.description or "").upper() if cap_comp else ""
        cap_feature = ref_to_feature.get(cap, "")

        best_ic = ""
        best_score = 0.0

        # --- Priority 1: Description mentions a specific IC value ---
        for ic in ic_refs:
            ic_comp = comp_map.get(ic)
            if ic_comp is None:
                continue
            ic_value = ic_comp.value.upper()
            if ic_value and ic_value in desc:
                best_ic = ic
                best_score = 1000.0
                break

        # --- Priority 2: Cap description has regulator keywords ---
        # Pair with regulator in same feature block or sharing the cap's rail.
        if best_score < 1000.0 and any(kw in desc for kw in _REGULATOR_DESC_KEYWORDS):
            for ic in ic_refs:
                if ic not in regulator_set:
                    continue
                ic_nets = ref_power_nets.get(ic, set())
                shared = cap_rails & ic_nets
                if not shared:
                    continue
                # Prefer same-feature regulator
                ic_feature = ref_to_feature.get(ic, "")
                feature_bonus = 10.0 if (cap_feature and cap_feature == ic_feature) else 0.0
                score = 500.0 + feature_bonus
                if score > best_score:
                    best_score = score
                    best_ic = ic

        # --- Priority 3: Specific rail match (few IC users) ---
        if best_score < 500.0:
            for ic in ic_refs:
                ic_comp = comp_map.get(ic)
                if ic_comp is None:
                    continue
                ic_nets = ref_power_nets.get(ic, set())
                shared_rails = cap_rails & ic_nets
                if not shared_rails:
                    continue
                # Skip busy rails — too many ICs to determine which one owns the cap
                specific_rails = {
                    r for r in shared_rails
                    if net_ic_count.get(r, 0) <= _MAX_IC_COUNT_FOR_RAIL_MATCH
                }
                if not specific_rails:
                    continue
                # Score: prefer fewer IC users (more specific)
                score = sum(
                    1.0 / max(net_ic_count.get(r, 1), 1)
                    for r in specific_rails
                )
                # Bonus for same feature block
                ic_feature = ref_to_feature.get(ic, "")
                if cap_feature and cap_feature == ic_feature:
                    score += 5.0
                if score > best_score:
                    best_score = score
                    best_ic = ic

        if best_ic and cap not in seen_caps:
            pairs.append((best_ic, cap))
            seen_caps.add(cap)

    return pairs


def _has_adc_component(requirements: ProjectRequirements) -> bool:
    """Return True if any component has ADC-related pins or values."""
    for comp in requirements.components:
        for pin in comp.pins:
            if pin.function is not None and pin.function.value in ("adc", "analog_in"):
                return True
        upper_value = comp.value.upper()
        if "ADC" in upper_value or "ADS1" in upper_value or "MCP3" in upper_value:
            return True
    return False


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
        if connected_count == 0:
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
        elif connected_count < len(connectable_pins):
            # Some pins connected, some not — informational
            unconnected = [
                p.number for p in connectable_pins
                if (comp.ref, p.number) not in connected_pins
            ]
            # Only flag non-power/non-NC pins that are truly floating
            # (power pins often get connected via power symbols, not nets)
            signal_unconnected = [
                p_num for p_num in unconnected
                if not any(
                    p.number == p_num and p.pin_type.value in ("power_in", "power_out")
                    for p in connectable_pins
                )
            ]
            if signal_unconnected:
                pin_list = ", ".join(signal_unconnected[:5])
                extra = len(signal_unconnected) - 5
                suffix = f" +{extra} more" if extra > 0 else ""
                items.append(ReviewItem(
                    category="connectivity",
                    severity="recommended",
                    title="Partially unconnected component",
                    description=(
                        f"{comp.ref} ({comp.value}) has {len(signal_unconnected)} "
                        f"unconnected signal pins: {pin_list}{suffix}"
                    ),
                    affected_refs=(comp.ref,),
                ))

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
                p.function is not None and p.function.value in ("adc", "analog_in")
                for p in c.pins
            ) or "ADC" in c.value.upper() or "ADS1" in c.value.upper()
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

        # Find ICs in this feature and their decoupling caps
        for ref in sorted(fb.components):
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

        # Find relay groups (relay + flyback diode + driver)
        relay_refs = [r for r in fb.components if r.startswith("K")]
        if relay_refs:
            # Find diodes and transistors sharing nets with relays
            relay_associated: set[str] = set(relay_refs)
            for relay_ref in relay_refs:
                relay_nets = ref_nets.get(relay_ref, set())
                for ref in fb.components:
                    if ref in relay_associated:
                        continue
                    if ref.startswith(("D", "Q")) and ref_nets.get(ref, set()) & relay_nets:
                        relay_associated.add(ref)
            if len(relay_associated) > len(relay_refs):
                subgroups.append(ComponentGroup(
                    name="Relay Driver Circuit",
                    description="Relays with flyback diodes and drivers",
                    refs=tuple(sorted(relay_associated)),
                ))

        # Find regulator subgroups (regulator + input/output caps)
        for ref in sorted(fb.components):
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
            # Find caps sharing power nets with this regulator
            reg_nets = ref_nets.get(ref, set())
            reg_power = {n for n in reg_nets if _is_power_net(n)}
            associated: list[str] = [ref]
            for cref in sorted(fb.components):
                if cref == ref or cref in cap_assigned:
                    continue
                if not cref.startswith(("C", "L")):
                    continue
                c_nets = ref_nets.get(cref, set())
                if c_nets & reg_power:
                    associated.append(cref)
            if len(associated) > 1:
                subgroups.append(ComponentGroup(
                    name=f"{ref} Regulator ({comp.value})",
                    description=f"{comp.value} with input/output passives",
                    refs=tuple(associated),
                ))

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
    if requirements.board_context is not None:
        ctx = requirements.board_context
        if ctx.target_system:
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
        if ctx.shared_grounds:
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

    # --- Board Summary ---
    s = review.board_summary
    lines.append("## Board Summary")
    lines.append(f"- Size: {s.board_size_mm[0]}x{s.board_size_mm[1]}mm")
    lines.append(f"- Components: {s.component_count}")
    lines.append(f"- Nets: {s.unique_nets}")
    lines.append(f"- Layers: {s.layer_count}")
    if s.power_nets:
        lines.append(f"- Power nets: {', '.join(s.power_nets)}")
    specials: list[str] = []
    if s.has_wifi:
        specials.append("WiFi")
    if s.has_relays:
        specials.append("Relays")
    if s.has_adc:
        specials.append("ADC")
    if specials:
        lines.append(f"- Special: {', '.join(specials)}")
    lines.append("")

    # --- Component Groups ---
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

    # Partition items by severity.
    required = [i for i in review.items if i.severity == "required"]
    recommended = [i for i in review.items if i.severity == "recommended"]
    optional = [i for i in review.items if i.severity == "optional"]

    if required:
        lines.append("## Required Actions")
        for item in required:
            ref_str = f" (affects: {', '.join(item.affected_refs)})" if item.affected_refs else ""
            lines.append(f"- [ ] **{item.title}**: {item.description}{ref_str}")
        lines.append("")

    if recommended:
        lines.append("## Recommended")
        for item in recommended:
            ref_str = f" (affects: {', '.join(item.affected_refs)})" if item.affected_refs else ""
            lines.append(f"- [ ] **{item.title}**: {item.description}{ref_str}")
        lines.append("")

    if optional:
        lines.append("## Optional")
        for item in optional:
            ref_str = f" (affects: {', '.join(item.affected_refs)})" if item.affected_refs else ""
            lines.append(f"- [ ] **{item.title}**: {item.description}{ref_str}")
        lines.append("")

    return "\n".join(lines)

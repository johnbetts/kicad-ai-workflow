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


def _find_ic_decoupling_pairs(
    requirements: ProjectRequirements,
) -> list[tuple[str, str]]:
    """Find (IC_ref, cap_ref) pairs for decoupling cap placement checks.

    Only pairs caps that are likely dedicated decoupling for a specific IC:
    - Cap has exactly 2 pins, both on power nets (VCC+GND pattern)
    - Cap description mentions the IC (e.g. "ESP32 decoupling")
    - Or cap shares a non-GND power net with the IC and that net has
      few IC connections (specific rail, not a bus like GND)

    This avoids the cartesian explosion of pairing every cap with every IC.
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
        # Non-GND power nets on this cap (the specific rail it decouples)
        cap_rails = {n for n in cap_nets if n.upper() not in _GND_NET_NAMES}

        # Check if cap description mentions a specific IC
        desc = (cap_comp.description or "").upper() if cap_comp else ""
        best_ic = ""
        best_score = 0.0

        for ic in ic_refs:
            ic_comp = comp_map.get(ic)
            if ic_comp is None:
                continue

            # Description match: "ESP32 decoupling" → matches U3 (ESP32)
            ic_value = ic_comp.value.upper()
            if ic_value and ic_value in desc:
                best_ic = ic
                best_score = 100
                break

            # Rail match: shared non-GND power net with few IC users
            ic_nets = ref_power_nets.get(ic, set())
            shared_rails = cap_rails & ic_nets
            if not shared_rails:
                continue
            # Score: prefer rails shared by fewer ICs (more specific)
            score = sum(1.0 / max(net_ic_count.get(r, 1), 1) for r in shared_rails)
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

    # --- Antenna clearance ---
    has_wifi, wifi_refs = _has_wifi_component(requirements)
    if has_wifi:
        items.append(ReviewItem(
            category="antenna",
            severity="required",
            title="Antenna keepout zone",
            description=(
                "Create keepout zone around antenna "
                "(no copper/GND pour within 5mm)"
            ),
            affected_refs=wifi_refs,
        ))

    # --- Edge clearance for WiFi antenna ---
    if has_wifi:
        items.append(ReviewItem(
            category="antenna",
            severity="required",
            title="Antenna edge clearance",
            description=(
                "Verify WiFi antenna extends past board edge or has clearance"
            ),
            affected_refs=wifi_refs,
        ))

    # --- Relay isolation ---
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
        items.append(ReviewItem(
            category="relay",
            severity="required",
            title="Relay trace width",
            description=(
                "Use wider traces (\u22651mm) for relay contact paths"
            ),
            affected_refs=relay_refs,
        ))

    # --- High-current traces ---
    power_nets = _find_power_nets(requirements)
    high_current_nets = tuple(
        n for n in power_nets if n.upper() not in _GND_NET_NAMES
    )
    if high_current_nets:
        # Find component refs connected to these nets.
        affected: set[str] = set()
        for net in requirements.nets:
            if net.name in high_current_nets:
                for conn in net.connections:
                    affected.add(conn.ref)
        net_list = ", ".join(high_current_nets)
        items.append(ReviewItem(
            category="power",
            severity="recommended",
            title="High-current trace width",
            description=(
                f"Increase trace width for {net_list} to \u22650.5mm"
            ),
            affected_refs=tuple(sorted(affected)),
        ))

    # --- Thermal relief for regulators ---
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

    # --- Decoupling verification ---
    decoupling_pairs = _find_ic_decoupling_pairs(requirements)
    for ic_ref, cap_ref in decoupling_pairs:
        items.append(ReviewItem(
            category="power",
            severity="recommended",
            title="Decoupling cap placement",
            description=(
                f"Verify {cap_ref} is within 5mm of {ic_ref} VCC pin"
            ),
            affected_refs=(ic_ref, cap_ref),
        ))

    # --- Zone fill reminder (always) ---
    items.append(ReviewItem(
        category="mechanical",
        severity="required",
        title="Zone fill",
        description=(
            "Run zone fill (Edit \u2192 Fill All Zones / press B) before final DRC"
        ),
        affected_refs=(),
    ))

    summary = _build_board_summary(requirements, pcb_design)

    return DesignReview(
        board_summary=summary,
        items=tuple(items),
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

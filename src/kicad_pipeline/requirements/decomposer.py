"""Requirements decomposer: structured dict → validated ProjectRequirements.

This module does **not** perform natural-language parsing (that is an LLM
concern).  Instead it accepts structured Python dicts (e.g. parsed from JSON
produced by an LLM) and builds validated, immutable
:class:`~kicad_pipeline.models.requirements.ProjectRequirements` objects.

Typical usage::

    builder = RequirementsBuilder(ProjectInfo(name="my-board"))
    builder.add_component(Component(...))
    builder.add_net(Net(...))
    req = builder.build()

    # Persist / restore
    save_requirements(req, Path("requirements.json"))
    req2 = load_requirements(Path("requirements.json"))
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import ComponentError, RequirementsError

if TYPE_CHECKING:
    from pathlib import Path
from kicad_pipeline.models.requirements import (
    Component,
    FeatureBlock,
    MCUPinMap,
    MechanicalConstraints,
    Net,
    NetConnection,
    Pin,
    PinAssignment,
    PinFunction,
    PinType,
    PowerBudget,
    PowerRail,
    ProjectInfo,
    ProjectRequirements,
    Recommendation,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutable builder
# ---------------------------------------------------------------------------


class RequirementsBuilder:
    """Mutable builder for constructing a :class:`ProjectRequirements` step by step.

    All mutating methods perform lightweight duplicate-detection so that
    problems are surfaced early.  Call :meth:`build` to obtain the final,
    immutable :class:`~kicad_pipeline.models.requirements.ProjectRequirements`.

    Typical use::

        builder = RequirementsBuilder(ProjectInfo(name="my-board"))
        builder.add_component(Component(...))
        builder.add_net(Net(...))
        req = builder.build()
    """

    def __init__(self, project: ProjectInfo) -> None:
        """Initialise the builder with top-level project metadata.

        Args:
            project: Project info (name, author, revision, description).
        """
        self._project = project
        self._components: list[Component] = []
        self._nets: list[Net] = []
        self._features: list[FeatureBlock] = []
        self._recommendations: list[Recommendation] = []
        self._pin_map: MCUPinMap | None = None
        self._power_budget: PowerBudget | None = None
        self._mechanical: MechanicalConstraints | None = None

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_component(self, component: Component) -> None:
        """Add a component to the design.

        Args:
            component: The component to add.

        Raises:
            ComponentError: If a component with the same ``ref`` already exists.
        """
        existing_refs = {c.ref for c in self._components}
        if component.ref in existing_refs:
            raise ComponentError(
                f"Component ref {component.ref!r} already exists in design"
            )
        self._components.append(component)
        log.debug("Added component %s (%s)", component.ref, component.value)

    def add_net(self, net: Net) -> None:
        """Add a net to the design.

        Args:
            net: The net to add.

        Raises:
            RequirementsError: If a net with the same ``name`` already exists.
        """
        existing_names = {n.name for n in self._nets}
        if net.name in existing_names:
            raise RequirementsError(
                f"Net {net.name!r} already exists in design"
            )
        self._nets.append(net)
        log.debug("Added net %s (%d connections)", net.name, len(net.connections))

    def add_feature(self, feature: FeatureBlock) -> None:
        """Add a functional feature block.

        Args:
            feature: The feature block to add.
        """
        self._features.append(feature)
        log.debug("Added feature %s", feature.name)

    def add_recommendation(self, rec: Recommendation) -> None:
        """Append an AI recommendation.

        Args:
            rec: The recommendation to append.
        """
        self._recommendations.append(rec)
        log.debug("Added recommendation [%s] %s", rec.severity, rec.category)

    def set_pin_map(self, pin_map: MCUPinMap) -> None:
        """Set (or replace) the MCU pin map.

        Args:
            pin_map: The pin map to store.
        """
        self._pin_map = pin_map
        log.debug("Set pin map for MCU %s", pin_map.mcu_ref)

    def set_power_budget(self, budget: PowerBudget) -> None:
        """Set (or replace) the power budget.

        Args:
            budget: The power budget to store.
        """
        self._power_budget = budget
        log.debug(
            "Set power budget: %.1f mA total across %d rails",
            budget.total_current_ma,
            len(budget.rails),
        )

    def set_mechanical(self, mech: MechanicalConstraints) -> None:
        """Set (or replace) mechanical constraints.

        Args:
            mech: Mechanical constraints (board size, mounting holes, etc.)
        """
        self._mechanical = mech
        log.debug(
            "Set mechanical constraints: %.1f x %.1f mm",
            mech.board_width_mm,
            mech.board_height_mm,
        )

    # ------------------------------------------------------------------
    # Validation & build
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Run basic consistency validation.

        Checks:

        * At least one component is present.
        * No duplicate component ``ref`` values (defensive; duplicates are also
          blocked at insertion time).
        * No duplicate net ``name`` values.
        * All :class:`~kicad_pipeline.models.requirements.NetConnection` refs
          in every net correspond to a known component ``ref``.

        Returns:
            A list of error message strings.  An empty list means no errors.
        """
        errors: list[str] = []

        if not self._components:
            errors.append("Design has no components")

        # Duplicate refs (defensive)
        refs: list[str] = [c.ref for c in self._components]
        seen_refs: set[str] = set()
        for ref in refs:
            if ref in seen_refs:
                errors.append(f"Duplicate component ref: {ref!r}")
            seen_refs.add(ref)

        # Duplicate net names (defensive)
        net_names: list[str] = [n.name for n in self._nets]
        seen_nets: set[str] = set()
        for name in net_names:
            if name in seen_nets:
                errors.append(f"Duplicate net name: {name!r}")
            seen_nets.add(name)

        # Net connections reference existing component refs
        known_refs: set[str] = {c.ref for c in self._components}
        for net in self._nets:
            for conn in net.connections:
                if conn.ref not in known_refs:
                    errors.append(
                        f"Net {net.name!r} references unknown component ref {conn.ref!r}"
                    )

        return errors

    def build(self) -> ProjectRequirements:
        """Build and return the immutable :class:`ProjectRequirements`.

        Calls :meth:`validate` first; raises if there are any errors.

        Returns:
            A fully validated, frozen :class:`ProjectRequirements`.

        Raises:
            RequirementsError: If :meth:`validate` returns any error strings.
        """
        errors = self.validate()
        if errors:
            summary = "; ".join(errors)
            raise RequirementsError(
                f"Requirements validation failed ({len(errors)} error(s)): {summary}"
            )

        req = ProjectRequirements(
            project=self._project,
            features=tuple(self._features),
            components=tuple(self._components),
            nets=tuple(self._nets),
            pin_map=self._pin_map,
            power_budget=self._power_budget,
            mechanical=self._mechanical,
            recommendations=tuple(self._recommendations),
        )
        log.info(
            "Built ProjectRequirements: %d components, %d nets, %d features",
            len(req.components),
            len(req.nets),
            len(req.features),
        )
        return req


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def requirements_to_dict(req: ProjectRequirements) -> dict[str, object]:
    """Serialize :class:`ProjectRequirements` to a plain, JSON-serializable dict.

    The output structure is::

        {
          "project": {"name": ..., "author": ..., "revision": ..., "description": ...},
          "features": [...],
          "components": [{"ref": ..., "value": ..., "footprint": ..., "lcsc": ...,
                          "description": ..., "datasheet": ..., "pins": [...]}],
          "nets": [{"name": ..., "connections": [{"ref": ..., "pin": ...}]}],
          "pin_map": {...} | null,
          "power_budget": {...} | null,
          "mechanical": {...} | null,
          "recommendations": [...]
        }

    Args:
        req: The :class:`ProjectRequirements` to serialize.

    Returns:
        A plain dict suitable for passing to :func:`json.dumps`.
    """

    def _pin_to_dict(pin: Pin) -> dict[str, object]:
        return {
            "number": pin.number,
            "name": pin.name,
            "pin_type": pin.pin_type.value,
            "function": pin.function.value if pin.function is not None else None,
            "net": pin.net,
        }

    def _component_to_dict(comp: Component) -> dict[str, object]:
        return {
            "ref": comp.ref,
            "value": comp.value,
            "footprint": comp.footprint,
            "lcsc": comp.lcsc,
            "description": comp.description,
            "datasheet": comp.datasheet,
            "pins": [_pin_to_dict(p) for p in comp.pins],
        }

    def _net_to_dict(net: Net) -> dict[str, object]:
        return {
            "name": net.name,
            "connections": [
                {"ref": c.ref, "pin": c.pin} for c in net.connections
            ],
        }

    def _feature_to_dict(f: FeatureBlock) -> dict[str, object]:
        return {
            "name": f.name,
            "description": f.description,
            "components": list(f.components),
            "nets": list(f.nets),
            "subcircuits": list(f.subcircuits),
        }

    def _rec_to_dict(r: Recommendation) -> dict[str, object]:
        return {
            "severity": r.severity,
            "category": r.category,
            "message": r.message,
            "affected_refs": list(r.affected_refs),
        }

    def _pin_assignment_to_dict(pa: PinAssignment) -> dict[str, object]:
        return {
            "mcu_ref": pa.mcu_ref,
            "pin_number": pa.pin_number,
            "pin_name": pa.pin_name,
            "function": pa.function.value,
            "net": pa.net,
            "notes": pa.notes,
        }

    def _pin_map_to_dict(pm: MCUPinMap) -> dict[str, object]:
        return {
            "mcu_ref": pm.mcu_ref,
            "assignments": [_pin_assignment_to_dict(a) for a in pm.assignments],
            "unassigned_gpio": list(pm.unassigned_gpio),
        }

    def _rail_to_dict(r: PowerRail) -> dict[str, object]:
        return {
            "name": r.name,
            "voltage": r.voltage,
            "current_ma": r.current_ma,
            "source_ref": r.source_ref,
        }

    def _power_budget_to_dict(pb: PowerBudget) -> dict[str, object]:
        return {
            "rails": [_rail_to_dict(r) for r in pb.rails],
            "total_current_ma": pb.total_current_ma,
            "notes": list(pb.notes),
        }

    def _mechanical_to_dict(m: MechanicalConstraints) -> dict[str, object]:
        return {
            "board_width_mm": m.board_width_mm,
            "board_height_mm": m.board_height_mm,
            "enclosure": m.enclosure,
            "mounting_hole_diameter_mm": m.mounting_hole_diameter_mm,
            "mounting_hole_positions": [list(pos) for pos in m.mounting_hole_positions],
            "notes": m.notes,
        }

    return {
        "project": {
            "name": req.project.name,
            "author": req.project.author,
            "revision": req.project.revision,
            "description": req.project.description,
        },
        "features": [_feature_to_dict(f) for f in req.features],
        "components": [_component_to_dict(c) for c in req.components],
        "nets": [_net_to_dict(n) for n in req.nets],
        "pin_map": _pin_map_to_dict(req.pin_map) if req.pin_map is not None else None,
        "power_budget": (
            _power_budget_to_dict(req.power_budget)
            if req.power_budget is not None
            else None
        ),
        "mechanical": (
            _mechanical_to_dict(req.mechanical) if req.mechanical is not None else None
        ),
        "recommendations": [_rec_to_dict(r) for r in req.recommendations],
    }


def requirements_from_dict(data: dict[str, object]) -> ProjectRequirements:
    """Deserialize :class:`ProjectRequirements` from a plain dict.

    Args:
        data: A dict with the same structure produced by
            :func:`requirements_to_dict`.

    Returns:
        A validated :class:`ProjectRequirements` instance.

    Raises:
        RequirementsError: If *data* is malformed or fails validation.
    """
    try:
        return _parse_requirements(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise RequirementsError(
            f"Failed to deserialize ProjectRequirements: {exc}"
        ) from exc


def _parse_requirements(data: dict[str, object]) -> ProjectRequirements:
    """Internal parser; propagates raw exceptions for wrapping by the caller."""

    # --- project ---
    proj_raw = _as_dict(data["project"])
    project = ProjectInfo(
        name=str(proj_raw["name"]),
        author=_optional_str(proj_raw.get("author")),
        revision=str(proj_raw.get("revision", "v0.1")),
        description=_optional_str(proj_raw.get("description")),
    )

    # --- features ---
    features: list[FeatureBlock] = []
    for f_raw in _as_list(data.get("features", [])):
        fd = _as_dict(f_raw)
        features.append(
            FeatureBlock(
                name=str(fd["name"]),
                description=str(fd["description"]),
                components=tuple(str(x) for x in _as_list(fd.get("components", []))),
                nets=tuple(str(x) for x in _as_list(fd.get("nets", []))),
                subcircuits=tuple(
                    str(x) for x in _as_list(fd.get("subcircuits", []))
                ),
            )
        )

    # --- components ---
    components: list[Component] = []
    for c_raw in _as_list(data.get("components", [])):
        cd = _as_dict(c_raw)
        pins: list[Pin] = []
        for p_raw in _as_list(cd.get("pins", [])):
            pd = _as_dict(p_raw)
            func_val = pd.get("function")
            pins.append(
                Pin(
                    number=str(pd["number"]),
                    name=str(pd["name"]),
                    pin_type=PinType(pd["pin_type"]),
                    function=PinFunction(str(func_val)) if func_val is not None else None,
                    net=_optional_str(pd.get("net")),
                )
            )
        components.append(
            Component(
                ref=str(cd["ref"]),
                value=str(cd["value"]),
                footprint=str(cd["footprint"]),
                lcsc=_optional_str(cd.get("lcsc")),
                description=_optional_str(cd.get("description")),
                datasheet=_optional_str(cd.get("datasheet")),
                pins=tuple(pins),
            )
        )

    # --- nets ---
    nets: list[Net] = []
    for n_raw in _as_list(data.get("nets", [])):
        nd = _as_dict(n_raw)
        connections: list[NetConnection] = []
        for conn_raw in _as_list(nd.get("connections", [])):
            cd2 = _as_dict(conn_raw)
            connections.append(
                NetConnection(ref=str(cd2["ref"]), pin=str(cd2["pin"]))
            )
        nets.append(Net(name=str(nd["name"]), connections=tuple(connections)))

    # --- pin_map ---
    pin_map: MCUPinMap | None = None
    pm_raw = data.get("pin_map")
    if pm_raw is not None:
        pm_dict = _as_dict(pm_raw)
        assignments: list[PinAssignment] = []
        for a_raw in _as_list(pm_dict.get("assignments", [])):
            ad = _as_dict(a_raw)
            assignments.append(
                PinAssignment(
                    mcu_ref=str(ad["mcu_ref"]),
                    pin_number=str(ad["pin_number"]),
                    pin_name=str(ad["pin_name"]),
                    function=PinFunction(str(ad["function"])),
                    net=str(ad["net"]),
                    notes=_optional_str(ad.get("notes")),
                )
            )
        pin_map = MCUPinMap(
            mcu_ref=str(pm_dict["mcu_ref"]),
            assignments=tuple(assignments),
            unassigned_gpio=tuple(
                str(x) for x in _as_list(pm_dict.get("unassigned_gpio", []))
            ),
        )

    # --- power_budget ---
    power_budget: PowerBudget | None = None
    pb_raw = data.get("power_budget")
    if pb_raw is not None:
        pb_dict = _as_dict(pb_raw)
        rails: list[PowerRail] = []
        for r_raw in _as_list(pb_dict.get("rails", [])):
            rd = _as_dict(r_raw)
            rails.append(
                PowerRail(
                    name=str(rd["name"]),
                    voltage=float(rd["voltage"]),  # type: ignore[arg-type]
                    current_ma=float(rd["current_ma"]),  # type: ignore[arg-type]
                    source_ref=str(rd["source_ref"]),
                )
            )
        power_budget = PowerBudget(
            rails=tuple(rails),
            total_current_ma=float(pb_dict["total_current_ma"]),  # type: ignore[arg-type]
            notes=tuple(str(x) for x in _as_list(pb_dict.get("notes", []))),
        )

    # --- mechanical ---
    mechanical: MechanicalConstraints | None = None
    mech_raw = data.get("mechanical")
    if mech_raw is not None:
        md = _as_dict(mech_raw)
        hole_positions: list[tuple[float, float]] = []
        for pos_raw in _as_list(md.get("mounting_hole_positions", [])):
            pos = _as_list(pos_raw)
            hole_positions.append((float(pos[0]), float(pos[1])))  # type: ignore[arg-type]
        mechanical = MechanicalConstraints(
            board_width_mm=float(md["board_width_mm"]),  # type: ignore[arg-type]
            board_height_mm=float(md["board_height_mm"]),  # type: ignore[arg-type]
            enclosure=_optional_str(md.get("enclosure")),
            mounting_hole_diameter_mm=float(
                md.get("mounting_hole_diameter_mm", 3.2)  # type: ignore[arg-type]
            ),
            mounting_hole_positions=tuple(hole_positions),
            notes=_optional_str(md.get("notes")),
        )

    # --- recommendations ---
    recommendations: list[Recommendation] = []
    for r_raw in _as_list(data.get("recommendations", [])):
        rd2 = _as_dict(r_raw)
        recommendations.append(
            Recommendation(
                severity=str(rd2["severity"]),
                category=str(rd2["category"]),
                message=str(rd2["message"]),
                affected_refs=tuple(
                    str(x) for x in _as_list(rd2.get("affected_refs", []))
                ),
            )
        )

    # Use the builder to get validation for free
    builder = RequirementsBuilder(project)
    for comp in components:
        builder.add_component(comp)
    for net in nets:
        builder.add_net(net)
    for feat in features:
        builder.add_feature(feat)
    for rec in recommendations:
        builder.add_recommendation(rec)
    if pin_map is not None:
        builder.set_pin_map(pin_map)
    if power_budget is not None:
        builder.set_power_budget(power_budget)
    if mechanical is not None:
        builder.set_mechanical(mechanical)

    return builder.build()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_requirements(req: ProjectRequirements, path: Path) -> None:
    """Serialize *req* and write it to a JSON file at *path*.

    Args:
        req: The requirements to persist.
        path: Destination file path.  Parent directories must exist.
    """
    data = requirements_to_dict(req)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info("Saved requirements to %s", path)


def load_requirements(path: Path) -> ProjectRequirements:
    """Load and deserialize a :class:`ProjectRequirements` from a JSON file.

    Args:
        path: Path to a JSON file previously written by :func:`save_requirements`.

    Returns:
        The deserialized :class:`ProjectRequirements`.

    Raises:
        RequirementsError: If the file is missing, not valid JSON, or fails
            requirements validation.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RequirementsError(f"Cannot read requirements file {path}: {exc}") from exc

    try:
        data: dict[str, object] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RequirementsError(
            f"Requirements file {path} is not valid JSON: {exc}"
        ) from exc

    log.info("Loading requirements from %s", path)
    return requirements_from_dict(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_dict(value: object) -> dict[str, object]:
    """Assert *value* is a dict and return it with a typed annotation."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict, got {type(value).__name__!r}")
    return value  # mypy narrows to dict[str,object] after isinstance guard


def _as_list(value: object) -> list[object]:
    """Assert *value* is a list and return it."""
    if not isinstance(value, list):
        raise TypeError(f"Expected list, got {type(value).__name__!r}")
    return value


def _optional_str(value: object) -> str | None:
    """Return *value* as str, or None if it is None."""
    if value is None:
        return None
    return str(value)

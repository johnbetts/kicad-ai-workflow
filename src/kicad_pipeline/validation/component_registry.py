"""Component registry for tracking verification status and known issues.

Loads a JSON-based registry of component specifications, tracks verification
status, known issues, and verified fixes.  Provides query and mutation
methods that rebuild frozen dataclass instances on every change.

Supports dual-catalog loading: ``data/footprint_catalog.json`` (physical
specs shared across parts) and ``data/parts_catalog.json`` (per-part
identity data).  Falls back to the legacy ``data/component_registry.json``
when catalogs are absent.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

from kicad_pipeline.exceptions import ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ComponentRegistryError(ValidationError):
    """Raised when the component registry cannot be loaded or is malformed."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownIssue:
    """A tracked issue for a registered component."""

    issue_id: str
    description: str
    found_date: str
    status: str  # "open", "fixed"


@dataclass(frozen=True)
class VerifiedFix:
    """A fix that has been verified against a specific commit."""

    issue_id: str
    fix_description: str
    fix_commit: str


@dataclass(frozen=True)
class PinSpec:
    """A single pin on a component."""

    number: str
    name: str
    pin_type: str  # "input", "output", "power_in", "passive", etc.


@dataclass(frozen=True)
class FootprintSpec:
    """Physical specification for a footprint type (shared across all parts using it)."""

    footprint_id: str
    description: str
    expected_pads: int
    expected_pad_type: str
    body_width_mm: float | None
    body_height_mm: float | None
    model_rotation_z: float
    model_offset_xy_max_mm: float
    kicad_ref_pad1_x: float
    kicad_ref_pad1_y: float


@dataclass(frozen=True)
class PartSpec:
    """Specification for a distinct JLCPCB/manufacturer part."""

    part_id: str
    lcsc: str | None
    mpn: str | None
    manufacturer: str | None
    value: str
    footprint_id: str  # FK -> FootprintSpec
    description: str
    datasheet_url: str | None
    ref_prefix: str
    pins: tuple[PinSpec, ...]
    verification_status: str
    last_verified_commit: str | None
    known_issues: tuple[KnownIssue, ...]
    verified_fixes: tuple[VerifiedFix, ...]


@dataclass(frozen=True)
class ComponentSpec:
    """Full specification for a registered component."""

    component_id: str
    ref: str
    value: str
    footprint_id: str
    description: str
    expected_pads: int
    expected_pad_type: str  # "smd" or "thru_hole"
    body_width_mm: float | None
    body_height_mm: float | None
    model_rotation_z: float
    model_offset_xy_max_mm: float
    # KiCad library pad 1 position — the STEP model origin aligns with
    # this position.  Used to compute the 3D model offset when the actual
    # footprint (e.g. JLCPCB) has pad 1 at a different location.
    kicad_ref_pad1_x: float
    kicad_ref_pad1_y: float
    lcsc: str | None
    datasheet_url: str | None
    verification_status: str  # "unverified", "verified", "failed"
    last_verified_commit: str | None
    known_issues: tuple[KnownIssue, ...]
    verified_fixes: tuple[VerifiedFix, ...]
    pins: tuple[PinSpec, ...]

    @classmethod
    def from_footprint_and_part(cls, fp: FootprintSpec, part: PartSpec) -> ComponentSpec:
        """Merge footprint physical data with part identity data."""
        # Parts with explicit pins may include a thermal/exposed pad that
        # the base footprint doesn't count.  Use the higher of the two.
        pads = fp.expected_pads
        if part.pins and len(part.pins) > pads:
            pads = len(part.pins)
        return cls(
            component_id=part.part_id,
            ref=f"{part.ref_prefix}1",
            value=part.value,
            footprint_id=fp.footprint_id,
            description=part.description or fp.description,
            expected_pads=pads,
            expected_pad_type=fp.expected_pad_type,
            body_width_mm=fp.body_width_mm,
            body_height_mm=fp.body_height_mm,
            model_rotation_z=fp.model_rotation_z,
            model_offset_xy_max_mm=fp.model_offset_xy_max_mm,
            kicad_ref_pad1_x=fp.kicad_ref_pad1_x,
            kicad_ref_pad1_y=fp.kicad_ref_pad1_y,
            lcsc=part.lcsc,
            datasheet_url=part.datasheet_url,
            verification_status=part.verification_status,
            last_verified_commit=part.last_verified_commit,
            known_issues=part.known_issues,
            verified_fixes=part.verified_fixes,
            pins=part.pins,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ComponentRegistry:
    """Load and query the component registry.

    Supports dual-catalog loading:

    1. ``data/footprint_catalog.json`` — physical footprint specs
    2. ``data/parts_catalog.json`` — per-part identity specs (references footprints)
    3. ``data/component_registry.json`` — legacy flat registry (fallback)

    When catalogs are present, parts are merged with their footprint to produce
    :class:`ComponentSpec` entries.  The legacy file is loaded afterward and only
    fills in IDs not already covered by the catalogs.
    """

    _DEFAULT_PATH: Path = (
        Path(__file__).resolve().parents[3] / "data" / "component_registry.json"
    )
    _DATA_DIR: Path = Path(__file__).resolve().parents[3] / "data"

    def __init__(self, path: Path | None = None) -> None:
        self._path: Path = path or self._DEFAULT_PATH
        self._specs: dict[str, ComponentSpec] = {}
        self._footprints: dict[str, FootprintSpec] = {}
        self._parts: dict[str, PartSpec] = {}
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        """Load registry, preferring dual catalogs with legacy fallback."""
        fp_path = self._DATA_DIR / "footprint_catalog.json"
        parts_path = self._DATA_DIR / "parts_catalog.json"

        catalogs_loaded = False

        # Step 1 & 2: Try loading footprint + parts catalogs
        if fp_path.exists() and parts_path.exists():
            try:
                self._load_footprint_catalog(fp_path)
                self._load_parts_catalog(parts_path)
                catalogs_loaded = True
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                logger.warning(
                    "Failed to load dual catalogs, falling back to legacy: %s", exc
                )
                self._footprints.clear()
                self._parts.clear()

        # Step 3: Merge each part with its footprint into self._specs
        if catalogs_loaded:
            self._merge_catalogs()

        # Step 4 & 5: Load legacy registry — only add entries not already covered
        self._load_legacy(skip_existing=catalogs_loaded)

    def _load_footprint_catalog(self, path: Path) -> None:
        """Load footprint_catalog.json into self._footprints."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries: list[dict[str, object]]
        if isinstance(raw, dict) and "footprints" in raw:
            entries = list(raw["footprints"].values())
        elif isinstance(raw, list):
            entries = raw
        else:
            raise ComponentRegistryError(
                f"Unexpected footprint catalog structure: {type(raw).__name__}"
            )
        for entry in entries:
            fp = self._parse_footprint_spec(entry)
            self._footprints[fp.footprint_id] = fp
        logger.info("Loaded %d footprints from %s", len(self._footprints), path)

    def _load_parts_catalog(self, path: Path) -> None:
        """Load parts_catalog.json into self._parts."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries: list[dict[str, object]]
        if isinstance(raw, dict) and "parts" in raw:
            entries = list(raw["parts"].values())
        elif isinstance(raw, list):
            entries = raw
        else:
            raise ComponentRegistryError(
                f"Unexpected parts catalog structure: {type(raw).__name__}"
            )
        for entry in entries:
            part = self._parse_part_spec(entry)
            self._parts[part.part_id] = part
        logger.info("Loaded %d parts from %s", len(self._parts), path)

    def _merge_catalogs(self) -> None:
        """Merge parts with their footprints into ComponentSpec entries.

        Also creates generic footprint-only entries so that lookups like
        ``get("R_0805")`` still work for footprint-only queries.
        """
        # Part-based merged entries
        for part in self._parts.values():
            fp = self._footprints.get(part.footprint_id)
            if fp is None:
                logger.warning(
                    "Part %s references unknown footprint %s — skipped",
                    part.part_id,
                    part.footprint_id,
                )
                continue
            self._specs[part.part_id] = ComponentSpec.from_footprint_and_part(fp, part)

        # Generic footprint-only entries (don't overwrite part-based entries)
        for fp in self._footprints.values():
            if fp.footprint_id not in self._specs:
                self._specs[fp.footprint_id] = ComponentSpec(
                    component_id=fp.footprint_id,
                    ref="X1",
                    value=fp.footprint_id,
                    footprint_id=fp.footprint_id,
                    description=fp.description,
                    expected_pads=fp.expected_pads,
                    expected_pad_type=fp.expected_pad_type,
                    body_width_mm=fp.body_width_mm,
                    body_height_mm=fp.body_height_mm,
                    model_rotation_z=fp.model_rotation_z,
                    model_offset_xy_max_mm=fp.model_offset_xy_max_mm,
                    kicad_ref_pad1_x=fp.kicad_ref_pad1_x,
                    kicad_ref_pad1_y=fp.kicad_ref_pad1_y,
                    lcsc=None,
                    datasheet_url=None,
                    verification_status="unverified",
                    last_verified_commit=None,
                    known_issues=(),
                    verified_fixes=(),
                    pins=(),
                )

    def _load_legacy(self, *, skip_existing: bool) -> None:
        """Load legacy component_registry.json.

        When *skip_existing* is True, entries whose ``component_id`` is already
        present in ``self._specs`` (populated from catalogs) are skipped.
        """
        if not self._path.exists():
            if not skip_existing:
                logger.warning(
                    "Component registry not found at %s — starting empty", self._path
                )
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ComponentRegistryError(
                f"Failed to load component registry from {self._path}: {exc}"
            ) from exc

        # Support both formats:
        # - {"schema_version": N, "components": {...}} (keyed by component_id)
        # - [...] (flat array of component dicts)
        if isinstance(raw, dict) and "components" in raw:
            entries = raw["components"].values()
        elif isinstance(raw, list):
            entries = raw
        else:
            raise ComponentRegistryError(
                f"Unexpected JSON structure: expected dict with 'components' or array, "
                f"got {type(raw).__name__}"
            )

        added = 0
        for entry in entries:
            spec = self._parse_spec(entry)
            if skip_existing and spec.component_id in self._specs:
                continue
            self._specs[spec.component_id] = spec
            added += 1

        logger.info("Loaded %d components from %s", added, self._path)

    def save(self) -> None:
        """Write current state back to JSON files.

        When dual catalogs have been loaded, writes ``footprint_catalog.json``
        and ``parts_catalog.json``.  Always writes the legacy
        ``component_registry.json`` for backward compatibility.
        """
        # Write dual catalogs if we have catalog data
        if self._footprints:
            self._save_footprint_catalog()
        if self._parts:
            self._save_parts_catalog()

        # Always write legacy format
        components = {
            s.component_id: self._spec_to_dict(s) for s in self.all_components()
        }
        data = {"schema_version": 1, "components": components}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved %d components to %s", len(components), self._path)

    def _save_footprint_catalog(self) -> None:
        """Write footprint_catalog.json."""
        fp_path = self._DATA_DIR / "footprint_catalog.json"
        footprints = {
            fp_id: dataclasses.asdict(fp)
            for fp_id, fp in sorted(self._footprints.items())
        }
        data = {"schema_version": 1, "footprints": footprints}
        fp_path.parent.mkdir(parents=True, exist_ok=True)
        fp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved %d footprints to %s", len(footprints), fp_path)

    def _save_parts_catalog(self) -> None:
        """Write parts_catalog.json."""
        parts_path = self._DATA_DIR / "parts_catalog.json"
        parts: dict[str, dict[str, object]] = {}
        for part_id, part in sorted(self._parts.items()):
            d = dataclasses.asdict(part)
            # Convert nested tuples (serialized as lists by asdict) — no change needed
            parts[part_id] = d
        data = {"schema_version": 1, "parts": parts}
        parts_path.parent.mkdir(parents=True, exist_ok=True)
        parts_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved %d parts to %s", len(parts), parts_path)

    # -- queries ------------------------------------------------------------

    def all_components(self) -> list[ComponentSpec]:
        """Return all registered components, sorted by component_id."""
        return sorted(self._specs.values(), key=lambda s: s.component_id)

    def get(self, component_id: str) -> ComponentSpec | None:
        """Lookup a single component by ID."""
        return self._specs.get(component_id)

    def get_footprint(self, footprint_id: str) -> FootprintSpec | None:
        """Lookup a footprint specification by ID."""
        return self._footprints.get(footprint_id)

    def get_part(self, part_id: str) -> PartSpec | None:
        """Lookup a part specification by ID."""
        return self._parts.get(part_id)

    def get_part_by_lcsc(self, lcsc: str) -> PartSpec | None:
        """Lookup a part specification by LCSC part number."""
        for part in self._parts.values():
            if part.lcsc == lcsc:
                return part
        return None

    def all_footprints(self) -> list[FootprintSpec]:
        """Return all registered footprints, sorted by footprint_id."""
        return sorted(self._footprints.values(), key=lambda f: f.footprint_id)

    def all_parts(self) -> list[PartSpec]:
        """Return all registered parts, sorted by part_id."""
        return sorted(self._parts.values(), key=lambda p: p.part_id)

    def open_issues(self) -> list[tuple[str, KnownIssue]]:
        """Return all (component_id, issue) pairs with status='open'."""
        results: list[tuple[str, KnownIssue]] = []
        for cid, spec in sorted(self._specs.items()):
            for issue in spec.known_issues:
                if issue.status == "open":
                    results.append((cid, issue))
        return results

    # -- mutations ----------------------------------------------------------

    def update_status(
        self, component_id: str, status: str, commit: str | None = None
    ) -> None:
        """Update verification status for a component.

        Replaces the internal :class:`ComponentSpec` with a new frozen instance.
        """
        if component_id not in self._specs:
            raise ComponentRegistryError(
                f"Component '{component_id}' not found in registry"
            )
        kwargs: dict[str, object] = {"verification_status": status}
        if commit is not None:
            kwargs["last_verified_commit"] = commit
        self._replace_spec(component_id, **kwargs)

    def record_issue(self, component_id: str, issue: KnownIssue) -> None:
        """Add a known issue to a component."""
        if component_id not in self._specs:
            raise ComponentRegistryError(
                f"Component '{component_id}' not found in registry"
            )
        old = self._specs[component_id]
        self._replace_spec(
            component_id,
            known_issues=(*old.known_issues, issue),
        )

    def update_model_offset(self, component_id: str, x: float, y: float) -> None:
        """Update the 3D model offset reference pad-1 position for a component.

        Args:
            component_id: Registry component ID to update.
            x: New kicad_ref_pad1_x value.
            y: New kicad_ref_pad1_y value.
        """
        if component_id not in self._specs:
            raise ComponentRegistryError(
                f"Component '{component_id}' not found in registry"
            )
        self._replace_spec(component_id, kicad_ref_pad1_x=x, kicad_ref_pad1_y=y)

    def record_fix(self, component_id: str, fix: VerifiedFix) -> None:
        """Record a verified fix for a component."""
        if component_id not in self._specs:
            raise ComponentRegistryError(
                f"Component '{component_id}' not found in registry"
            )
        old = self._specs[component_id]
        self._replace_spec(
            component_id,
            verified_fixes=(*old.verified_fixes, fix),
        )

    # -- internal helpers ---------------------------------------------------

    def _replace_spec(self, component_id: str, **kwargs: object) -> None:
        """Rebuild a frozen ComponentSpec with updated fields.

        Also propagates footprint-level fields (``kicad_ref_pad1_x/y``,
        ``model_rotation_z``, ``model_offset_xy_max_mm``) back to the
        matching :class:`FootprintSpec` so that ``save()`` persists them
        to the footprint catalog.
        """
        old = self._specs[component_id]
        field_dict: dict[str, object] = {
            f.name: getattr(old, f.name) for f in fields(old)
        }
        field_dict.update(kwargs)
        new_spec = ComponentSpec(**field_dict)  # type: ignore[arg-type]
        self._specs[component_id] = new_spec

        # Propagate footprint-level changes back to _footprints
        fp_fields = ("kicad_ref_pad1_x", "kicad_ref_pad1_y",
                     "model_rotation_z", "model_offset_xy_max_mm")
        fp_changes = {k: v for k, v in kwargs.items() if k in fp_fields}
        if fp_changes and new_spec.footprint_id in self._footprints:
            old_fp = self._footprints[new_spec.footprint_id]
            fp_dict: dict[str, object] = {
                f.name: getattr(old_fp, f.name) for f in fields(old_fp)
            }
            fp_dict.update(fp_changes)
            self._footprints[new_spec.footprint_id] = FootprintSpec(**fp_dict)  # type: ignore[arg-type]

    @staticmethod
    def _parse_footprint_spec(entry: dict[str, object]) -> FootprintSpec:
        """Parse a single JSON object into a :class:`FootprintSpec`."""
        body_w = entry.get("body_width_mm")
        body_h = entry.get("body_height_mm")
        return FootprintSpec(
            footprint_id=str(entry.get("footprint_id", "")),
            description=str(entry.get("description", "")),
            expected_pads=int(entry.get("expected_pads", 0)),  # type: ignore[arg-type]
            expected_pad_type=str(entry.get("expected_pad_type", "smd")),
            body_width_mm=float(body_w) if body_w is not None else None,
            body_height_mm=float(body_h) if body_h is not None else None,
            model_rotation_z=float(entry.get("model_rotation_z", 0.0)),  # type: ignore[arg-type]
            model_offset_xy_max_mm=float(entry.get("model_offset_xy_max_mm", 0.0)),  # type: ignore[arg-type]
            kicad_ref_pad1_x=float(entry.get("kicad_ref_pad1_x", 0.0)),  # type: ignore[arg-type]
            kicad_ref_pad1_y=float(entry.get("kicad_ref_pad1_y", 0.0)),  # type: ignore[arg-type]
        )

    @staticmethod
    def _parse_part_spec(entry: dict[str, object]) -> PartSpec:
        """Parse a single JSON object into a :class:`PartSpec`."""
        known_issues = tuple(
            KnownIssue(
                issue_id=str(ki.get("issue_id", "")),
                description=str(ki.get("description", "")),
                found_date=str(ki.get("found_date", "")),
                status=str(ki.get("status", "open")),
            )
            for ki in (entry.get("known_issues") or [])  # type: ignore[union-attr]
        )
        verified_fixes = tuple(
            VerifiedFix(
                issue_id=str(vf.get("issue_id", "")),
                fix_description=str(vf.get("fix_description", "")),
                fix_commit=str(vf.get("fix_commit", "")),
            )
            for vf in (entry.get("verified_fixes") or [])  # type: ignore[union-attr]
        )
        pins = tuple(
            PinSpec(
                number=str(p.get("number", "")),
                name=str(p.get("name", "")),
                pin_type=str(p.get("pin_type", "passive")),
            )
            for p in (entry.get("pins") or [])  # type: ignore[union-attr]
        )
        last_commit = entry.get("last_verified_commit")
        lcsc_raw = entry.get("lcsc")
        mpn_raw = entry.get("mpn")
        mfr_raw = entry.get("manufacturer")
        datasheet_raw = entry.get("datasheet_url")
        return PartSpec(
            part_id=str(entry.get("part_id", "")),
            lcsc=str(lcsc_raw) if lcsc_raw is not None else None,
            mpn=str(mpn_raw) if mpn_raw is not None else None,
            manufacturer=str(mfr_raw) if mfr_raw is not None else None,
            value=str(entry.get("value", "")),
            footprint_id=str(entry.get("footprint_id", "")),
            description=str(entry.get("description", "")),
            datasheet_url=str(datasheet_raw) if datasheet_raw is not None else None,
            ref_prefix=str(entry.get("ref_prefix", "X")),
            pins=pins,
            verification_status=str(entry.get("verification_status", "unverified")),
            last_verified_commit=str(last_commit) if last_commit is not None else None,
            known_issues=known_issues,
            verified_fixes=verified_fixes,
        )

    @staticmethod
    def _parse_spec(entry: dict[str, object]) -> ComponentSpec:
        """Parse a single JSON object into a :class:`ComponentSpec`."""
        known_issues = tuple(
            KnownIssue(
                issue_id=str(ki.get("issue_id", "")),
                description=str(ki.get("description", "")),
                found_date=str(ki.get("found_date", "")),
                status=str(ki.get("status", "open")),
            )
            for ki in (entry.get("known_issues") or [])  # type: ignore[union-attr]
        )

        verified_fixes = tuple(
            VerifiedFix(
                issue_id=str(vf.get("issue_id", "")),
                fix_description=str(vf.get("fix_description", "")),
                fix_commit=str(vf.get("fix_commit", "")),
            )
            for vf in (entry.get("verified_fixes") or [])  # type: ignore[union-attr]
        )

        pins = tuple(
            PinSpec(
                number=str(p.get("number", "")),
                name=str(p.get("name", "")),
                pin_type=str(p.get("pin_type", "passive")),
            )
            for p in (entry.get("pins") or [])  # type: ignore[union-attr]
        )

        body_w = entry.get("body_width_mm")
        body_h = entry.get("body_height_mm")
        last_commit = entry.get("last_verified_commit")
        lcsc_raw = entry.get("lcsc")
        datasheet_raw = entry.get("datasheet_url")

        return ComponentSpec(
            component_id=str(entry.get("component_id", "")),
            ref=str(entry.get("ref", "")),
            value=str(entry.get("value", "")),
            footprint_id=str(entry.get("footprint_id", "")),
            description=str(entry.get("description", "")),
            expected_pads=int(entry.get("expected_pads", 0)),  # type: ignore[arg-type]
            expected_pad_type=str(entry.get("expected_pad_type", "smd")),
            body_width_mm=float(body_w) if body_w is not None else None,
            body_height_mm=float(body_h) if body_h is not None else None,
            model_rotation_z=float(entry.get("model_rotation_z", 0.0)),  # type: ignore[arg-type]
            model_offset_xy_max_mm=float(entry.get("model_offset_xy_max_mm", 0.0)),  # type: ignore[arg-type]
            kicad_ref_pad1_x=float(entry.get("kicad_ref_pad1_x", 0.0)),  # type: ignore[arg-type]
            kicad_ref_pad1_y=float(entry.get("kicad_ref_pad1_y", 0.0)),  # type: ignore[arg-type]
            lcsc=str(lcsc_raw) if lcsc_raw is not None else None,
            datasheet_url=str(datasheet_raw) if datasheet_raw is not None else None,
            verification_status=str(entry.get("verification_status", "unverified")),
            last_verified_commit=str(last_commit) if last_commit is not None else None,
            known_issues=known_issues,
            verified_fixes=verified_fixes,
            pins=pins,
        )

    @staticmethod
    def _spec_to_dict(spec: ComponentSpec) -> dict[str, object]:
        """Serialize a :class:`ComponentSpec` to a JSON-compatible dict."""
        return {
            "component_id": spec.component_id,
            "ref": spec.ref,
            "value": spec.value,
            "footprint_id": spec.footprint_id,
            "description": spec.description,
            "expected_pads": spec.expected_pads,
            "expected_pad_type": spec.expected_pad_type,
            "body_width_mm": spec.body_width_mm,
            "body_height_mm": spec.body_height_mm,
            "model_rotation_z": spec.model_rotation_z,
            "model_offset_xy_max_mm": spec.model_offset_xy_max_mm,
            "kicad_ref_pad1_x": spec.kicad_ref_pad1_x,
            "kicad_ref_pad1_y": spec.kicad_ref_pad1_y,
            "lcsc": spec.lcsc,
            "datasheet_url": spec.datasheet_url,
            "verification_status": spec.verification_status,
            "last_verified_commit": spec.last_verified_commit,
            "known_issues": [
                dataclasses.asdict(ki) for ki in spec.known_issues
            ],
            "verified_fixes": [
                dataclasses.asdict(vf) for vf in spec.verified_fixes
            ],
            "pins": [dataclasses.asdict(p) for p in spec.pins],
        }

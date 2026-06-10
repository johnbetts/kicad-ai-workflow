"""Loaders for part-class rules and human feedback locks.

Part rules (``data/part_rules.json``) attach per-part-class constraints
(keepouts, edge pins, isolation domains) to components by pattern match.
Feedback locks persist human sign-off corrections as constraints with
:attr:`ConstraintSource.HUMAN_FEEDBACK` so a correction can never regress.

Matching semantics (deliberately narrow -- no fuzzy matching):

* ``ref_prefix`` -- EXACT match against the component ref's alphabetic
  prefix: ``"K"`` matches ``K1`` and ``K12`` but never ``KA1``.
* ``footprint_contains`` -- case-insensitive substring of the footprint.

Unknown keys anywhere in either file raise: silent tolerance would hide
typos like ``"keepuot"`` that silently drop a safety constraint.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kicad_pipeline.exceptions import ConfigurationError
from kicad_pipeline.models.pcb import Point
from kicad_pipeline.placement_v2.ir import (
    Axis,
    CellKeepout,
    ConstraintSource,
    Edge,
    EdgePin,
    IsolationGap,
    KeepoutKind,
    PadRef,
    PinAttach,
    Polygon,
    SequenceAlong,
)

if TYPE_CHECKING:
    from pathlib import Path

    from kicad_pipeline.models.requirements import Component

logger = logging.getLogger(__name__)

_ALPHA_PREFIX_RE = re.compile(r"^([A-Za-z]+)")

_RULE_KEYS = frozenset({"match", "keepout", "edge_pin", "isolation_domain"})
_MATCH_KEYS = frozenset({"ref_prefix", "footprint_contains"})
_KEEPOUT_KEYS = frozenset({"kind", "polygon_mm"})
_GAP_KEYS = frozenset({"domain_a", "domain_b", "min_mm"})
_TOP_KEYS = frozenset({"rules", "isolation_gaps"})

_LOCK_KEYS: dict[str, frozenset[str]] = {
    "pin_attach": frozenset({"type", "src", "dst", "net", "max_mm", "ideal_mm"}),
    "sequence": frozenset({"type", "axis", "refs", "pitch_mm", "max_span_mm"}),
    "edge_pin": frozenset({"type", "ref", "edge", "face_out", "max_edge_distance_mm"}),
}


def ref_alpha_prefix(ref: str) -> str:
    """Return the leading alphabetic prefix of a ref designator ("K12" -> "K")."""
    m = _ALPHA_PREFIX_RE.match(ref)
    return m.group(1) if m else ""


@dataclass(frozen=True)
class PartRuleMatch:
    """Component matcher: exact ref prefix and/or footprint substring."""

    ref_prefix: str | None = None
    footprint_contains: str | None = None

    def matches(self, component: Component) -> bool:
        """True when the component satisfies every specified criterion."""
        if self.ref_prefix is None and self.footprint_contains is None:
            return False
        if self.ref_prefix is not None and ref_alpha_prefix(component.ref) != self.ref_prefix:
            return False
        return not (
            self.footprint_contains is not None
            and self.footprint_contains.lower() not in component.footprint.lower()
        )


@dataclass(frozen=True)
class KeepoutSpec:
    """Keepout template attached to every matching component (cell-local frame)."""

    kind: KeepoutKind
    polygon: Polygon


@dataclass(frozen=True)
class PartRule:
    """One part-class rule: a matcher plus the constraints it implies."""

    match: PartRuleMatch
    keepout: KeepoutSpec | None = None
    edge_pin: bool = False
    isolation_domain: str | None = None


@dataclass(frozen=True)
class PartRuleSet:
    """Validated contents of a part rules JSON file."""

    rules: tuple[PartRule, ...]
    isolation_gaps: tuple[IsolationGap, ...]


@dataclass(frozen=True)
class CompiledPartRules:
    """Part rules applied to a concrete component list."""

    keepouts: tuple[CellKeepout, ...]
    edge_pins: tuple[EdgePin, ...]
    isolation: tuple[IsolationGap, ...]
    domains: tuple[tuple[str, str], ...]  # (ref, domain) assignments


@dataclass(frozen=True)
class FeedbackLocks:
    """Constraints persisted from human sign-off feedback."""

    pin_attach: tuple[PinAttach, ...] = ()
    sequences: tuple[SequenceAlong, ...] = ()
    edge_pins: tuple[EdgePin, ...] = ()


def _fail(path: Path, message: str) -> ConfigurationError:
    return ConfigurationError(f"part rules file {path}: {message}")


def _check_keys(obj: dict[str, object], allowed: frozenset[str], ctx: str, path: Path) -> None:
    unknown = set(obj) - allowed
    if unknown:
        raise _fail(path, f"unknown key(s) {sorted(unknown)} in {ctx}; allowed: {sorted(allowed)}")


def _as_dict(obj: object, ctx: str, path: Path) -> dict[str, object]:
    if not isinstance(obj, dict):
        raise _fail(path, f"{ctx} must be an object, got {type(obj).__name__}")
    return obj


def _as_str(obj: object, ctx: str, path: Path) -> str:
    if not isinstance(obj, str):
        raise _fail(path, f"{ctx} must be a string, got {type(obj).__name__}")
    return obj


def _as_float(obj: object, ctx: str, path: Path) -> float:
    if isinstance(obj, bool) or not isinstance(obj, int | float):
        raise _fail(path, f"{ctx} must be a number, got {type(obj).__name__}")
    return float(obj)


def _parse_polygon(obj: object, ctx: str, path: Path) -> Polygon:
    if not isinstance(obj, list) or len(obj) < 3:
        raise _fail(path, f"{ctx} must be a list of >= 3 [x, y] points")
    points: list[Point] = []
    for i, entry in enumerate(obj):
        if not isinstance(entry, list) or len(entry) != 2:
            raise _fail(path, f"{ctx}[{i}] must be a 2-element [x, y] list")
        x = _as_float(entry[0], f"{ctx}[{i}].x", path)
        y = _as_float(entry[1], f"{ctx}[{i}].y", path)
        points.append(Point(x, y))
    return tuple(points)


def _parse_rule(obj: object, idx: int, path: Path) -> PartRule:
    rule = _as_dict(obj, f"rules[{idx}]", path)
    _check_keys(rule, _RULE_KEYS, f"rules[{idx}]", path)
    if "match" not in rule:
        raise _fail(path, f"rules[{idx}] missing required key 'match'")
    match_obj = _as_dict(rule["match"], f"rules[{idx}].match", path)
    _check_keys(match_obj, _MATCH_KEYS, f"rules[{idx}].match", path)
    if not match_obj:
        raise _fail(path, f"rules[{idx}].match must specify at least one criterion")
    ref_prefix = (
        _as_str(match_obj["ref_prefix"], f"rules[{idx}].match.ref_prefix", path)
        if "ref_prefix" in match_obj
        else None
    )
    fp_contains = (
        _as_str(match_obj["footprint_contains"], f"rules[{idx}].match.footprint_contains", path)
        if "footprint_contains" in match_obj
        else None
    )
    keepout: KeepoutSpec | None = None
    if "keepout" in rule:
        ko = _as_dict(rule["keepout"], f"rules[{idx}].keepout", path)
        _check_keys(ko, _KEEPOUT_KEYS, f"rules[{idx}].keepout", path)
        if "kind" not in ko or "polygon_mm" not in ko:
            raise _fail(path, f"rules[{idx}].keepout requires 'kind' and 'polygon_mm'")
        kind_str = _as_str(ko["kind"], f"rules[{idx}].keepout.kind", path)
        try:
            kind = KeepoutKind(kind_str)
        except ValueError as exc:
            valid = sorted(k.value for k in KeepoutKind)
            raise _fail(path, f"rules[{idx}].keepout.kind {kind_str!r} not in {valid}") from exc
        polygon = _parse_polygon(ko["polygon_mm"], f"rules[{idx}].keepout.polygon_mm", path)
        keepout = KeepoutSpec(kind=kind, polygon=polygon)
    edge_pin = rule.get("edge_pin", False)
    if not isinstance(edge_pin, bool):
        raise _fail(path, f"rules[{idx}].edge_pin must be a boolean")
    isolation_domain = (
        _as_str(rule["isolation_domain"], f"rules[{idx}].isolation_domain", path)
        if "isolation_domain" in rule
        else None
    )
    return PartRule(
        match=PartRuleMatch(ref_prefix=ref_prefix, footprint_contains=fp_contains),
        keepout=keepout,
        edge_pin=edge_pin,
        isolation_domain=isolation_domain,
    )


def load_part_rules(path: Path) -> PartRuleSet:
    """Load and strictly validate a part rules JSON file.

    Raises :class:`ConfigurationError` for invalid JSON, unknown keys,
    or malformed values -- never tolerates and never warns-and-continues.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data: object = json.load(fh)
    except OSError as exc:
        raise _fail(path, f"cannot read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _fail(path, f"invalid JSON: {exc}") from exc
    top = _as_dict(data, "top level", path)
    _check_keys(top, _TOP_KEYS, "top level", path)
    rules_obj = top.get("rules", [])
    if not isinstance(rules_obj, list):
        raise _fail(path, "'rules' must be a list")
    rules = tuple(_parse_rule(r, i, path) for i, r in enumerate(rules_obj))
    gaps_obj = top.get("isolation_gaps", [])
    if not isinstance(gaps_obj, list):
        raise _fail(path, "'isolation_gaps' must be a list")
    gaps: list[IsolationGap] = []
    for i, g in enumerate(gaps_obj):
        gap = _as_dict(g, f"isolation_gaps[{i}]", path)
        _check_keys(gap, _GAP_KEYS, f"isolation_gaps[{i}]", path)
        if set(gap) != _GAP_KEYS:
            raise _fail(path, f"isolation_gaps[{i}] requires keys {sorted(_GAP_KEYS)}")
        gaps.append(
            IsolationGap(
                domain_a=_as_str(gap["domain_a"], f"isolation_gaps[{i}].domain_a", path),
                domain_b=_as_str(gap["domain_b"], f"isolation_gaps[{i}].domain_b", path),
                min_mm=_as_float(gap["min_mm"], f"isolation_gaps[{i}].min_mm", path),
                source=ConstraintSource.PART_RULE,
            )
        )
    logger.debug("loaded %d part rules, %d isolation gaps from %s", len(rules), len(gaps), path)
    return PartRuleSet(rules=rules, isolation_gaps=tuple(gaps))


def apply_part_rules(
    rule_set: PartRuleSet, components: tuple[Component, ...]
) -> CompiledPartRules:
    """Apply loaded rules to a component list, producing concrete constraints.

    Keepouts are owned by each matching component (owner-local frame);
    edge pins and domain assignments are emitted per matching ref.
    Isolation gaps are board-level and pass through unconditionally.
    """
    keepouts: list[CellKeepout] = []
    edge_pins: list[EdgePin] = []
    domains: list[tuple[str, str]] = []
    for rule in rule_set.rules:
        for comp in components:
            if not rule.match.matches(comp):
                continue
            if rule.keepout is not None:
                keepouts.append(
                    CellKeepout(
                        owner=comp.ref,
                        polygon=rule.keepout.polygon,
                        kind=rule.keepout.kind,
                        source=ConstraintSource.PART_RULE,
                    )
                )
            if rule.edge_pin:
                edge_pins.append(
                    EdgePin(ref=comp.ref, edge=None, source=ConstraintSource.PART_RULE)
                )
            if rule.isolation_domain is not None:
                domains.append((comp.ref, rule.isolation_domain))
    return CompiledPartRules(
        keepouts=tuple(keepouts),
        edge_pins=tuple(edge_pins),
        isolation=rule_set.isolation_gaps,
        domains=tuple(domains),
    )


def _lock_fail(path: Path, message: str) -> ValueError:
    return ValueError(f"feedback locks file {path}: {message}")


def _parse_pad_ref(obj: object, ctx: str, path: Path) -> PadRef:
    if not isinstance(obj, str) or "." not in obj:
        raise _lock_fail(path, f"{ctx} must be a 'REF.PIN' string, got {obj!r}")
    ref, pin = obj.split(".", 1)
    if not ref or not pin:
        raise _lock_fail(path, f"{ctx} must be a 'REF.PIN' string, got {obj!r}")
    return PadRef(ref=ref, pin=pin)


def _parse_lock(obj: object, idx: int, path: Path) -> PinAttach | SequenceAlong | EdgePin:
    if not isinstance(obj, dict):
        raise _lock_fail(path, f"locks[{idx}] must be an object")
    lock_type = obj.get("type")
    if not isinstance(lock_type, str) or lock_type not in _LOCK_KEYS:
        raise _lock_fail(
            path, f"locks[{idx}].type must be one of {sorted(_LOCK_KEYS)}, got {lock_type!r}"
        )
    unknown = set(obj) - _LOCK_KEYS[lock_type]
    if unknown:
        raise _lock_fail(path, f"locks[{idx}] has unknown key(s) {sorted(unknown)}")
    try:
        return _build_lock(obj, lock_type, path)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("feedback locks file"):
            raise
        raise _lock_fail(path, f"locks[{idx}] is malformed: {exc}") from exc


def _build_lock(
    obj: dict[str, object], lock_type: str, path: Path
) -> PinAttach | SequenceAlong | EdgePin:
    src = ConstraintSource.HUMAN_FEEDBACK
    if lock_type == "pin_attach":
        max_mm = _as_float_lock(obj["max_mm"], path)
        ideal = _as_float_lock(obj["ideal_mm"], path) if "ideal_mm" in obj else max_mm / 2.0
        return PinAttach(
            src=_parse_pad_ref(obj["src"], "src", path),
            dst=_parse_pad_ref(obj["dst"], "dst", path),
            net=str(obj["net"]),
            max_mm=max_mm,
            ideal_mm=ideal,
            source=src,
        )
    if lock_type == "sequence":
        refs_obj = obj["refs"]
        if not isinstance(refs_obj, list) or not all(isinstance(r, str) for r in refs_obj):
            raise _lock_fail(path, "sequence refs must be a list of strings")
        pitch = _as_float_lock(obj["pitch_mm"], path) if obj.get("pitch_mm") is not None else None
        span = (
            _as_float_lock(obj["max_span_mm"], path)
            if obj.get("max_span_mm") is not None
            else None
        )
        return SequenceAlong(
            axis=Axis(str(obj["axis"])),
            refs=tuple(refs_obj),
            pitch_mm=pitch,
            max_span_mm=span,
            source=src,
        )
    edge = Edge(str(obj["edge"])) if obj.get("edge") is not None else None
    face_out = obj.get("face_out", True)
    if not isinstance(face_out, bool):
        raise _lock_fail(path, "edge_pin face_out must be a boolean")
    max_dist = (
        _as_float_lock(obj["max_edge_distance_mm"], path) if "max_edge_distance_mm" in obj else 5.0
    )
    return EdgePin(
        ref=str(obj["ref"]),
        edge=edge,
        face_out=face_out,
        max_edge_distance_mm=max_dist,
        source=src,
    )


def _as_float_lock(obj: object, path: Path) -> float:
    if isinstance(obj, bool) or not isinstance(obj, int | float):
        raise _lock_fail(path, f"expected a number, got {type(obj).__name__}")
    return float(obj)


def load_feedback_locks(path: Path) -> FeedbackLocks:
    """Load human feedback locks; a missing file means "no locks".

    Raises :class:`ValueError` naming the path for any malformed content.
    """
    if not path.exists():
        logger.debug("no feedback locks file at %s", path)
        return FeedbackLocks()
    try:
        with path.open(encoding="utf-8") as fh:
            data: object = json.load(fh)
    except json.JSONDecodeError as exc:
        raise _lock_fail(path, f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"locks"}:
        raise _lock_fail(path, "top level must be an object with the single key 'locks'")
    locks_obj = data["locks"]
    if not isinstance(locks_obj, list):
        raise _lock_fail(path, "'locks' must be a list")
    attaches: list[PinAttach] = []
    sequences: list[SequenceAlong] = []
    edge_pins: list[EdgePin] = []
    for i, entry in enumerate(locks_obj):
        try:
            lock = _parse_lock(entry, i, path)
        except ValueError as exc:
            if str(exc).startswith("feedback locks file"):
                raise
            raise _lock_fail(path, f"locks[{i}] is malformed: {exc}") from exc
        if isinstance(lock, PinAttach):
            attaches.append(lock)
        elif isinstance(lock, SequenceAlong):
            sequences.append(lock)
        else:
            edge_pins.append(lock)
    logger.debug(
        "loaded %d feedback locks from %s",
        len(attaches) + len(sequences) + len(edge_pins),
        path,
    )
    return FeedbackLocks(
        pin_attach=tuple(attaches),
        sequences=tuple(sequences),
        edge_pins=tuple(edge_pins),
    )

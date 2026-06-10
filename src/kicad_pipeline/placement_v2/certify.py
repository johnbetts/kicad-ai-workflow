"""Stage 0 — certified part library for placement engine v2.

A board may only instantiate certified part tuples. Certification is
content-addressed: each :class:`PartCertificate` records a SHA-256 of a
canonical footprint serialization, so a footprint that drifts from the
KiCad library it was certified against invalidates its own certificate
(the library-sync check). Lookup is EXACT-key only — no substring or
lib-id fuzzy matching, no silent parametric fallback. A missing or
stale certificate is a build error with a one-command remedy.

See ``docs/placement_v2_architecture.md`` section 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from kicad_pipeline.exceptions import ValidationError
from kicad_pipeline.models.pcb import FootprintArc, FootprintCircle, FootprintLine
from kicad_pipeline.placement_v2.ir import Severity, Violation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kicad_pipeline.models.pcb import Footprint

logger = logging.getLogger(__name__)

_COORD_PRECISION = 6
"""Decimal places used when canonicalizing coordinates for hashing."""

_COURTYARD_LAYERS = ("F.CrtYd", "B.CrtYd")
"""Footprint graphic layers that define an explicit courtyard."""

_FALLBACK_COURTYARD_INFLATE_MM = 0.25
"""Pad-bbox inflation used when a footprint has no explicit courtyard."""

_REMEDY_COMMAND = "python scripts/verify_components.py --certify"
"""One-command remedy suggested when a part has no certificate."""

XY = tuple[float, float]
XYZ = tuple[float, float, float]
PolygonXY = tuple[XY, ...]


class UncertifiedPartError(ValidationError):
    """Raised when a part key has no certificate in the store."""


class CertificateMismatchError(ValidationError):
    """Raised when a footprint's hash no longer matches its certificate.

    This is the library-sync check: a footprint that changed since
    certification (different pads, courtyard, or lib_id) must be
    re-certified before it can be placed.
    """


@dataclass(frozen=True)
class PadGeom:
    """One pad's geometry, relative to the footprint origin."""

    pin: str
    x: float
    y: float
    width: float
    height: float
    through_hole: bool


@dataclass(frozen=True)
class PartCertificate:
    """Proven geometry and provenance for one certified part.

    ``model_sha256`` is ``None`` (with ``resolved=False``) when the 3D
    model path could not be hashed — e.g. unexpanded
    ``${KICAD10_3DMODEL_DIR}`` environment paths. The unresolved path
    is still stored for traceability.
    """

    key: str
    footprint_id: str
    footprint_sha256: str
    model_path: str | None
    model_sha256: str | None
    resolved: bool
    model_offset: XYZ
    model_rotation: XYZ
    pad_map: tuple[PadGeom, ...]
    courtyard: PolygonXY
    body: PolygonXY
    checks_passed: tuple[str, ...]
    certified_at: str
    kicad_lib_version: str


def _round(value: float) -> float:
    return round(value, _COORD_PRECISION)


def _derive_pad_map(fp: Footprint) -> tuple[PadGeom, ...]:
    """Pad geometry sorted by (pin, x, y) for deterministic ordering."""
    geoms = tuple(
        PadGeom(
            pin=pad.number,
            x=_round(pad.position.x),
            y=_round(pad.position.y),
            width=_round(pad.size_x),
            height=_round(pad.size_y),
            through_hole=pad.pad_type == "thru_hole",
        )
        for pad in fp.pads
    )
    return tuple(sorted(geoms, key=lambda g: (g.pin, g.x, g.y)))


def _bbox_polygon(x1: float, y1: float, x2: float, y2: float) -> PolygonXY:
    return (
        (_round(x1), _round(y1)),
        (_round(x2), _round(y1)),
        (_round(x2), _round(y2)),
        (_round(x1), _round(y2)),
    )


def _courtyard_points(fp: Footprint) -> tuple[XY, ...]:
    """All vertices of explicit courtyard graphics (order-insensitive)."""
    pts: list[XY] = []
    for graphic in fp.graphics:
        if graphic.layer not in _COURTYARD_LAYERS:
            continue
        if isinstance(graphic, FootprintLine):
            pts.extend(((graphic.start.x, graphic.start.y), (graphic.end.x, graphic.end.y)))
        elif isinstance(graphic, FootprintArc):
            pts.extend(
                (
                    (graphic.start.x, graphic.start.y),
                    (graphic.mid.x, graphic.mid.y),
                    (graphic.end.x, graphic.end.y),
                )
            )
        elif isinstance(graphic, FootprintCircle):
            pts.extend(((graphic.center.x, graphic.center.y), (graphic.end.x, graphic.end.y)))
    return tuple(pts)


def _derive_courtyard(fp: Footprint) -> PolygonXY:
    """Footprint-local courtyard polygon (canonical bounding rectangle).

    Uses the bounding box of explicit courtyard-layer graphics when
    present (order-insensitive, so hashing is stable). Otherwise falls
    back to the pad bounding box inflated by 0.25 mm.
    """
    pts = _courtyard_points(fp)
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return _bbox_polygon(min(xs), min(ys), max(xs), max(ys))
    inflate = _FALLBACK_COURTYARD_INFLATE_MM
    if not fp.pads:
        return _bbox_polygon(-inflate, -inflate, inflate, inflate)
    xs = [pad.position.x + sx * pad.size_x / 2.0 for pad in fp.pads for sx in (-1.0, 1.0)]
    ys = [pad.position.y + sy * pad.size_y / 2.0 for pad in fp.pads for sy in (-1.0, 1.0)]
    return _bbox_polygon(
        min(xs) - inflate, min(ys) - inflate, max(xs) + inflate, max(ys) + inflate
    )


def compute_footprint_sha256(fp: Footprint) -> str:
    """Hash a canonical serialization of a footprint's geometry.

    The serialization sorts pads and canonicalizes the courtyard as a
    bounding rectangle, so semantically identical footprints hash equal
    regardless of construction order. Includes lib_id, pad geometry
    (number, type, shape, position, size, drill, layers), and courtyard.
    """
    pads = sorted(
        (
            pad.number,
            pad.pad_type,
            pad.shape,
            _round(pad.position.x),
            _round(pad.position.y),
            _round(pad.size_x),
            _round(pad.size_y),
            _round(pad.drill_diameter or 0.0),
            ",".join(sorted(pad.layers)),
        )
        for pad in fp.pads
    )
    payload = {
        "courtyard": [list(p) for p in _derive_courtyard(fp)],
        "lib_id": fp.lib_id,
        "pads": [list(p) for p in pads],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_model(path_str: str) -> tuple[str | None, bool]:
    """Hash a 3D model file if its path resolves to a readable file.

    Returns ``(sha256, resolved)``. Unexpanded environment paths (e.g.
    ``${KICAD10_3DMODEL_DIR}/...``) and missing files yield
    ``(None, False)``.
    """
    if "${" in path_str:
        return (None, False)
    path = Path(path_str)
    if not path.is_file():
        return (None, False)
    return (hashlib.sha256(path.read_bytes()).hexdigest(), True)


def build_certificate(
    fp: Footprint,
    *,
    key: str,
    kicad_lib_version: str = "kicad10",
    certified_at: str,
    checks_passed: tuple[str, ...] = (),
    body: PolygonXY | None = None,
) -> PartCertificate:
    """Build a :class:`PartCertificate` from a footprint.

    Pad map, courtyard, and hash are derived from the footprint; the 3D
    model offset/rotation come from ``fp.models[0]`` when present.
    ``body`` is the projected 3D body outline when the caller has one
    (e.g. from STEP projection); it defaults to the courtyard.
    ``certified_at`` is caller-provided (ISO date) so this function
    stays pure and deterministic.
    """
    courtyard = _derive_courtyard(fp)
    model_path: str | None = None
    model_sha: str | None = None
    resolved = False
    offset: XYZ = (0.0, 0.0, 0.0)
    rotation: XYZ = (0.0, 0.0, 0.0)
    if fp.models:
        model = fp.models[0]
        model_path = model.path
        model_sha, resolved = _hash_model(model.path)
        offset = model.offset
        rotation = model.rotate
    return PartCertificate(
        key=key,
        footprint_id=fp.lib_id,
        footprint_sha256=compute_footprint_sha256(fp),
        model_path=model_path,
        model_sha256=model_sha,
        resolved=resolved,
        model_offset=offset,
        model_rotation=rotation,
        pad_map=_derive_pad_map(fp),
        courtyard=courtyard,
        body=body if body is not None else courtyard,
        checks_passed=checks_passed,
        certified_at=certified_at,
        kicad_lib_version=kicad_lib_version,
    )


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"Expected number in certificate JSON, got {value!r}")
    return float(value)


def _as_xyz(value: object) -> XYZ:
    items = cast("list[object]", value)
    if len(items) != 3:
        raise ValidationError(f"Expected 3-element vector, got {value!r}")
    return (_as_float(items[0]), _as_float(items[1]), _as_float(items[2]))


def _as_polygon(value: object) -> PolygonXY:
    return tuple(
        (_as_float(cast("list[object]", pt)[0]), _as_float(cast("list[object]", pt)[1]))
        for pt in cast("list[object]", value)
    )


def _cert_to_dict(cert: PartCertificate) -> dict[str, object]:
    return {
        "key": cert.key,
        "footprint_id": cert.footprint_id,
        "footprint_sha256": cert.footprint_sha256,
        "model_path": cert.model_path,
        "model_sha256": cert.model_sha256,
        "resolved": cert.resolved,
        "model_offset": list(cert.model_offset),
        "model_rotation": list(cert.model_rotation),
        "pad_map": [
            {
                "pin": g.pin,
                "x": g.x,
                "y": g.y,
                "width": g.width,
                "height": g.height,
                "through_hole": g.through_hole,
            }
            for g in cert.pad_map
        ],
        "courtyard": [list(p) for p in cert.courtyard],
        "body": [list(p) for p in cert.body],
        "checks_passed": list(cert.checks_passed),
        "certified_at": cert.certified_at,
        "kicad_lib_version": cert.kicad_lib_version,
    }


def _cert_from_dict(data: dict[str, object]) -> PartCertificate:
    model_path = data["model_path"]
    model_sha = data["model_sha256"]
    pad_map = tuple(
        PadGeom(
            pin=str(entry["pin"]),
            x=_as_float(entry["x"]),
            y=_as_float(entry["y"]),
            width=_as_float(entry["width"]),
            height=_as_float(entry["height"]),
            through_hole=bool(entry["through_hole"]),
        )
        for entry in cast("list[dict[str, object]]", data["pad_map"])
    )
    return PartCertificate(
        key=str(data["key"]),
        footprint_id=str(data["footprint_id"]),
        footprint_sha256=str(data["footprint_sha256"]),
        model_path=None if model_path is None else str(model_path),
        model_sha256=None if model_sha is None else str(model_sha),
        resolved=bool(data["resolved"]),
        model_offset=_as_xyz(data["model_offset"]),
        model_rotation=_as_xyz(data["model_rotation"]),
        pad_map=pad_map,
        courtyard=_as_polygon(data["courtyard"]),
        body=_as_polygon(data["body"]),
        checks_passed=tuple(str(c) for c in cast("list[object]", data["checks_passed"])),
        certified_at=str(data["certified_at"]),
        kicad_lib_version=str(data["kicad_lib_version"]),
    )


@dataclass(frozen=True)
class CertificateStore:
    """Immutable, exact-key store of part certificates.

    ``add`` returns a NEW store; ``lookup`` is exact-key only — there is
    deliberately no substring or fuzzy matching, and no fallback.
    """

    certs: tuple[PartCertificate, ...] = ()

    @staticmethod
    def load(path: Path) -> CertificateStore:
        """Load a store from a JSON file; a missing file is an empty store."""
        if not path.is_file():
            logger.info("Certificate store %s not found; starting empty", path)
            return CertificateStore()
        with path.open(encoding="utf-8") as fh:
            data = cast("dict[str, object]", json.load(fh))
        raw = cast("list[dict[str, object]]", data.get("certificates", []))
        certs = tuple(_cert_from_dict(entry) for entry in raw)
        logger.info("Loaded %d certificates from %s", len(certs), path)
        return CertificateStore(certs=certs)

    def save(self, path: Path) -> None:
        """Write the store as JSON (sorted keys, stable diffs)."""
        payload = {
            "certificates": [_cert_to_dict(c) for c in sorted(self.certs, key=lambda c: c.key)],
            "schema_version": 1,
        }
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved %d certificates to %s", len(self.certs), path)

    def add(self, cert: PartCertificate) -> CertificateStore:
        """New store with *cert* added (replacing any cert with the same key)."""
        kept = tuple(c for c in self.certs if c.key != cert.key)
        return CertificateStore(certs=(*kept, cert))

    def lookup(self, key: str) -> PartCertificate:
        """Exact-key lookup; raises :class:`UncertifiedPartError` if absent."""
        for cert in self.certs:
            if cert.key == key:
                return cert
        raise UncertifiedPartError(
            f"No certificate for part {key!r}. Certification is a hard gate: "
            f"run `{_REMEDY_COMMAND} {key}` to certify it."
        )

    def lookup_verified(self, key: str, fp: Footprint) -> PartCertificate:
        """Lookup AND assert the footprint still matches its certificate.

        Raises :class:`CertificateMismatchError` when the footprint's
        canonical hash drifted from the certified hash (stale library).
        """
        cert = self.lookup(key)
        actual = compute_footprint_sha256(fp)
        if actual != cert.footprint_sha256:
            raise CertificateMismatchError(
                f"Footprint {fp.lib_id!r} for part {key!r} no longer matches its "
                f"certificate (certified {cert.footprint_sha256[:12]}, actual "
                f"{actual[:12]}). The library changed since certification: "
                f"re-run `{_REMEDY_COMMAND} {key}`."
            )
        return cert

    def __contains__(self, key: str) -> bool:
        return any(c.key == key for c in self.certs)

    def __len__(self) -> int:
        return len(self.certs)


def certify_board_footprints(
    footprints: Mapping[str, Footprint],
    keys: Mapping[str, str],
    store: CertificateStore,
) -> tuple[Violation, ...]:
    """Hard certification gate: verify every footprint against the store.

    For each ref, looks up its part key and verifies the footprint hash
    against the certificate. Returns one CRITICAL violation per failure
    (missing key, missing certificate, or hash drift) so the caller gets
    the complete list. An empty tuple means the gate passes.
    """
    violations: list[Violation] = []
    for ref in sorted(footprints):
        fp = footprints[ref]
        key = keys.get(ref)
        if key is None:
            violations.append(
                Violation(
                    constraint="certificate",
                    refs=(ref,),
                    severity=Severity.CRITICAL,
                    measured=0.0,
                    limit=0.0,
                    message=(
                        f"{ref}: no part key provided — every placed footprint "
                        f"must map to a certified part (LCSC or parametric key)"
                    ),
                )
            )
            continue
        try:
            store.lookup_verified(key, fp)
        except (UncertifiedPartError, CertificateMismatchError) as exc:
            violations.append(
                Violation(
                    constraint="certificate",
                    refs=(ref,),
                    severity=Severity.CRITICAL,
                    measured=0.0,
                    limit=0.0,
                    message=f"{ref}: {exc}",
                )
            )
    if violations:
        logger.warning(
            "Certification gate FAILED: %d of %d footprints uncertified",
            len(violations),
            len(footprints),
        )
    else:
        logger.info("Certification gate passed for %d footprints", len(footprints))
    return tuple(violations)

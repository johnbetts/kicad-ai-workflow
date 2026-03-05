"""Board template registry for standard form factors.

Provides pre-defined board templates for common form factors (Raspberry Pi HAT,
Arduino Uno shield, generic boards) including dimensions, mounting holes, and
fixed component positions.
"""

from __future__ import annotations

from dataclasses import dataclass

from kicad_pipeline.exceptions import ConfigurationError
from kicad_pipeline.models.requirements import MechanicalConstraints


@dataclass(frozen=True)
class MountingHoleDef:
    """Definition of a single mounting hole on a board template.

    Attributes:
        x_mm: X position in mm from the board origin.
        y_mm: Y position in mm from the board origin.
        diameter_mm: Drill diameter in mm.
    """

    x_mm: float
    y_mm: float
    diameter_mm: float = 2.7


@dataclass(frozen=True)
class FixedComponentDef:
    """Definition of a component with a fixed position on a board template.

    Attributes:
        ref_pattern: Reference designator pattern (e.g. "J1").
        x_mm: X position in mm from the board origin.
        y_mm: Y position in mm from the board origin.
        rotation: Rotation angle in degrees.
        description: Human-readable description of the component.
    """

    ref_pattern: str
    x_mm: float
    y_mm: float
    rotation: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class BoardTemplate:
    """A pre-defined board form factor with mechanical constraints.

    Attributes:
        name: Template identifier (e.g. "RPI_HAT").
        board_width_mm: Board width in mm.
        board_height_mm: Board height in mm.
        mounting_holes: Positions and sizes of mounting holes.
        fixed_components: Components with fixed positions on the board.
        description: Human-readable description.
        corner_radius_mm: Corner rounding radius in mm (0 for sharp corners).
    """

    name: str
    board_width_mm: float
    board_height_mm: float
    mounting_holes: tuple[MountingHoleDef, ...]
    fixed_components: tuple[FixedComponentDef, ...] = ()
    description: str = ""
    corner_radius_mm: float = 0.0


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, BoardTemplate] = {
    "RPI_HAT": BoardTemplate(
        name="RPI_HAT",
        board_width_mm=65.0,
        board_height_mm=56.0,
        mounting_holes=(
            MountingHoleDef(x_mm=3.5, y_mm=3.5),
            MountingHoleDef(x_mm=3.5, y_mm=52.5),
            MountingHoleDef(x_mm=61.5, y_mm=3.5),
            MountingHoleDef(x_mm=61.5, y_mm=52.5),
        ),
        fixed_components=(
            FixedComponentDef(
                ref_pattern="J1",
                x_mm=32.504,
                y_mm=3.502,
                description="40-pin GPIO header center (pin 1 at 8.374, 4.772)",
            ),
        ),
        description="Raspberry Pi HAT (Hardware Attached on Top) form factor",
        corner_radius_mm=3.0,
    ),
    "ARDUINO_UNO": BoardTemplate(
        name="ARDUINO_UNO",
        board_width_mm=68.6,
        board_height_mm=53.3,
        mounting_holes=(
            MountingHoleDef(x_mm=14.0, y_mm=2.54),
            MountingHoleDef(x_mm=15.24, y_mm=50.8),
            MountingHoleDef(x_mm=66.04, y_mm=7.62),
            MountingHoleDef(x_mm=66.04, y_mm=35.56),
        ),
        description="Arduino Uno R3 shield form factor",
    ),
    "GENERIC_50X50": BoardTemplate(
        name="GENERIC_50X50",
        board_width_mm=50.0,
        board_height_mm=50.0,
        mounting_holes=(
            MountingHoleDef(x_mm=3.5, y_mm=3.5, diameter_mm=3.2),
            MountingHoleDef(x_mm=46.5, y_mm=3.5, diameter_mm=3.2),
            MountingHoleDef(x_mm=3.5, y_mm=46.5, diameter_mm=3.2),
            MountingHoleDef(x_mm=46.5, y_mm=46.5, diameter_mm=3.2),
        ),
        description="Generic 50x50mm board with M3 corner mounting holes",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_template(name: str) -> BoardTemplate:
    """Retrieve a board template by name.

    Args:
        name: Template name (case-insensitive).

    Returns:
        The matching :class:`BoardTemplate`.

    Raises:
        ConfigurationError: If no template with that name exists.
    """
    key = name.upper()
    if key not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES.keys()))
        raise ConfigurationError(
            f"Unknown board template '{name}'; available: {available}"
        )
    return _TEMPLATES[key]


def list_templates() -> list[str]:
    """Return the names of all available board templates.

    Returns:
        Sorted list of template name strings.
    """
    return sorted(_TEMPLATES.keys())


def detect_template(mechanical: MechanicalConstraints | None) -> BoardTemplate | None:
    """Auto-detect a board template from mechanical constraints.

    Detection priority:

    1. Explicit ``board_template`` field on *mechanical*.
    2. Keywords in ``mechanical.notes`` (e.g. "raspberry pi hat").
    3. Dimensions matching a known template within 0.5mm tolerance.

    Args:
        mechanical: Mechanical constraints from the project requirements.

    Returns:
        The matching :class:`BoardTemplate`, or ``None`` if no match is found.
    """
    if mechanical is None:
        return None

    # 1. Explicit board_template field
    if mechanical.board_template is not None:
        key = mechanical.board_template.upper().replace(" ", "_").replace("-", "_")
        if key in _TEMPLATES:
            return _TEMPLATES[key]

    # 2. Keyword matching on notes
    if mechanical.notes:
        notes_lower = mechanical.notes.lower()
        _keyword_map: list[tuple[tuple[str, ...], str]] = [
            (("raspberry pi hat", "rpi hat", "rpi_hat"), "RPI_HAT"),
            (("arduino uno", "arduino_uno"), "ARDUINO_UNO"),
        ]
        for keywords, tmpl_name in _keyword_map:
            if any(kw in notes_lower for kw in keywords):
                return _TEMPLATES[tmpl_name]

    # 3. Dimension matching (within 0.5mm tolerance)
    tol = 0.5
    for tmpl in _TEMPLATES.values():
        if (
            abs(mechanical.board_width_mm - tmpl.board_width_mm) < tol
            and abs(mechanical.board_height_mm - tmpl.board_height_mm) < tol
        ):
            return tmpl

    return None


def template_to_mechanical_constraints(tmpl: BoardTemplate) -> MechanicalConstraints:
    """Convert a :class:`BoardTemplate` to :class:`MechanicalConstraints`.

    Args:
        tmpl: Board template to convert.

    Returns:
        A :class:`MechanicalConstraints` suitable for use in
        :class:`~kicad_pipeline.models.requirements.ProjectRequirements`.
    """
    hole_positions = tuple(
        (hole.x_mm, hole.y_mm) for hole in tmpl.mounting_holes
    )
    drill = tmpl.mounting_holes[0].diameter_mm if tmpl.mounting_holes else 3.2
    return MechanicalConstraints(
        board_width_mm=tmpl.board_width_mm,
        board_height_mm=tmpl.board_height_mm,
        mounting_hole_diameter_mm=drill,
        mounting_hole_positions=hole_positions,
        notes=tmpl.description,
    )

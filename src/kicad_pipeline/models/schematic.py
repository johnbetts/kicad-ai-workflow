"""Data models for KiCad schematic representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


class StrokeType(Enum):
    """Line stroke style for schematic drawing primitives."""

    DEFAULT = "default"
    DASH = "dash"
    DOT = "dot"
    DASH_DOT = "dash_dot"
    SOLID = "solid"


@dataclass(frozen=True)
class Point:
    """2D coordinate in mm."""

    x: float
    y: float


@dataclass(frozen=True)
class Stroke:
    """Line stroke style."""

    width: float = 0.0
    stroke_type: StrokeType = StrokeType.DEFAULT


@dataclass(frozen=True)
class FontEffect:
    """Text font settings."""

    size_x: float = 1.27
    size_y: float = 1.27
    bold: bool = False
    italic: bool = False
    hidden: bool = False
    justify: str = ""  # "", "left", "right"


@dataclass(frozen=True)
class TextProperty:
    """A positioned text label (ref, value, etc.) on a symbol."""

    text: str
    position: Point
    rotation: float = 0.0
    effects: FontEffect = field(default_factory=FontEffect)


@dataclass(frozen=True)
class LibPin:
    """A pin definition inside a lib_symbol."""

    number: str
    name: str
    pin_type: str  # "input", "output", "bidirectional", "passive", "power_in", etc.
    at: Point
    rotation: float  # degrees
    length: float  # mm, typically 2.54
    name_effects: FontEffect = field(default_factory=FontEffect)
    number_effects: FontEffect = field(default_factory=FontEffect)


@dataclass(frozen=True)
class LibPolyline:
    """A polyline in a lib_symbol (for drawing the body)."""

    points: tuple[Point, ...]
    stroke: Stroke = field(default_factory=Stroke)
    fill: str = "none"  # "none", "outline", "background"


@dataclass(frozen=True)
class LibRectangle:
    """A rectangle in a lib_symbol."""

    start: Point
    end: Point
    stroke: Stroke = field(default_factory=Stroke)
    fill: str = "background"


@dataclass(frozen=True)
class LibCircle:
    """A circle in a lib_symbol."""

    center: Point
    radius: float
    stroke: Stroke = field(default_factory=Stroke)
    fill: str = "none"


LibShape: TypeAlias = LibPolyline | LibRectangle | LibCircle


@dataclass(frozen=True)
class LibSymbol:
    """A symbol definition for the lib_symbols section."""

    lib_id: str  # "Device:R", "kicad-ai:ESP32-S3-WROOM-1"
    pins: tuple[LibPin, ...]
    shapes: tuple[LibPolyline | LibRectangle | LibCircle, ...]  # drawing primitives
    extends: str | None = None  # for unit-based symbols


@dataclass(frozen=True)
class SymbolInstance:
    """A placed symbol instance in the schematic."""

    lib_id: str  # "Device:R"
    ref: str  # "R1"
    value: str  # "10k"
    footprint: str  # "R_0805"
    position: Point
    rotation: float = 0.0
    lcsc: str | None = None
    uuid: str = ""
    unit: int = 1
    in_bom: bool = True
    on_board: bool = True
    ref_property: TextProperty | None = None
    value_property: TextProperty | None = None


@dataclass(frozen=True)
class Wire:
    """A schematic wire segment."""

    start: Point
    end: Point
    stroke: Stroke = field(default_factory=Stroke)
    uuid: str = ""


@dataclass(frozen=True)
class Junction:
    """A wire junction (T or + connection)."""

    position: Point
    diameter: float = 0.0
    uuid: str = ""


@dataclass(frozen=True)
class NoConnect:
    """A no-connect marker on a pin."""

    position: Point
    uuid: str = ""


@dataclass(frozen=True)
class Label:
    """A local net label."""

    text: str
    position: Point
    rotation: float = 0.0
    effects: FontEffect = field(default_factory=FontEffect)
    uuid: str = ""


@dataclass(frozen=True)
class GlobalLabel:
    """A global net label (cross-sheet connection)."""

    text: str
    shape: str  # "input", "output", "bidirectional", "tri_state", "passive"
    position: Point
    rotation: float = 0.0
    effects: FontEffect = field(default_factory=FontEffect)
    uuid: str = ""


@dataclass(frozen=True)
class PowerSymbol:
    """A power symbol (VCC, GND, etc.) instance."""

    lib_id: str  # "power:+3.3V", "power:GND"
    position: Point
    ref: str  # "#PWR01"
    value: str  # "+3.3V"
    rotation: float = 0.0
    uuid: str = ""


@dataclass(frozen=True)
class Schematic:
    """Complete KiCad schematic."""

    lib_symbols: tuple[LibSymbol, ...]
    symbols: tuple[SymbolInstance, ...]
    power_symbols: tuple[PowerSymbol, ...]
    wires: tuple[Wire, ...]
    junctions: tuple[Junction, ...]
    no_connects: tuple[NoConnect, ...]
    labels: tuple[Label, ...]
    global_labels: tuple[GlobalLabel, ...]
    version: int = 20250114
    generator: str = "kicad-ai-pipeline"
    generator_version: str = "9.0"
    paper: str = "A4"
    title: str = ""
    date: str = ""
    revision: str = ""
    company: str = ""

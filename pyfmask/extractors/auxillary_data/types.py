from enum import Enum
from enum import auto
from typing import NamedTuple


class AuxTypes(Enum):
    GSWO: int = auto()
    DEM: int = auto()
    MAPZEN: int = auto()


class Coordinate(NamedTuple):
    x: float
    y: float


class BoundingBox(NamedTuple):
    NORTH: float
    EAST: float
    SOUTH: float
    WEST: float

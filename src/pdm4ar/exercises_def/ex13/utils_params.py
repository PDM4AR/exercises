from dataclasses import dataclass


@dataclass(frozen=True)
class PlanetParams:
    center: list[float, float]
    radius: float


@dataclass(frozen=True)
class AsteroidParams:
    start: list[float, float]
    radius: float
    velocity: list[float, float]
    orientation: float

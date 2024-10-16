from dataclasses import dataclass

@dataclass(frozen=True)
class SatelliteParams():
    orbit_r: float
    omega: float
    tau: float
    radius: float

@dataclass(frozen=True)
class PlanetParams():
    center: list[float, float]
    radius: float
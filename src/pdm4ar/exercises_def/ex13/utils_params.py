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


@dataclass(frozen=True)
class DynamicsDisturbanceParams:
    """A short external acceleration burst applied by the simulator."""

    start_time: float
    duration: float
    linear_acc_sigma: float
    angular_acc_sigma: float
    seed: int

    def __post_init__(self):
        if self.start_time < 0:
            raise ValueError("Disturbance start_time must be non-negative.")
        if self.duration <= 0:
            raise ValueError("Disturbance duration must be positive.")
        if self.linear_acc_sigma < 0 or self.angular_acc_sigma < 0:
            raise ValueError("Disturbance standard deviations must be non-negative.")

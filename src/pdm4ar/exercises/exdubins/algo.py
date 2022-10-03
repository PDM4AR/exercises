from typing import Tuple, List
import numpy as np
from dg_commons import SE2Transform
from pdm4ar.exercises.exdubins.structures import Curve, Line, Path


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> float:
    pass


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> Tuple[Curve, Curve]:
    pass


def calculate_tangent_btw_curves(curve1: Curve, curve2: Curve) -> List[Line]:
    pass


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    pass


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    pass

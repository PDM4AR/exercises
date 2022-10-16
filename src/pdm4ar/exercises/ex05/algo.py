from typing import List
import numpy as np
from dg_commons import SE2Transform
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence
from dg_commons import SE2Transform

from pdm4ar.exercises.exdubins.structures import Curve, Line, Path, Segment, TurningCircle
from pdm4ar.exercises_def.exdubins.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


@dataclass(frozen=True)
class DubinsParam:
    min_radius: float

    def __post__init__(self):
        assert self.min_radius > 0, "Minimum radius has to be larger than 0"


class Dubins(PathPlanner):
    def __int__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[Segment]:
        """ Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        self.path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(self.path) 
        return se2_list


class ReedsShepp(PathPlanner):
    def __int__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """ Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations the car needs to follow
        """
        self.path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(self.path)
        return se2_list


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # TODO implement here your solution
    return DubinsParam(min_radius=0)


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    return TurningCircle(left_circle=Curve.create_circle(), right_circle=Curve.create_circle())


def calculate_tangent_btw_circles(circle1: Curve, circle2: Curve) -> List[Line]:
    # TODO implement here your solution
    return [] # i.e., [Line(),]


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Important: Please keep segments with zero length in the return list!
    return [] # e.g., [Curve(), Line(),..]


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Important: Please keep segments with zero length in the return list!
    return [] # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]

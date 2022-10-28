from typing import List
import numpy as np
from dg_commons import SE2Transform
from abc import ABC, abstractmethod
from typing import Sequence
from dg_commons import SE2Transform

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __int__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[SE2Transform]:
        """ Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        self.path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        # fixme why self?!
        se2_list = extract_path_points(self.path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __int__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """ Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow 
        """
        self.path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        # same as above
        se2_list = extract_path_points(self.path)
        return se2_list


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # TODO implement here your solution
    return DubinsParam(min_radius=0)


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    dummy_circle = Curve.create_circle(center=SE2Transform.identity(), config_on_circle=SE2Transform.identity,
                                       radius=0.1, curve_type=DubinsSegmentType.LEFT)  # TODO remove
    return TurningCircle(left=dummy_circle, right=dummy_circle)


def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> List[Line]:
    # TODO implement here your solution
    return []  # i.e., [Line(),...]


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!
    return []  # e.g., [Curve(), Line(),..]


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
    return []  # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]

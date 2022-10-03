from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from dg_commons import SE2Transform

from pdm4ar.exercises.exdubins.algo import *
from pdm4ar.exercises.exdubins.structures import Segment
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

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[Segment]: # TODO change to Sequence[SE2Transform]:
        wheel_base = 3 # e.g.
        max_steering_angle = np.pi/6 # e.g.
        radius = calculate_car_turning_radius(wheel_base, max_steering_angle)
        path = calculate_dubins_path(start, end, radius=radius)
        se2_list = extract_path_points(path)

        return se2_list


class ReedsShepp(PathPlanner):
    def __int__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        # todo implement here
        return [start, end]

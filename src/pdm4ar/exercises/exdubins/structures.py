from abc import abstractmethod
from enum import IntEnum, unique
from typing import List

import numpy as np
from dg_commons import SE2Transform


def mod_2_pi(x: float) -> float:
    return x - 2 * np.pi * np.floor(x / (2 * np.pi))


@unique
class DubinsSegmentType(IntEnum):
    RIGHT = -1
    STRAIGHT = 0
    LEFT = 1


@unique
class Gear(IntEnum):
    REVERSE = -1
    FORWARD = 1


class Segment:
    def __init__(self, segment_type: DubinsSegmentType, start_config: SE2Transform, end_config: SE2Transform, gear: Gear):
        self.type = segment_type
        self.start_config = start_config
        self.end_config = end_config
        self.gear = gear
        self.length: float

    @abstractmethod
    def __str__(self):
        pass


Path = List[Segment]


class Line(Segment):
    def __init__(self, start_config: SE2Transform, end_config: SE2Transform, gear: Gear = Gear.FORWARD):
        self.length = np.linalg.norm(end_config.p - start_config.p)
        if np.abs(self.length) >= 1e-8:
            self.direction = (end_config.p - start_config.p) / self.length
        else:
            self.direction = SE2Transform.identity().p
        super().__init__(DubinsSegmentType.STRAIGHT, start_config, end_config, gear)

    def __str__(self):
        return f"S{'-' if self.gear is Gear.REVERSE else ''}({self.length :.1f})"


class Curve(Segment):
    def __init__(self, start_config: SE2Transform, end_config: SE2Transform, center: SE2Transform,
                 radius: float, curve_type: DubinsSegmentType, arc_angle: float = 2 * np.pi, gear: Gear = Gear.FORWARD):

        assert center.theta == 0
        assert curve_type is not DubinsSegmentType.STRAIGHT

        # self.length = TODO TBD if want student to implement
        self.radius = radius
        self.center = center
        self.arc_angle = mod_2_pi(arc_angle)
        super().__init__(curve_type, start_config, end_config, gear)

    def __str__(self):
        return f"L{'-' if self.gear is Gear.REVERSE else ''}({np.rad2deg(self.arc_angle):.1f})" if self.type is DubinsSegmentType.LEFT\
            else f"R{'-' if self.gear is Gear.REVERSE else ''}({np.rad2deg(self.arc_angle):.1f})"

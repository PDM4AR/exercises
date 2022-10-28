from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import List
import numpy as np
from dg_commons import SE2Transform


def mod_2_pi(x: float) -> float:
    # fixme: simply use https://docs.python.org/3/library/math.html#math.fmod
    return x - 2 * np.pi * np.floor(x / (2 * np.pi))


@dataclass(frozen=True)
class DubinsParam:
    min_radius: float

    def __post__init__(self):
        assert self.min_radius > 0, "Minimum radius has to be larger than 0"


@unique
class DubinsSegmentType(IntEnum):
    RIGHT = -1
    STRAIGHT = 0
    LEFT = 1


@unique
class Gear(IntEnum):
    REVERSE = -1
    FORWARD = 1


class Segment(ABC):
    """ Abstract class defining the basic properties of a path segment  """

    def __init__(self, segment_type: DubinsSegmentType, start_config: SE2Transform, end_config: SE2Transform,
                 gear: Gear):
        self.type = segment_type
        self.start_config = start_config
        self.end_config = end_config
        self.gear = gear
        self.length: float

    @abstractmethod
    def __str__(self):
        pass


Path = List[Segment]
"""Here we consider a path as a list of segments"""


class Line(Segment):
    """ Class defining a line segment of a path
    
    Attributes:
    ----------
        type (fixed) :      DubinsSegmentType.STRAIGHT 
        
        start_config :      SE2Transform
            The configuration (x,y,theta) of the car at the start of the line
        
        end_config :        SE2Transform
            The configuration (x,y,theta) of the car at the end of the line

        length :            float
            The length of the line segment, i.e., the distance between start and end configuration

        direction:          np.array
            A unit vector pointing from start to end configuration
            If start == end, then direction = SE2Transform.identity().p

        gear:               Gear        (Default = Gear.Forward)
            Whether the car completes the line from start to end in forward gear or reverse gear


     """

    def __init__(self, start_config: SE2Transform, end_config: SE2Transform, gear: Gear = Gear.FORWARD):
        self.length = np.linalg.norm(end_config.p - start_config.p)
        if np.abs(self.length) >= 1e-8:
            self.direction = (end_config.p - start_config.p) / self.length
        else:
            self.direction = SE2Transform.identity().p
        super().__init__(DubinsSegmentType.STRAIGHT, start_config, end_config, gear)

    def __str__(self) -> str:
        return f"S{'-' if self.gear is Gear.REVERSE else ''}({self.length :.1f})"

    def __repr__(self) -> str:
        return str(self)


class Curve(Segment):
    """ Class defining a curve segment of a path
    
    Attributes:
    ----------
        type   :   DubinsSegmentType.LEFT  or  DubinsSegmentType.RIGHT 
        
        start_config:   SE2Transform
            The configuration (x,y,theta) of the car at the start of the curve
        
        end_config:     SE2Transform
            The configuration (x,y,theta) of the car at the end of the curve

        center:         SE2Transform
            The center of the turning circle (x,y,theta==0)

        radius:         float
            Turning radius
        
        arc_angle:      float [0, 2*pi) (default = 0)
            Angle of the curve segment. Note that 2*pi == 0, since a full 360deg turn is never in the optimal path

        length:         float
            The length of the curve segment, i.e., radius * arc_angle

        direction:       np.array
            A unit vector pointing from start to end configuration
            If start == end, then direction = SE2Transform.identity().p

        gear:            Gear      (Default = Gear.Forward)
            Whether the car completes the curve from start to end in forward gear or reverse gear

     """

    def __init__(self, start_config: SE2Transform, end_config: SE2Transform, center: SE2Transform,
                 radius: float, curve_type: DubinsSegmentType, arc_angle: float = 0, gear: Gear = Gear.FORWARD):
        assert center.theta == 0
        assert curve_type is not DubinsSegmentType.STRAIGHT
        assert radius > 0

        self.length = radius * arc_angle
        self.radius = radius
        self.center = center
        self.arc_angle = mod_2_pi(arc_angle)
        super().__init__(curve_type, start_config, end_config, gear)

    def __str__(self) -> str:
        return f"L{'-' if self.gear is Gear.REVERSE else ''}({np.rad2deg(self.arc_angle):.1f})" if self.type is DubinsSegmentType.LEFT \
            else f"R{'-' if self.gear is Gear.REVERSE else ''}({np.rad2deg(self.arc_angle):.1f})"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def create_circle(center: SE2Transform, config_on_circle: SE2Transform, radius: float,
                      curve_type: DubinsSegmentType) -> 'Curve':
        """Helper method for creating a basic Curve object specifying a turning circle
            :param center:              SE2Transform,  The center of the turning circle (x,y,theta==0)
            :param config_on_circle:    SE2Transform.  Valid configuration on the turning circle
            :param radius:              float > 0.     Radius
            :param curve_type           DubinsSegmentType.LEFT or  DubinsSegmentType.RIGHT  If the car drives a left or right curve

            returns a Curve object with the specified parameters and sets start_config = end_config = point_on_circle, arc_angle = 0"""
        return Curve(center=center, start_config=config_on_circle, end_config=config_on_circle, radius=radius,
                     curve_type=curve_type)


@dataclass
class TurningCircle:
    """ Defines the possible turning circles at the current configuration """
    left: Curve
    right: Curve

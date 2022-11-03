from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from typing import List, Sequence

import numpy as np
from geometry import SE2value

__all__ = [
    "Point",
    "Segment",
    "Circle",
    "Triangle",
    "AABB",
    "Polygon",
    "Path",
    "Pose2D",
]


@dataclass(frozen=True)
class GeoPrimitive(ABC):
    @abstractmethod
    def apply_SE2transform(self, t: SE2value) -> "GeoPrimitive":
        pass


@dataclass(frozen=True)
class Point(GeoPrimitive):
    x: float
    y: float

    def apply_SE2transform(self, t: SE2value) -> "Point":
        p = t @ np.array([self.x, self.y, 1])
        return Point(p[0], p[1])


@dataclass(frozen=True)
class Segment(GeoPrimitive):
    p1: Point
    p2: Point

    def apply_SE2transform(self, t: SE2value) -> "Segment":
        p1 = self.p1.apply_SE2transform(t)
        p2 = self.p2.apply_SE2transform(t)
        return replace(self, p1=p1, p2=p2)


@dataclass(frozen=True)
class Circle(GeoPrimitive):
    center: Point
    radius: float

    def apply_SE2transform(self, t: SE2value) -> "Circle":
        c1 = self.center.apply_SE2transform(t)
        return replace(self, center=c1)


@dataclass(frozen=True)
class Triangle(GeoPrimitive):
    v1: Point
    v2: Point
    v3: Point

    def apply_SE2transform(self, t: SE2value) -> "Triangle":
        points = _transform_points(t, [self.v1, self.v2, self.v3])
        return Triangle(*points)

    def center(self):
        return Point(
            (self.v1.x + self.v2.x + self.v3.x) / 3,
            (self.v1.y + self.v2.y + self.v3.y) / 3,
        )


@dataclass(frozen=True)
class AABB(GeoPrimitive):
    p_min: Point
    p_max: Point

    def apply_SE2transform(self, t: SE2value) -> "AABB":
        points = _transform_points(t, [self.p_min, self.p_max])
        return AABB(*points)


@dataclass(frozen=True)
class Polygon(GeoPrimitive):
    vertices: List[Point]

    def apply_SE2transform(self, t: SE2value) -> "Polygon":
        transformed_vertices = _transform_points(t, self.vertices)
        return replace(self, vertices=transformed_vertices)

    def center(self):
        number_of_vertices = len(self.vertices)
        return Point(
            sum([v.x for v in self.vertices]) / number_of_vertices,
            sum([v.y for v in self.vertices]) / number_of_vertices,
        )


@dataclass(frozen=True)
class Path(GeoPrimitive):
    waypoints: List[Point]

    def apply_SE2transform(self, t: SE2value) -> "Path":
        transformed_waypoints = _transform_points(t, self.waypoints)
        return replace(self, waypoints=transformed_waypoints)

    def __len__(self):
        return len(self.waypoints)


@dataclass(frozen=True)
class Pose2D:
    position: Point
    theta: float


def _transform_points(t: SE2value, points: Sequence[Point]) -> Sequence[Point]:
    return [p.apply_SE2transform(t) for p in points]

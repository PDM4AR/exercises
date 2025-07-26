from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from typing import Any, Sequence

import numpy as np
from geometry import SE2value
import matplotlib.pyplot as plt

__all__ = [
    "Point",
    "Segment",
    "Circle",
    "Triangle",
    "AABB",
    "Polygon",
    "Path",
    "Polygon_3D",
    "Point_3D",
    "Segment_3D",
]


@dataclass(frozen=True)
class GeoPrimitive(ABC):
    @abstractmethod
    def apply_SE2transform(self, t: SE2value) -> "GeoPrimitive":
        pass

    @abstractmethod
    def visualize(self, ax: Any):
        pass

    @abstractmethod
    def get_boundaries(self) -> tuple["Point", "Point"]:
        pass


@dataclass(frozen=True)
class Point(GeoPrimitive):
    x: float
    y: float

    def apply_SE2transform(self, t: SE2value) -> "Point":
        p = t @ np.array([self.x, self.y, 1])
        return Point(p[0], p[1])

    def visualize(self, ax: Any):
        # Draw Point
        ax.plot(self.x, self.y, marker="x", markersize=10)

    def get_boundaries(self) -> tuple["Point", "Point"]:
        return self, self


@dataclass(frozen=True)
class Segment(GeoPrimitive):
    p1: Point
    p2: Point

    def apply_SE2transform(self, t: SE2value) -> "Segment":
        p1 = self.p1.apply_SE2transform(t)
        p2 = self.p2.apply_SE2transform(t)
        return replace(self, p1=p1, p2=p2)

    def visualize(self, ax: Any, colour="r"):
        ax.plot(
            [self.p1.x, self.p2.x],
            [self.p1.y, self.p2.y],
            marker="x",
            markersize=10,
            color=colour,
        )

    def get_boundaries(self) -> tuple["Point", "Point"]:
        p_min = Point(min(self.p1.x, self.p2.x), min(self.p1.y, self.p2.y))
        p_max = Point(max(self.p1.x, self.p2.x), max(self.p1.y, self.p2.y))
        return p_min, p_max


@dataclass(frozen=True)
class Circle(GeoPrimitive):
    center: Point
    radius: float

    def apply_SE2transform(self, t: SE2value) -> "Circle":
        c1 = self.center.apply_SE2transform(t)
        return replace(self, center=c1)

    def visualize(self, ax: Any):
        draw_circle = plt.Circle(
            (self.center.x, self.center.y),
            self.radius,
            color="r",
            fill=False,
            linewidth=5,
        )
        ax.set_aspect(1)
        ax.add_artist(draw_circle)

    def get_boundaries(self) -> tuple["Point", "Point"]:
        return (
            Point(self.center.x - self.radius, self.center.y - self.radius),
            Point(self.center.x + self.radius, self.center.y + self.radius),
        )


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

    def visualize(self, ax: Any):
        draw_triangle = plt.Polygon(
            [[self.v1.x, self.v1.y], [self.v2.x, self.v2.y], [self.v3.x, self.v3.y]],
            color="r",
            fill=False,
            linewidth=5,
        )
        ax.set_aspect(1)
        ax.add_artist(draw_triangle)

    def get_boundaries(self) -> tuple["Point", "Point"]:
        p_min = Point(
            min([self.v1.x, self.v2.x, self.v3.x]),
            min([self.v1.y, self.v2.y, self.v3.y]),
        )
        p_max = Point(
            max([self.v1.x, self.v2.x, self.v3.x]),
            max([self.v1.y, self.v2.y, self.v3.y]),
        )
        return p_min, p_max


@dataclass(frozen=True)
class AABB(GeoPrimitive):
    p_min: Point
    p_max: Point

    def apply_SE2transform(self, t: SE2value) -> "AABB":
        points = _transform_points(t, [self.p_min, self.p_max])
        return AABB(*points)

    def visualize(self, ax: Any):
        raise NotImplementedError()

    def get_boundaries(self) -> tuple["Point", "Point"]:
        raise NotImplementedError()


@dataclass(frozen=True)
class Polygon(GeoPrimitive):
    vertices: list[Point]

    def apply_SE2transform(self, t: SE2value) -> "Polygon":
        transformed_vertices = _transform_points(t, self.vertices)
        return replace(self, vertices=transformed_vertices)

    def center(self):
        number_of_vertices = len(self.vertices)
        return Point(
            sum([v.x for v in self.vertices]) / number_of_vertices,
            sum([v.y for v in self.vertices]) / number_of_vertices,
        )

    def visualize(self, ax: Any):
        draw_poly = plt.Polygon(
            [[p.x, p.y] for p in self.vertices],
            color="r",
            fill=False,
            linewidth=5,
        )
        ax.set_aspect(1)
        ax.add_artist(draw_poly)

    def get_boundaries(self) -> tuple["Point", "Point"]:
        p_min = Point(
            min([v.x for v in self.vertices]),
            min([v.y for v in self.vertices]),
        )
        p_max = Point(
            max([v.x for v in self.vertices]),
            max([v.y for v in self.vertices]),
        )
        return p_min, p_max


@dataclass(frozen=True)
class Path(GeoPrimitive):
    waypoints: list[Point]

    def apply_SE2transform(self, t: SE2value) -> "Path":
        transformed_waypoints = _transform_points(t, self.waypoints)
        return replace(self, waypoints=transformed_waypoints)

    def __len__(self):
        return len(self.waypoints)

    def visualize(self, ax: Any):
        ax.plot(
            [w.x for w in self.waypoints],
            [w.y for w in self.waypoints],
            "gx--",
            markersize=15,
        )

    def get_boundaries(self) -> tuple["Point", "Point"]:
        p_min = Point(
            min([v.x for v in self.waypoints]),
            min([v.y for v in self.waypoints]),
        )
        p_max = Point(
            max([v.x for v in self.waypoints]),
            max([v.y for v in self.waypoints]),
        )
        return p_min, p_max


def _transform_points(t: SE2value, points: Sequence[Point]) -> Sequence[Point]:
    return [p.apply_SE2transform(t) for p in points]


################### 3-D Polygons, Segments and Points:. May be removed.
@dataclass(frozen=True)
class Point_3D(GeoPrimitive):
    x: float
    y: float
    z: float

    def apply_SE2transform(self, t: SE2value) -> "Point":
        p = t @ np.array([self.x, self.y, 1])
        return Point(p[0], p[1])

    def visualize(self, ax: Any):
        # Draw Point
        ax.plot(self.x, self.y, marker="x", markersize=10)

    def get_boundaries(self) -> tuple["Point", "Point"]:
        return self, self


@dataclass(frozen=True)
class Segment_3D(GeoPrimitive):
    p1: Point_3D
    p2: Point_3D
    # TODO: FOR A TA: FINISH OUT THE 3D SEGMENT CLASS.

    def apply_SE2transform(self, t: SE2value) -> "Segment":
        p1 = self.p1.apply_SE2transform(t)
        p2 = self.p2.apply_SE2transform(t)
        return replace(self, p1=p1, p2=p2)

    def visualize(self, ax: Any):
        ax.plot([self.p1.x, self.p2.x], [self.p1.y, self.p2.y], marker="x", markersize=10)

    def get_boundaries(self) -> tuple["Point", "Point"]:
        p_min = Point(min(self.p1.x, self.p2.x), min(self.p1.y, self.p2.y))
        p_max = Point(max(self.p1.x, self.p2.x), max(self.p1.y, self.p2.y))
        return p_min, p_max


@dataclass(frozen=True)
class Polygon_3D(GeoPrimitive):
    vertices: list[Point_3D]
    # TODO: For A TA: FINISH OUT the 3D POLYGON CLASS

    def apply_SE2transform(self, t: SE2value) -> "Polygon":
        transformed_vertices = _transform_points(t, self.vertices)
        return replace(self, vertices=transformed_vertices)

    def center(self):
        number_of_vertices = len(self.vertices)
        return Point_3D(
            sum([v.x for v in self.vertices]) / number_of_vertices,
            sum([v.y for v in self.vertices]) / number_of_vertices,
            sum([v.z for v in self.vertices]) / number_of_vertices,
        )

    def visualize(self, ax: Any):
        draw_poly = plt.Polygon(
            [[p.x, p.y] for p in self.vertices],
            color="r",
            fill=False,
            linewidth=5,
        )
        ax.set_aspect(1)
        ax.add_artist(draw_poly)

    def get_boundaries(self) -> tuple["Point_3D", "Point_3D"]:
        p_min = Point(
            min([v.x for v in self.vertices]),
            min([v.y for v in self.vertices]),
        )
        p_max = Point(
            max([v.x for v in self.vertices]),
            max([v.y for v in self.vertices]),
        )
        return p_min, p_max

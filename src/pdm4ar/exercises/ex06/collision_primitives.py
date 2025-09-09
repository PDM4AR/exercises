from typing import List, Optional, Union

import numpy as np
import triangle as tr
from pdm4ar.exercises_def.ex06.structures import *


class CollisionPrimitives_SeparateAxis:
    """
    Implementation of the Separating Axis Theorem for 2D collision detection.

    The Separating Axis Theorem states that two convex objects do not intersect
    if there exists a line onto which the projections of the objects do not overlap.
    """

    # Task 1
    @staticmethod
    def proj_polygon(p: Union[Polygon, Circle], ax: Segment) -> Segment:
        """
        Project a polygon or circle onto an axis.

        Args:
            p (Union[Polygon, Circle]): Polygon or Circle to project
            ax (Segment): Axis (as a Segment) to project onto

        Returns:
            Segment: Segment representing the projection interval
        """
        start_1 = 0  # placeholder
        end_1 = 0  # placeholder
        start_2 = 0  # placeholder
        end_2 = 0  # placeholder

        # TODO: Implement function
        raise NotImplementedError  # remove when you have written your code
        return Segment(Point(start_1, end_1), Point(start_2, end_2))

    # Task 2.a
    @staticmethod
    def overlap(s1: Segment, s2: Segment) -> bool:
        """
        Check if two segments overlap.

        Args:
            s1 (Segment): First segment
            s2 (Segment): Second segment

        Returns:
            bool: True if segments overlap, False otherwise
        """
        placeholder = True  # placeholder

        # TODO: Implement Function
        raise NotImplementedError  # remove when you have written your code
        return placeholder

    # Task 2.b
    @staticmethod
    def get_axes(p1: Polygon, p2: Polygon) -> list[Segment]:
        """
        Get all candidate separating axes for two polygons.

        For 2D polygons, we only need to check axes orthogonal to the edges.

        Args:
            p1 (Polygon): First polygon
            p2 (Polygon): Second polygon

        Returns:
            list[Segment]: List of segments representing separating axes
        """
        axes = []  # Populate with Segment types

        # TODO: Implement function
        raise NotImplementedError  # remove when you have written your code
        return axes

    @staticmethod
    def separating_axis_thm(
        p1: Polygon,
        p2: Union[Polygon, Circle],
    ) -> tuple[bool, Optional[Segment]]:
        """
        Apply the Separating Axis Theorem for collision detection.

        Tests all candidate axes and checks if projections overlap.
        Handles both polygon-polygon and polygon-circle cases.

        Args:
            p1 (Polygon): First polygon
            p2 (Union[Polygon, Circle]): Second polygon or circle

        Returns:
            tuple[bool, Optional[Segment]]: Tuple of (collision_detected, separating_axis)
                - collision_detected: True if objects collide
                - separating_axis: Optional axis for visualization
        """

        if isinstance(p2, Polygon):  # Task 2c

            # TODO: Implement your solution for if polygon here. Exercise 2
            raise NotImplementedError  # remove when you have written your code
            # return False, axis

        elif isinstance(p2, Circle):  # Task 3b

            # TODO: Implement your solution for SAT for circles here. Exercise 3
            # raise NotImplementedError
            raise NotImplementedError  # remove when you have written your code
            # return False, axis

        else:
            print("If we get here we have done a big mistake.")
            return False, None

    # Task 3a
    @staticmethod
    def get_axes_cp(circ: Circle, poly: Polygon) -> list[Segment]:
        """
        Get candidate separating axes for circle-polygon collision.

        For circles, we need to check:
        - Axes normal to polygon edges
        - Axis from circle center to closest polygon vertex

        Args:
            circ (Circle): Circle primitive
            poly (Polygon): Polygon primitive

        Returns:
            list[Segment]: List of segments representing separating axes
        """
        axes = []

        # TODO: Implement function
        raise NotImplementedError  # remove when you have written your code
        return axes


class CollisionPrimitives:
    """
    Collection of collision detection methods for various primitive pairs.
    """

    NUMBER_OF_SAMPLES = 100

    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        """
        Check collision between circle and point.

        Args:
            c (Circle): Circle primitive
            p (Point): Point primitive

        Returns:
            bool: True if point is inside circle, False otherwise
        """
        return (p.x - c.center.x) ** 2 + (p.y - c.center.y) ** 2 < c.radius**2

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        """
        Check collision between triangle and point using barycentric coordinates.

        Args:
            t (Triangle): Triangle primitive
            p (Point): Point primitive

        Returns:
            bool: True if point is inside triangle, False otherwise
        """
        area_orig = np.abs((t.v2.x - t.v1.x) * (t.v3.y - t.v1.y) - (t.v3.x - t.v1.x) * (t.v2.y - t.v1.y))

        area1 = np.abs((t.v1.x - p.x) * (t.v2.y - p.y) - (t.v2.x - p.x) * (t.v1.y - p.y))
        area2 = np.abs((t.v2.x - p.x) * (t.v3.y - p.y) - (t.v3.x - p.x) * (t.v2.y - p.y))
        area3 = np.abs((t.v3.x - p.x) * (t.v1.y - p.y) - (t.v1.x - p.x) * (t.v3.y - p.y))

        if np.abs(area1 + area2 + area3 - area_orig) < 1e-3:
            return True

        return False

    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        """
        Check collision between polygon and point using triangulation.

        Args:
            poly (Polygon): Polygon primitive
            p (Point): Point primitive

        Returns:
            bool: True if point is inside polygon, False otherwise
        """
        triangulation_result = tr.triangulate(dict(vertices=np.array([[v.x, v.y] for v in poly.vertices])))

        triangles = [
            Triangle(
                Point(triangle[0, 0], triangle[0, 1]),
                Point(triangle[1, 0], triangle[1, 1]),
                Point(triangle[2, 0], triangle[2, 1]),
            )
            for triangle in triangulation_result["vertices"][triangulation_result["triangles"]]
        ]

        for t in triangles:
            if CollisionPrimitives.triangle_point_collision(t, p):
                return True

        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        """
        Check collision between circle and line segment.

        Args:
            c (Circle): Circle primitive
            segment (Segment): Segment primitive

        Returns:
            bool: True if circle intersects with segment, False otherwise
        """
        inside_1 = CollisionPrimitives.circle_point_collision(c, segment.p1)
        inside_2 = CollisionPrimitives.circle_point_collision(c, segment.p2)

        if inside_1 or inside_2:
            return True

        dist_x = segment.p1.x - segment.p2.x
        dist_y = segment.p1.y - segment.p2.y
        segment_len = np.sqrt(dist_x**2 + dist_y**2)

        dot = (
            ((c.center.x - segment.p1.x) * (segment.p2.x - segment.p1.x))
            + ((c.center.y - segment.p1.y) * (segment.p2.y - segment.p1.y))
        ) / pow(segment_len, 2)

        closest_point = Point(
            segment.p1.x + (dot * (segment.p2.x - segment.p1.x)),
            segment.p1.y + (dot * (segment.p2.y - segment.p1.y)),
        )

        # Check whether point is on the segment segment or not
        segment_len_1 = np.sqrt((segment.p1.x - closest_point.x) ** 2 + (segment.p1.y - closest_point.y) ** 2)
        segment_len_2 = np.sqrt((segment.p2.x - closest_point.x) ** 2 + (segment.p2.y - closest_point.y) ** 2)

        if np.abs(segment_len_1 + segment_len_2 - segment_len) > 1e-3:
            return False

        closest_dist = np.sqrt((c.center.x - closest_point.x) ** 2 + (c.center.y - closest_point.y) ** 2)

        if closest_dist < c.radius:
            return True

        return False

    @staticmethod
    def sample_segment(segment: Segment) -> list[Point]:
        """
        Sample points along a segment for collision testing.

        Args:
            segment (Segment): Segment to sample

        Returns:
            list[Point]: List of sampled points along the segment
        """

        x_diff = (segment.p1.x - segment.p2.x) / CollisionPrimitives.NUMBER_OF_SAMPLES
        y_diff = (segment.p1.y - segment.p2.y) / CollisionPrimitives.NUMBER_OF_SAMPLES

        return [
            Point(x_diff * i + segment.p2.x, y_diff * i + segment.p2.y)
            for i in range(CollisionPrimitives.NUMBER_OF_SAMPLES)
        ]

    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        """
        Check collision between triangle and segment using point sampling.

        Args:
            t (Triangle): Triangle primitive
            segment (Segment): Segment primitive

        Returns:
            bool: True if triangle intersects with segment, False otherwise
        """
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:
            if CollisionPrimitives.triangle_point_collision(t, point):
                return True

        return False

    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        """
        Check collision between polygon and segment using point sampling.

        Args:
            p (Polygon): Polygon primitive
            segment (Segment): Segment primitive

        Returns:
            bool: True if polygon intersects with segment, False otherwise
        """
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:
            if CollisionPrimitives.polygon_point_collision(p, point):
                return True

        return False

    @staticmethod
    def polygon_segment_collision_aabb(p: Polygon, segment: Segment) -> bool:
        """
        Check collision using AABB optimization.

        First checks if segment intersects polygon's bounding box,
        then performs detailed collision detection.

        Args:
            p (Polygon): Polygon primitive
            segment (Segment): Segment primitive

        Returns:
            bool: True if polygon intersects with segment, False otherwise
        """
        aabb = CollisionPrimitives._poly_to_aabb(p)
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:

            if aabb.p_min.x > point.x or aabb.p_min.y > point.y:
                continue

            if aabb.p_max.x < point.x or aabb.p_max.y < point.y:
                continue

            if CollisionPrimitives.polygon_point_collision(p, point):
                return True

        return False

    @staticmethod
    def _poly_to_aabb(g: Polygon) -> AABB:
        """
        Convert polygon to axis-aligned bounding box.

        Args:
            g (Polygon): Polygon to convert

        Returns:
            AABB: AABB bounding box for the polygon
        """
        x_values = [v.x for v in g.vertices]
        y_values = [v.y for v in g.vertices]

        return AABB(Point(min(x_values), min(y_values)), Point(max(x_values), max(y_values)))

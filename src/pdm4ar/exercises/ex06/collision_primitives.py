from pdm4ar.exercises_def.ex06.structures import *
from triangle import triangulate
import numpy as np
from typing import Union, Optional


class CollisionPrimitives_SeparateAxis:
    """
    Class for Implementing the Separate Axis Theorem


    A docstring with expected inputs and outputs is provided for each of the functions
    you are to implement. You do not need to adhere to the given variable names, but you should adhere to
    the datatypes of inputs and outputs.


    ## To be added to the website.
    The Separating Axis Theorem states: If two sets are closed and at least one of them is compact, then there is a hyperplane between them,
    and even two parallel hyperplanes separated by a gap. An axis that is orthogonal to a separating hyperplane is deemed a separating axis, because
    the orthogonal projections of the convex bodies onto the axis are disjoint.

    ## THEOREM
    Let A and B be two disjoint nonempty convex subsets of R^n. Then there exist a nonzero vector v anda  real number c s.t.
    <x,v> >= c and <y,v> <= c. For all x in A and y in B. i.e. the hyperplane <.,v> = c separates A and B.

    If both sets are clsoed, and at least one of them is compact, then the separation can be strict, that is,
    <x,v> > c_1 and <y,v> < c_2 for some c_1 > c_2


    In this exercise, we will be implementing the Separating Axis Theorem for 2d Primitives.

    Axis are represented as Segments with length N (world length)

    """

    # Task 1
    @staticmethod
    def proj_polygon(p: Union[Polygon, Circle], ax: Segment) -> Segment:
        """
        Project the Polygon onto the axis, represented as a Segment.
        Inputs:
        Polygon p,
        a candidate axis ax to project onto

        Outputs:
        segment: a (shorter) segment with start and endpoints of where the polygon has been projected to.

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
        Inputs:
        s1: a Segment
        s2: a Segment

        Outputs:
        bool: True if segments overlap. False o.w.
        """
        placeholder = True  # placeholder

        # TODO: Implement Function
        raise NotImplementedError  # remove when you have written your code
        return placeholder

    # Task 2.b
    @staticmethod
    def get_axes(p1: Polygon, p2: Polygon) -> list[Segment]:
        """
        Get all Candidate Separating Axes.
        Hint: These are 2D Polygons, recommend searching over axes that are orthogonal to the edges only.
        Rather than returning infinite Segments, return one axis per Edge1-Edge2 pairing. Return an Axis of size N (worldlength).

        Inputs:
        p1, p2: Polygons to obtain separating Axes over.
        Outputs:
        list[Segment]: A list of segments of size N (worldlength) that represent each of the valid separating axes for the two polygons.
        """
        axes = []  # Populate with Segment types

        # TODO: Implement function
        raise NotImplementedError  # remove when you have written your code
        return axes

    # Task 2.c
    @staticmethod
    def separating_axis_thm(
        p1: Polygon,
        p2: Union[Polygon, Circle],
    ) -> tuple[bool, Optional[Segment]]:
        """
        Get Candidate Separating Axes.
        Once obtained, loop over the Axes, project the polygons onto each acis and check overlap of the projected segments.
        If an axis with a non-overlapping projection is found, we can terminate early. Conclusion: The polygons do not collide.

        IMPORTANT
        This Method Evaluates task 2 and Task 3.
        Task 2 checks the separate axis theorem for two polygons.
        Task 3 checks the separate axis theorem for a circle and a polygon
        We have provided a skeleton on this method to distinguish the two test cases, feel free to use any helper methods above, but your output must come
        from  separating_axis_thm().

        Hint: You can use previously implemented methods, such as overlap() and get_axes()

        Inputs:
        p1, p2: Candidate Polygons
        Outputs:
        Output as a tuple
        bool: True if Polygons dont Collide. False o.w.
        Segment: An Optional argument that allows you to visualize the axis you are projecting onto.
        """

        if isinstance(p2, Polygon):

            # TODO: Implement your solution for if polygon here. Exercise 2
            raise NotImplementedError  # remove when you have written your code
            # return (bool, axis)

        elif isinstance(p2, Circle):

            # TODO: Implement your solution for SAT for circles here. Exercise 3
            # raise NotImplementedError
            raise NotImplementedError  # remove when you have written your code
            # return (bool, axis)

        else:
            print("If we get here we have done a big mistake - TAs")
            return (bool, axis)

    # Task 3
    @staticmethod
    def get_axes_cp(circ: Circle, poly: Polygon) -> list[Segment]:
        """
        Get all Candidate Separating Axes.
        Hint: Notice that the circle is a polygon with infinite number of edges. Fortunately we do not need to check all axes normal to the edges.
        It's sufficient to check the axes normal to the polygon edges plus ONE axis formed by the circle center and the closest vertice of the polygon.

        Inputs:
        circ, poly: Cicle and Polygon to check, respectively.
        Ouputs:
        list[Segment]: A list of segments of size N (worldlength) that represent each of the valid separating axes for the two polygons.
        """
        axes = []

        # TODO
        raise NotImplementedError  # remove when you have written your code
        return axes


class CollisionPrimitives:
    """
    Class of collision primitives
    """

    NUMBER_OF_SAMPLES = 100

    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        """
        Given function.
        Checks if a circle and a point are in collision.

        Inputs:
        c: Circle primitive
        p: Point primitive

        Outputs:
        bool: True if in Collision, False otherwise
        """
        return (p.x - c.center.x) ** 2 + (p.y - c.center.y) ** 2 < c.radius**2

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        """
        Given function.
        Checks if a Triangle and a Point are in Collision

        Inputs:
        t: Triangle Primitive
        p: Point Primitive

        Outputs:
        bool: True if in Collision, False otherwise.
        """
        area_orig = np.abs(
            (t.v2.x - t.v1.x) * (t.v3.y - t.v1.y)
            - (t.v3.x - t.v1.x) * (t.v2.y - t.v1.y)
        )

        area1 = np.abs(
            (t.v1.x - p.x) * (t.v2.y - p.y) - (t.v2.x - p.x) * (t.v1.y - p.y)
        )
        area2 = np.abs(
            (t.v2.x - p.x) * (t.v3.y - p.y) - (t.v3.x - p.x) * (t.v2.y - p.y)
        )
        area3 = np.abs(
            (t.v3.x - p.x) * (t.v1.y - p.y) - (t.v1.x - p.x) * (t.v3.y - p.y)
        )

        if np.abs(area1 + area2 + area3 - area_orig) < 1e-3:
            return True

        return False

    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        """
        Given function.

        Input:
        poly: Polygon primitive
        p: Point primitive

        Outputs
        bool: True if in Collision, False otherwise.
        """
        triangulation_result = tr.triangulate(
            dict(vertices=np.array([[v.x, v.y] for v in poly.vertices]))
        )

        triangles = [
            Triangle(
                Point(triangle[0, 0], triangle[0, 1]),
                Point(triangle[1, 0], triangle[1, 1]),
                Point(triangle[2, 0], triangle[2, 1]),
            )
            for triangle in triangulation_result["vertices"][
                triangulation_result["triangles"]
            ]
        ]

        for t in triangles:
            if CollisionPrimitives.triangle_point_collision(t, p):
                return True

        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        """
        Given function

        Input:
        c: Circle primitive
        segment: Segment primitive

        Outputs:
        bool: True if in collision, False otherwise.
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
        segment_len_1 = np.sqrt(
            (segment.p1.x - closest_point.x) ** 2
            + (segment.p1.y - closest_point.y) ** 2
        )
        segment_len_2 = np.sqrt(
            (segment.p2.x - closest_point.x) ** 2
            + (segment.p2.y - closest_point.y) ** 2
        )

        if np.abs(segment_len_1 + segment_len_2 - segment_len) > 1e-3:
            return False

        closest_dist = np.sqrt(
            (c.center.x - closest_point.x) ** 2 + (c.center.y - closest_point.y) ** 2
        )

        if closest_dist < c.radius:
            return True

        return False

    @staticmethod
    def sample_segment(segment: Segment) -> list[Point]:

        x_diff = (segment.p1.x - segment.p2.x) / CollisionPrimitives.NUMBER_OF_SAMPLES
        y_diff = (segment.p1.y - segment.p2.y) / CollisionPrimitives.NUMBER_OF_SAMPLES

        return [
            Point(x_diff * i + segment.p2.x, y_diff * i + segment.p2.y)
            for i in range(CollisionPrimitives.NUMBER_OF_SAMPLES)
        ]

    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        """
        Given function.

        Input:
        t: Triangle Primitive
        segment: Segment primitive

        Outputs:
        bool: True if in collision, False otherwise.
        """
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:
            if CollisionPrimitives.triangle_point_collision(t, point):
                return True

        return False

    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        """
        Given function.

        Input:
        p: Polygon primitive
        segment: segment primitive

        Outputs:
        bool: True if in collision, False otherwise
        """
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:
            if CollisionPrimitives.polygon_point_collision(p, point):
                return True

        return False

    @staticmethod
    def polygon_segment_collision_aabb(p: Polygon, segment: Segment) -> bool:
        """
        Given Function
        Casts a polygon into an AABB, then determines if the bounding box and a segment are in collision

        Inputs:
        p: Polygon primitive
        segment: Segment primitive

        Outputs:
        bool: True if in collision, False otherwise.
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
        Given Function
        Casts a Polygon type into an AABB

        Inputs:
        g: Polygon

        Outputs:
        AABB: Bounding Box for the Polygon.
        """
        x_values = [v.x for v in g.vertices]
        y_values = [v.y for v in g.vertices]

        return AABB(
            Point(min(x_values), min(y_values)), Point(max(x_values), max(y_values))
        )

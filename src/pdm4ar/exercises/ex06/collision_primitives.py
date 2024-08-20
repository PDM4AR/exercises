from pdm4ar.exercises_def.ex06.structures import *
from triangle import triangulate
from shapely import geometry  # TODO Remove
import numpy as np
from typing import Union, Optional


class CollisionPrimitives_SeparateAxis:
    """
    Class for Implementing the Separate Axis Theorem

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

        # poly = p
        # seg = ax
        # seg_shapely = geometry.LineString([[seg.p1.x, seg.p1.y], [seg.p2.x, seg.p2.y]])
        # min_dist = np.inf
        # min_proj_pt = None
        # max_dist = -np.inf
        # max_proj_pt = None
        # for vertice in poly.vertices:
        #     vertice_shapely = geometry.Point(vertice.x, vertice.y)
        #     dist = seg_shapely.project(vertice_shapely)
        #     if dist < min_dist:
        #         min_dist = dist
        #         min_proj_pt = seg_shapely.interpolate(dist)
        #     if dist > max_dist:
        #         max_dist = dist
        #         max_proj_pt = seg_shapely.interpolate(dist)
        # if min_proj_pt is not None and max_proj_pt is not None:
        #     pt1_proj = Point(x=min_proj_pt.x, y=min_proj_pt.y)
        #     pt2_proj = Point(x=max_proj_pt.x, y=max_proj_pt.y)
        #     proj_seg = Segment(pt1_proj, pt2_proj)

        # return proj_seg
        # TODO
        raise NotImplementedError

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

        # # TODO
        raise NotImplementedError

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

        # # TODO
        raise NotImplementedError

    # Task 2.c
    @staticmethod
    def separating_axis_thm(
        p1: Polygon, p2: Union[Polygon, Circle]
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

        Inputs:
        p1, p2: Candidate Polygons
        Outputs:
        bool: True if Polygons dont Collide. False o.w.
        """

        # if isinstance(p2, Polygon):
        #     poly1_shapely = geometry.Polygon([[p.x, p.y] for p in p1.vertices])
        #     poly2_shapely = geometry.Polygon([[p.x, p.y] for p in p2.vertices])
        #     ans = poly1_shapely.intersects(poly2_shapely)  # sorry students
        #     # TODO: Implement your solution for if polygon here. Exercise 2
        #     # raise NotImplementedError
        #     return (ans, axis)

        # elif isinstance(p2, Circle):

        #     # TODO Implement your solution for SAT for circles here. Exercise 3
        #     # raise NotImplementedError
        #     poly1_shapely = geometry.Polygon([[p.x, p.y] for p in p1.vertices])
        #     circ_shapely = geometry.Point(p2.center.x, p2.center.y).buffer(p2.radius)
        #     ans = poly1_shapely.intersects(circ_shapely)
        #     return (ans, axis)

        # else:
        #     print("If we get here we have done a big mistake - TAs")
        # return (ans, axis)

    # Task 3
    @staticmethod
    def get_axes_cp(circ: Circle, poly: Polygon) -> list[Segment]:
        """
        Get axes between a circle and a polygon.
        Hint: A sufficient condition is to only check the axis that is orthogonal to the polygon edge that is closest to the circle center.

        Inputs:
        circ, poly: Cicle and Polygon to check, respectively.
        Ouputs:
        list[Segment]: A one-elemet list of a separating axis to the closest vertex.
        """

        # # TODO
        raise NotImplementedError


class CollisionPrimitives:
    """
    Class of collusion primitives
    """

    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        return False

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        return False

    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        return False

    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        return False

    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        return False

    @staticmethod
    def polygon_segment_collision_aabb(p: Polygon, segment: Segment) -> bool:
        return False

    @staticmethod
    def _poly_to_aabb(g: Polygon) -> AABB:
        # todo feel free to implement functions that upper-bound a shape with an
        #  AABB or simpler shapes for faster collision checks
        return AABB(p_min=Point(0, 0), p_max=Point(1, 1))

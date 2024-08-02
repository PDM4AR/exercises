from pdm4ar.exercises_def.ex06.structures import *
from triangle import triangulate
from shapely import geometry  # TODO Remove
import numpy as np


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
    def proj_polygon(p: Polygon, ax: Segment) -> Segment:
        """
        Project the Polygon onto the axis, represented as a Segment.
        Inputs:
        Polygon p,
        a candidate axis ax to project onto

        Outputs:
        segment: a (shorter) segment with start and endpoints of where the polygon has been projected to.

        """
        poly = p
        seg = ax
        seg_shapely = geometry.LineString([[seg.p1.x, seg.p1.y], [seg.p2.x, seg.p2.y]])
        min_dist = np.inf
        min_proj_pt = None
        max_dist = -np.inf
        max_proj_pt = None
        for vertice in poly.vertices:
            vertice_shapely = geometry.Point(vertice.x, vertice.y)
            dist = seg_shapely.project(vertice_shapely)
            if dist < min_dist:
                min_dist = dist
                min_proj_pt = seg_shapely.interpolate(dist)
            if dist > max_dist:
                max_dist = dist
                max_proj_pt = seg_shapely.interpolate(dist)
        if min_proj_pt is not None and max_proj_pt is not None:
            pt1_proj = Point(x=min_proj_pt.x, y=min_proj_pt.y)
            pt2_proj = Point(x=max_proj_pt.x, y=max_proj_pt.y)
            proj_seg = Segment(pt1_proj, pt2_proj)

        return proj_seg

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

        # TODO
        raise NotImplementedError

    # Task 2.b
    @staticmethod
    def get_axes(p1: Polygon, p2: Polygon) -> list[Segment]:
        """
        Get all Candidate Separating Axes.
        Hint: These are 2D Polygons, recommend searching over axes that are parallel to the edges only.
        Rather than returning infinite Segments, return one axis per Edge1-Edge2 pairing. Return an Axis of size N (worldlength)

        Inputs:
        p1, p2: Polygons to obtain separating Axes over.
        Outputs:
        list[Segment]: A list of segments of size N (worldlength) that represent each of the valid separating axes for the two polygons.
        """

        # TODO
        raise NotImplementedError

    # Task 2.c
    @staticmethod
    def separating_axis_thm(p1: Polygon, p2: Polygon) -> bool:
        """
        Get Candidate Separating Axes.
        Once obtained, loop over the Axes, project the polygons onto each acis and check overlap of the projected segments.
        If an axis with a non-overlapping projection is found, we can terminate early. Conclusion: The polygons do not collide.

        Inputs:
        p1, p2: Candidate Polygons
        Outputs:
        bool: True if Polygons dont Collide. False o.w.
        """

        poly1_shapely = geometry.Polygon([[p.x, p.y] for p in p1.vertices])
        poly2_shapely = geometry.Polygon([[p.x, p.y] for p in p2.vertices])
        ans = poly1_shapely.intersects(poly2_shapely)  # sorry students

        return ans

    # Task 3
    @staticmethod
    def get_axes_cp(circ: Circle, poly: Polygon) -> list[Segment]:
        """
        Get axes between a circle and a polygon.
        Hint: A sufficient condition is to only check the axis that is formed between the circle center, and the polygon vertex closest to the circle.

        Inputs:
        circ, poly: Cicle and Polygon to check, respectively.
        Ouputs:
        list[Segment]: A one-elemet list of a separating axis to the closest vertex.
        """

        # TODO
        raise NotImplementedError

    # Task 4 (Optional Tasks )
    @staticmethod
    def project_polygon_3d(poly: Polygon_3D, ax: Segment_3D) -> Polygon:
        """
        Project the 3D Polygon to the plane orthogonal to the 3D Axis.
        HINT: Project each vertex onto the plane, then the 2D polygon is formed by the convex hull of all the projected points.
        HINT2: Usage of shapely library is encouraged to get the convex hull.

        Inputs:
        poly: a 3D body (denoted polygon in 3d Space)
        ax: a Segment made of 3d Points. (Or a hyperplane? TODO: Verify with Yueshan.)
        Outputs:
        Polygon: a 2D polygon that represents the projection of the 3D Polygon.
        """

        # TODO
        raise NotImplementedError

    @staticmethod
    def separate_axis_thm_simplified_3d(p1: Polygon_3D, p2: Polygon_3D) -> bool:
        """
        Project the 3D polygons onto the z-axis and check if the projected polygons overlap, using the previously implemented Separating Axis Thm method.
        Note: This is a conservative check because a separating plane may still exist. For a precise collision check,
        we need to test all planes formed by the corss products between any of the two edges of the polygons.

        Inputs:
        p1, p2: 3D polygons (bodies)
        Outputs:
        bool: true if SAT finds no collisions , false o.w.
        """

        # TODO
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

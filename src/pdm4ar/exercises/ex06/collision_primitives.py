from pdm4ar.exercises_def.ex06.structures import *
from pdm4ar.exercises_def.ex06.structures import GeoPrimitive


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
    def circle_line_collision(c: Circle, segment: Segment) -> bool:
        return False

    @staticmethod
    def polygon_line_collision(p: Polygon, segment: Segment) -> bool:
        return False

    @staticmethod
    def polygon_line_collision_aabb(p: Polygon, segment: Segment) -> bool:
        return False

    @staticmethod
    def _poly_to_aabb(g: Polygon) -> AABB:
        # todo feel free to implement functions that upper-bound a shape with an
        #  AABB or simpler shapes for faster collision checks
        return AABB(p_min=Point(0, 0), p_max=Point(1, 1))

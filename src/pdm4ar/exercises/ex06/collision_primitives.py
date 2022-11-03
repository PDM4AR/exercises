from pdm4ar.exercises_def.ex06.structures import *


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
    def covert_polygon_to_aabb(p: Polygon) -> AABB:
        return AABB(p_min=Point(0, 0), p_max=Point(1, 1))

    @staticmethod
    def polygon_line_collision_aabb(p: Polygon, segment: Segment) -> bool:
        return False

    @staticmethod
    def polygon_line_collision(p: Polygon, segment: Segment) -> bool:
        return False

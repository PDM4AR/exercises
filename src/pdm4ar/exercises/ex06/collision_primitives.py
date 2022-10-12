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
    def circle_line_collision(c: Circle, line: Line) -> bool:
        return False

    @staticmethod
    def polygon_line_collision(p: Polygon, line: Line) -> bool:
        return False

    @staticmethod
    def polygon_line_collision_aabb(p: Polygon, line: Line) -> bool:
        return False

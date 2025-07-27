from typing import List

import shapely
from dg_commons import SE2Transform
from pdm4ar.exercises.ex06.collision_primitives import (
    CollisionPrimitives,
    CollisionPrimitives_SeparateAxis,
)
from pdm4ar.exercises_def.ex06.structures import (
    Circle,
    GeoPrimitive,
    Path,
    Point,
    Polygon,
    Segment,
    Triangle,
)

##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

COLLISION_PRIMITIVES = {
    Point: {
        Circle: lambda x, y: CollisionPrimitives.circle_point_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_point_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_point_collision(y, x),
    },
    Segment: {
        Circle: lambda x, y: CollisionPrimitives.circle_segment_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_segment_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_segment_collision_aabb(y, x),
    },
    Triangle: {
        Point: CollisionPrimitives.triangle_point_collision,
        Segment: CollisionPrimitives.triangle_segment_collision,
    },
    Circle: {
        Point: CollisionPrimitives.circle_point_collision,
        Segment: CollisionPrimitives.circle_segment_collision,
    },
    Polygon: {
        Point: CollisionPrimitives.polygon_point_collision,
        Segment: CollisionPrimitives.polygon_segment_collision_aabb,
    },
}


def check_collision(p_1: GeoPrimitive, p_2: GeoPrimitive) -> bool:
    """
    Check collision between two geometric primitives.

    Uses the collision detection functions from CollisionPrimitives class.

    Args:
        p_1: First geometric primitive
        p_2: Second geometric primitive

    Returns:
        True if primitives collide, False otherwise
    """
    assert type(p_1) in COLLISION_PRIMITIVES, "Collision primitive does not exist."
    assert type(p_2) in COLLISION_PRIMITIVES[type(p_1)], "Collision primitive does not exist."

    collision_func = COLLISION_PRIMITIVES[type(p_1)][type(p_2)]

    return collision_func(p_1, p_2)


##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not


def geo_primitive_to_shapely(p: GeoPrimitive):
    """
    Convert geometric primitive to Shapely object.

    Args:
        p: Geometric primitive to convert

    Returns:
        Corresponding Shapely geometry object
    """
    if isinstance(p, Point):
        return shapely.Point(p.x, p.y)
    elif isinstance(p, Segment):
        return shapely.LineString([[p.p1.x, p.p1.y], [p.p2.x, p.p2.y]])
    elif isinstance(p, Circle):
        return shapely.Point(p.center.x, p.center.y).buffer(p.radius)
    elif isinstance(p, Triangle):
        return shapely.Polygon([[p.v1.x, p.v1.y], [p.v2.x, p.v2.y], [p.v3.x, p.v3.y]])
    else:  # Polygon
        vertices = []
        for vertex in p.vertices:
            vertices += [(vertex.x, vertex.y)]
        return shapely.Polygon(vertices)


class CollisionChecker:
    """
    Collision detection for a circular differential drive robot.

    Implements various collision checking methods including basic collision detection,
    occupancy grids, R-trees, and safety certificates.
    """

    def __init__(self):
        pass

    def path_collision_check(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Check path for collisions using basic collision detection.

        Args:
            t: Robot path with waypoints
            r: Robot radius
            obstacles: List of obstacle primitives (Triangle, Circle, Polygon)

        Returns:
            List of indices of collided line segments (0-indexed)
        """
        return []

    def path_collision_check_occupancy_grid(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Check path for collisions using occupancy grid method.

        Generates an occupancy grid representation of the map for efficient
        collision checking.

        Args:
            t: Robot path with waypoints
            r: Robot radius
            obstacles: List of obstacle primitives (Triangle, Circle, Polygon)

        Returns:
            List of indices of collided line segments (0-indexed)
        """
        return []

    def path_collision_check_r_tree(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Check path for collisions using R-tree spatial indexing.

        Builds an R-tree of obstacles for efficient spatial queries.
        Can use custom implementation or Shapely's STRTree.

        Args:
            t: Robot path with waypoints
            r: Robot radius
            obstacles: List of obstacle primitives (Triangle, Circle, Polygon)

        Returns:
            List of indices of collided line segments (0-indexed)
        """
        return []

    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: SE2Transform,
        next_pose: SE2Transform,
        observed_obstacles: list[GeoPrimitive],
    ) -> bool:
        """
        Check for collisions during robot movement in robot frame.

        Args:
            r: Robot radius
            current_pose: Current robot pose
            next_pose: Target robot pose
            observed_obstacles: Obstacles in robot coordinate frame

        Returns:
            True if collision detected during movement, False otherwise
        """
        return False

    def path_collision_check_safety_certificate(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Check path for collisions using safety certificates method.

        Implements the safety certificates procedure as described in:
        https://journals.sagepub.com/doi/full/10.1177/0278364915625345

        Args:
            t: Robot path with waypoints
            r: Robot radius
            obstacles: List of obstacle primitives (Triangle, Circle, Polygon)

        Returns:
            List of indices of collided line segments (0-indexed)
        """
        return []

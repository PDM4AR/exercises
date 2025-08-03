from typing import List, Sequence, Tuple

import numpy as np
from dg_commons import SE2Transform
from pdm4ar.exercises.ex06.collision_primitives import CollisionPrimitives
from pdm4ar.exercises_def.ex06.structures import (
    AABB,
    Capsule,
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

    This function uses the collision detection methods implemented in the CollisionPrimitives class
    to determine if two geometric shapes intersect or overlap.

    Args:
        p_1 (GeoPrimitive): First geometric primitive
        p_2 (GeoPrimitive): Second geometric primitive

    Returns:
        bool: True if the primitives collide, False otherwise

    Raises:
        AssertionError: If collision primitive types are not supported
    """
    assert type(p_1) in COLLISION_PRIMITIVES, "Collision primitive does not exist."
    assert type(p_2) in COLLISION_PRIMITIVES[type(p_1)], "Collision primitive does not exist."

    collision_func = COLLISION_PRIMITIVES[type(p_1)][type(p_2)]

    return collision_func(p_1, p_2)


class CollisionChecker:
    """
    Collision detection system for a circular differential drive robot.

    This class provides various collision checking methods including basic collision detection,
    occupancy grid-based checking, R-tree spatial indexing, and optimization-based collision detection.
    It handles path collision checking for circular robots moving through environments with
    geometric obstacles (triangles, circles, and polygons).
    """

    @staticmethod
    def path_collision_check(t: Path, r: float, obstacles: List[GeoPrimitive]) -> List[int]:
        """
        Check for collisions along a robot path using basic collision detection.

        Args:
            t (Path): Robot path in waypoints
            r (float): Robot radius
            obstacles (List[GeoPrimitive]): List of obstacles (Triangle, Circle, Polygon only)

        Returns:
            List[int]: Indices of colliding path segments (0-indexed, where 0 is first segment)
        """

        return []

    @staticmethod
    def path_collision_check_occupancy_grid(t: Path, r: float, obstacles: List[GeoPrimitive]) -> List[int]:
        """
        Check path collisions using occupancy grid representation.

        Converts the environment to a discrete grid and samples path segments to check for collisions.

        Args:
            t (Path): Robot path in waypoints
            r (float): Robot radius
            obstacles (List[GeoPrimitive]): List of obstacles (Triangle, Circle, Polygon only)

        Returns:
            List[int]: Indices of colliding path segments
        """

        return []

    @staticmethod
    def path_collision_check_r_tree(t: Path, r: float, obstacles: List[GeoPrimitive]) -> List[int]:
        """
        Check path collisions using R-tree spatial indexing for efficiency.

        Builds an R-tree data structure for fast spatial queries of obstacles,
        then checks path segments against nearby obstacles only.

        You are free to implement your own R-Tree or you could use STRTree of shapely module.

        Args:
            t (Path): Robot path in waypoints
            r (float): Robot radius
            obstacles (List[GeoPrimitive]): List of obstacles (Triangle, Circle, Polygon only)

        Returns:
            List[int]: Indices of colliding path segments

        """

        return []

    @staticmethod
    def collision_check_robot_frame(
        r: float,
        current_pose: SE2Transform,
        next_pose: SE2Transform,
        observed_obstacles: List[GeoPrimitive],
    ) -> bool:
        """
        Check collision during robot movement between two poses.

        Verifies if a circular robot can move safely from current pose to next pose
        given obstacles observed in the robot's local frame.

        Args:
            r (float): Robot radius
            current_pose (SE2Transform): Current robot pose in world frame
            next_pose (SE2Transform): Target robot pose in world frame
            observed_obstacles (List[GeoPrimitive]): Obstacles in robot's local frame

        Returns:
            bool: True if collision detected during movement, False if path is clear
        """
        return False

    @staticmethod
    def path_collision_check_opt(t: Path, r: float, obstacles: List[GeoPrimitive]) -> List[int]:
        """
        Check path collisions using optimization-based collision detection.

        Implements the DCOL [1] (Differentiable Collision Detection) framework that formulates
        collision detection as a convex optimization problem. This method solves for the
        minimum uniform scaling applied to each primitive before they intersect, providing
        a fully differentiable collision detection metric.

        You can use the code structure of `OptCollisionCheckingPrimitives` class and call `OptCollisionCheckingPrimitives.check_collision`
        in this method. Or you can implement your own DCOL-based collision checking algorithm.

        We will only call this method during exercise evaluation.

        Args:
            t (Path): Robot path containing waypoints to check
            r (float): Robot radius for collision checking
            obstacles (List[GeoPrimitive]): List of obstacles (Triangle, Circle, Polygon only)

        Returns:
            List[int]: Indices of colliding path segments (0-indexed)

        References:
            [1] https://arxiv.org/abs/2207.00669 - DCOL: Differentiable Collision Detection
        """

        return []

from pdm4ar.exercises_def.ex06.structures import GeoPrimitive, Pose2D, Path
from pdm4ar.exercises_def.ex06.ex06 import check_collision
from typing import List


class CollisionChecker:
    """
    This class implements the collision check ability of a simple planner for a circular differential drive robot.

    Note that check_collision could be used to check collision between given GeoPrimitives
    check_collision function uses the functions that you implemted in CollisionPrimitives class.
    """

    def __init__(self):
        pass

    def path_collision_check(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        return []

    def path_collision_check_occupancy_grid(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will generate an occupancy grid of the given map.
        Then, occupancy grid will be used to check collisions.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        return []

    def path_collision_check_r_tree(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will build an R-Tree of the given obstacles.
        You are free to implement your own R-Tree or you could use STRTree of shapely module.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        return []

    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: Pose2D,
        next_pose: Pose2D,
        observed_obstacles: List[GeoPrimitive],
    ) -> bool:
        """
        Returns there exists a collision or not during the movement of a circular differential dirve robot until its next pose.

            Parameters:
                    r (float): Radius of circular differential drive robot
                    current_pose (Pose2D): Current pose of the circular differential drive robot
                    next_pose (Pose2D): Next pose of the circular differential drive robot
                    observed_obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives in robot frame
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        return False

    def path_collision_check_safety_certificate(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will implement the safety certificates procedure for collision checking.
        You are free to use shapely to calculate distance between a point anf GoePrimitive.
        For detailed information regarding to algorithm,
        Pplease check the Algorithm 1 inside the following paper https://journals.sagepub.com/doi/full/10.1177/0278364915625345.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        return []

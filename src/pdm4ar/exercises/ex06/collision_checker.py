from pdm4ar.exercises_def.ex06.data import *


class CollisionChecker:
    """
    This class implements the collision check ability of a simple planner.
    """

    def __init__(self):
        pass

    @staticmethod
    def extend_obstacle(r: float, obstacle: Polygon) -> Polygon:
        return Polygon([])

    def path_collision_check(
        self, t: Path, r: float, obstacles: List[Polygon]
    ) -> List[int]:
        return []

    def path_collision_check_occupancy_grid(
        self, t: Path, r: float, obstacles: List[Polygon]
    ) -> List[int]:
        return []

    def path_collision_check_r_tree(
        self, t: Path, r: float, obstacles: List[Polygon]
    ) -> List[int]:
        return []

    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: Pose2D,
        next_position: Point,
        observed_obstacles: List[Polygon],
    ) -> bool:
        return False

    def path_collision_check_safety_certificate(
        self, t: Path, r: float, obstacles: List[Polygon]
    ) -> List[int]:
        return []

import timeit
import numpy as np
import random
from typing import Any, Callable, Sequence, Tuple, List
from dataclasses import dataclass
from reprep import Report

from pdm4ar.exercises.ex06.collision_checker import (
    CollisionChecker,
)
from pdm4ar.exercises.ex06.collision_primitives import (
    CollisionPrimitives,
)
from pdm4ar.exercises_def.ex06 import Pose2D, Polygon
from pdm4ar.exercises_def.ex06.visualization import (
    visualize_circle_point,
    visualize_triangle_point,
    visualize_polygon_point,
    visualize_circle_line,
    visualize_polygon_line,
    visualize_map_path,
    visualize_robot_frame_map,
)
from pdm4ar.exercises_def.structures import Exercise, ExIn, PerformanceResults
from pdm4ar.exercises_def.ex06.data import DataGenerator

RANDOM_SEED = 0

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)

@dataclass
class TestCollisionCheck(ExIn):
    number_of_test_cases: int
    step_id: int
    name: str
    sample_generator: Callable
    visualizer: Callable
    ex_function: Callable
    eval_function: Callable
    eval_weight: float

    def str_id(self) -> str:
        return f"step-{self.step_id}-"

@dataclass(frozen=True)
class CollisionCheckWeightedAccuracy(PerformanceResults):
    accuracy: float

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1

@dataclass(frozen=True)
class CollisionCheckPerformance(PerformanceResults):
    accuracy: float
    weight: float
    """Percentage of correct comparisons"""

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1

    @staticmethod
    def perf_aggregator(eval_list: Sequence['CollisionCheckPerformance'], total_weight: float) -> 'CollisionCheckWeightedAccuracy':

        if len(eval_list) == 0:
            return CollisionCheckWeightedAccuracy(0.0)

        total_acccuracy = np.sum([eval.accuracy * eval.weight for eval in eval_list])

        return CollisionCheckWeightedAccuracy(total_acccuracy/total_weight)

def _collision_check_rep(
    algo_in: TestCollisionCheck, alg_out: Any
) -> Tuple[CollisionCheckPerformance, Report]:

    r = Report(algo_in.name)

    eval_list = []

    for ex_num in range(algo_in.number_of_test_cases):
        data = algo_in.sample_generator(ex_num)
        algo_in.visualizer(r, f"step-{algo_in.step_id}-{ex_num}", data)
        start = timeit.default_timer()
        estimate = algo_in.ex_function(*data[:-1])
        stop = timeit.default_timer()
        eval_list.append(algo_in.eval_function(data, estimate))
        r.text(
            f"{algo_in.str_id()}-{ex_num}",
            f"Ground Truth = {data[-1]} | Estimation = {estimate} | Execution Time = {round(stop - start, 5)}",
        )

    r.text(
        f"{algo_in.str_id()}-results",
        "\n".join([f"Accuracy #{ex_num}: {ex_perf}" for ex_num, ex_perf in enumerate(eval_list)] + [f"Total Accuracy = {np.mean(eval_list)}"])
    )

    return CollisionCheckPerformance(np.mean(eval_list), algo_in.eval_weight), r


def algo_placeholder(ex_in):
    return None

def float_eval_function(data, estimation):
    return float(data[-1] == estimation)

def idx_list_eval_function(data, estimation):
    path_len = len(data[0])
    ground_truth_bool = np.array([i in data[-1] for i in range(path_len-1)])
    estimation_bool = np.array([i in estimation for i in range(path_len-1)])

    return (ground_truth_bool == estimation_bool).mean()

def collision_check_robot_frame_loop(
    poses: List[Pose2D],
    r: float,
    observed_obstacles_list: List[List[Polygon]],
    map: List[Polygon],
) -> List[int]:
    # Initialize Collision Checker
    collision_checker = CollisionChecker()
    # Iterate Over Path
    result = []
    for i, (pose, next_pose, observed_obstacles) in enumerate(
        zip(poses[:-1], poses[1:], observed_obstacles_list)
    ):
        if collision_checker.collision_check_robot_frame(
            r, pose, next_pose.position, observed_obstacles
        ):
            result.append(i)
    return result

def get_exercise6() -> Exercise:
    # Set Random Seed
    set_random_seed(RANDOM_SEED)

    # Generate Test Data
    test_values = [
        TestCollisionCheck(
            10,
            1,
            "Point-Circle Collision Primitive",
            DataGenerator.generate_circle_point_collision_data,
            visualize_circle_point,
            CollisionPrimitives.circle_point_collision,
            float_eval_function,
            5
        ),  # Step 1
        TestCollisionCheck(
            10,
            2,
            "Point-Triangle Collision Primitive",
            DataGenerator.generate_triangle_point_collision_data,
            visualize_triangle_point,
            CollisionPrimitives.triangle_point_collision,
            float_eval_function,
            10
        ),  # Step 2
        TestCollisionCheck(
            10,
            3,
            "Point-Polygon Collision Primitive",
            DataGenerator.generate_polygon_point_collision_data,
            visualize_polygon_point,
            CollisionPrimitives.polygon_point_collision,
            float_eval_function,
            10
        ),  # Step 3
        TestCollisionCheck(
            10,
            4,
            "Line-Circle Collision Primitive",
            DataGenerator.generate_circle_line_collision_data,
            visualize_circle_line,
            CollisionPrimitives.circle_line_collision,
            float_eval_function,
            10
        ),  # Step 4
        TestCollisionCheck(
            10,
            5,
            "Line-Polygon Collision Primitive",
            DataGenerator.generate_polygon_line_collision_data,
            visualize_polygon_line,
            CollisionPrimitives.polygon_line_collision,
            float_eval_function,
            5
        ),  # Step 5
        TestCollisionCheck(
            10,
            6,
            "Line-Polygon (AABB) Collision Primitive",
            DataGenerator.generate_polygon_line_collision_data,
            visualize_polygon_line,
            CollisionPrimitives.polygon_line_collision_aabb,
            float_eval_function,
            5
        ),  # Step 6
        TestCollisionCheck(
            5,
            7,
            "Path Collision Check",
            DataGenerator().generate_random_robot_map_and_path,
            visualize_map_path,
            CollisionChecker().path_collision_check,
            idx_list_eval_function,
            20
        ),  # Step 7
        TestCollisionCheck(
            5,
            8,
            "Path Collision Check - Occupancy Grid",
            DataGenerator().generate_random_robot_map_and_path,
            visualize_map_path,
            CollisionChecker().path_collision_check_occupancy_grid,
            idx_list_eval_function,
            20
        ),  # Step 8
        TestCollisionCheck(
            5,
            9,
            "Path Collision Check - R-Tree",
            DataGenerator().generate_random_robot_map_and_path,
            visualize_map_path,
            CollisionChecker().path_collision_check_r_tree,
            idx_list_eval_function,
            30
        ),  # Step 9
        TestCollisionCheck(
            5,
            10,
            "Collision Check - Rigid Body Transformation",
            DataGenerator().generate_robot_frame_data,
            visualize_robot_frame_map,
            collision_check_robot_frame_loop,
            idx_list_eval_function,
            20
        ),  # Step 10
        TestCollisionCheck(
            5,
            11,
            "Path Collision Check - Safety Certificates",
            DataGenerator().generate_random_robot_map_and_path,
            visualize_map_path,
            CollisionChecker().path_collision_check_safety_certificate,
            idx_list_eval_function,
            30
        ),  # Step 11
    ]
    
    total_weight = np.sum([t.eval_weight for t in test_values])

    return Exercise[TestCollisionCheck, Any](
        desc = "This exercise is about the collision checking methods.",
        evaluation_fun = _collision_check_rep,
        perf_aggregator = lambda x: CollisionCheckPerformance.perf_aggregator(x, total_weight),
        test_values = test_values,
        expected_results = None
    )

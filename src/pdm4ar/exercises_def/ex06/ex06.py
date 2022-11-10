import timeit
import numpy as np
import random
from typing import Any, Callable, Sequence, Tuple, List
from dataclasses import dataclass

from dg_commons import SE2Transform
from reprep import Report

from pdm4ar.exercises.ex06.collision_checker import (
    CollisionChecker,
)
from pdm4ar.exercises.ex06.collision_primitives import (
    CollisionPrimitives,
)
from pdm4ar.exercises_def.ex06.structures import Polygon
from pdm4ar.exercises_def.ex06.visualization import (
    visualize_circle_point,
    visualize_triangle_point,
    visualize_polygon_point,
    visualize_circle_line,
    visualize_triangle_line,
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
    eval_weights: Tuple[float, float]

    def str_id(self) -> str:
        return f"step-{self.step_id}-"


@dataclass(frozen=True)
class CollisionCheckWeightedAccuracy(PerformanceResults):
    accuracy: float
    solve_time: float

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1


@dataclass(frozen=True)
class CollisionCheckPerformance(PerformanceResults):
    accuracy: float
    solve_time: float
    weights: Tuple[float, float]
    """Percentage of correct comparisons"""

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1

    @staticmethod
    def perf_aggregator(
        eval_list: Sequence["CollisionCheckPerformance"],
        total_weights: Tuple[float, float],
    ) -> "CollisionCheckWeightedAccuracy":

        if len(eval_list) == 0:
            return CollisionCheckWeightedAccuracy(0.0, np.inf)

        total_acccuracy = np.sum(
            [eval.accuracy * eval.weights[0] for eval in eval_list]
        )
        total_solve_time = np.sum(
            [eval.solve_time * eval.weights[1] for eval in eval_list]
        )
        return CollisionCheckWeightedAccuracy(
            total_acccuracy / total_weights[0], total_solve_time / total_weights[1]
        )


def _collision_check_rep(
    algo_in: TestCollisionCheck, alg_out: Any
) -> Tuple[CollisionCheckPerformance, Report]:

    r = Report(algo_in.name)

    accuracy_list = []
    solve_times = []

    for ex_num in range(algo_in.number_of_test_cases):
        data = algo_in.sample_generator(ex_num)
        algo_in.visualizer(r, f"step-{algo_in.step_id}-{ex_num}", data)
        start = timeit.default_timer()
        estimate = algo_in.ex_function(*data[:-1])
        stop = timeit.default_timer()
        accuracy_list.append(algo_in.eval_function(data, estimate))
        solve_times.append(stop - start)
        r.text(
            f"{algo_in.str_id()}-{ex_num}",
            f"Ground Truth = {data[-1]} | Estimation = {estimate} | Execution Time = {round(stop - start, 5)}",
        )

    r.text(
        f"{algo_in.str_id()}-results",
        "\n".join(
            [
                f"Accuracy #{ex_num}: {ex_perf}"
                for ex_num, ex_perf in enumerate(accuracy_list)
            ]
            + [f"Total Accuracy = {np.mean(accuracy_list)}"]
            + [f"Average Solving Time = {np.mean(solve_times)}"]
        ),
    )

    return (
        CollisionCheckPerformance(
            np.mean(accuracy_list), np.mean(solve_times), algo_in.eval_weights
        ),
        r,
    )


def algo_placeholder(ex_in):
    return None


def float_eval_function(data, estimation):
    return float(data[-1] == estimation)


def idx_list_eval_function(data, estimation):
    path_len = len(data[0])
    ground_truth_bool = np.array([i in data[-1] for i in range(path_len - 1)])
    estimation_bool = np.array([i in estimation for i in range(path_len - 1)])

    return (ground_truth_bool == estimation_bool).mean()


def collision_check_robot_frame_loop(
    poses: List[SE2Transform],
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
            r, pose, next_pose, observed_obstacles
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
            (5, 0),
        ),  # Step 1
        TestCollisionCheck(
            10,
            2,
            "Point-Triangle Collision Primitive",
            DataGenerator.generate_triangle_point_collision_data,
            visualize_triangle_point,
            CollisionPrimitives.triangle_point_collision,
            float_eval_function,
            (10, 0),
        ),  # Step 2
        TestCollisionCheck(
            10,
            3,
            "Point-Polygon Collision Primitive",
            DataGenerator.generate_polygon_point_collision_data,
            visualize_polygon_point,
            CollisionPrimitives.polygon_point_collision,
            float_eval_function,
            (10, 0),
        ),  # Step 3
        TestCollisionCheck(
            10,
            4,
            "Segment-Circle Collision Primitive",
            DataGenerator.generate_circle_segment_collision_data,
            visualize_circle_line,
            CollisionPrimitives.circle_segment_collision,
            float_eval_function,
            (10, 0),
        ),  # Step 4
        TestCollisionCheck(
            10,
            5,
            "Segment-Triangle Collision Primitive",
            DataGenerator.generate_tringle_segment_collision_data,
            visualize_triangle_line,
            CollisionPrimitives.triangle_segment_collision,
            float_eval_function,
            (10, 0),
        ),  # Step 5
        TestCollisionCheck(
            10,
            6,
            "Segment-Polygon Collision Primitive",
            DataGenerator.generate_polygon_segment_collision_data,
            visualize_polygon_line,
            CollisionPrimitives.polygon_segment_collision,
            float_eval_function,
            (5, 0),
        ),  # Step 6
        TestCollisionCheck(
            10,
            7,
            "Segment-Polygon (AABB) Collision Primitive",
            DataGenerator.generate_polygon_segment_collision_data,
            visualize_polygon_line,
            CollisionPrimitives.polygon_segment_collision_aabb,
            float_eval_function,
            (5, 0),
        ),  # Step 7
        TestCollisionCheck(
            5,
            8,
            "Path Collision Check",
            lambda x: DataGenerator().generate_random_robot_map_and_path(8, x),
            visualize_map_path,
            CollisionChecker().path_collision_check,
            idx_list_eval_function,
            (20, 20),
        ),  # Step 8
        TestCollisionCheck(
            5,
            9,
            "Path Collision Check - Occupancy Grid",
            lambda x: DataGenerator().generate_random_robot_map_and_path(9, x),
            visualize_map_path,
            CollisionChecker().path_collision_check_occupancy_grid,
            idx_list_eval_function,
            (20, 20),
        ),  # Step 9
        TestCollisionCheck(
            5,
            10,
            "Path Collision Check - R-Tree",
            lambda x: DataGenerator().generate_random_robot_map_and_path(10, x),
            visualize_map_path,
            CollisionChecker().path_collision_check_r_tree,
            idx_list_eval_function,
            (30, 30),
        ),  # Step 10
        TestCollisionCheck(
            5,
            11,
            "Collision Check - Rigid Body Transformation",
            DataGenerator().generate_robot_frame_data,
            visualize_robot_frame_map,
            collision_check_robot_frame_loop,
            idx_list_eval_function,
            (20, 20),
        ),  # Step 11
        TestCollisionCheck(
            5,
            12,
            "Path Collision Check - Safety Certificates",
            lambda x: DataGenerator().generate_random_robot_map_and_path(12, x),
            visualize_map_path,
            CollisionChecker().path_collision_check_safety_certificate,
            idx_list_eval_function,
            (30, 30),
        ),  # Step 12
    ]

    total_weights = (
        np.sum([t.eval_weights[0] for t in test_values]),
        np.sum([t.eval_weights[1] for t in test_values]),
    )

    return Exercise[TestCollisionCheck, Any](
        desc="This exercise is about the collision checking methods.",
        evaluation_fun=_collision_check_rep,
        perf_aggregator=lambda x: CollisionCheckPerformance.perf_aggregator(
            x, total_weights
        ),
        test_values=test_values,
        expected_results=None,
    )

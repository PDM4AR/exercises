import timeit
import numpy as np
import random
from typing import Any, Callable, Sequence
from shapely.geometry import LineString
from dataclasses import dataclass

from dg_commons import SE2Transform
from reprep import Report

from pdm4ar.exercises.ex06.collision_checker import (
    CollisionChecker,
)
from pdm4ar.exercises.ex06.collision_primitives import (
    CollisionPrimitives,
    CollisionPrimitives_SeparateAxis,
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
    visualize_axis_poly,
    viusalize_SAT_poly,
    viusalize_SAT_poly_circle,
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
    eval_weights: tuple[float, float]

    def str_id(self) -> str:
        return f"step-{self.step_id}-"


@dataclass(frozen=True)
class CollisionCheckWeightedPerformance(PerformanceResults):
    accuracy: float
    solve_time: float
    performances: dict[int, dict[str, float]]

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1
        assert 0 < self.solve_time


@dataclass(frozen=True)
class CollisionCheckPerformance(PerformanceResults):
    accuracy: float
    solve_time: float
    weights: tuple[float, float]
    step_id: int
    """Percentage of correct comparisons"""

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1

    @staticmethod
    def perf_aggregator(
        eval_list: Sequence["CollisionCheckPerformance"],
        total_weights: tuple[float, float],
    ) -> "CollisionCheckWeightedPerformance":

        if len(eval_list) == 0:
            return CollisionCheckWeightedPerformance(0.0, np.inf, {})

        total_acccuracy = np.sum(
            [eval.accuracy * eval.weights[0] for eval in eval_list]
        )
        total_solve_time = np.sum(
            [eval.solve_time * eval.weights[1] for eval in eval_list]
        )
        performances = {
            eval.step_id: {"accuracy": eval.accuracy, "solve_time": eval.solve_time}
            for eval in eval_list
        }

        return CollisionCheckWeightedPerformance(
            total_acccuracy / total_weights[0],
            total_solve_time / total_weights[1],
            performances,
        )


def _collision_check_rep(
    algo_in: TestCollisionCheck, alg_out: Any
) -> tuple[CollisionCheckPerformance, Report]:

    # Set Random Seed
    set_random_seed(RANDOM_SEED)

    r = Report(algo_in.name)

    accuracy_list = []
    solve_times = []

    for ex_num in range(algo_in.number_of_test_cases):
        data = algo_in.sample_generator(ex_num)
        start = timeit.default_timer()
        estimate = algo_in.ex_function(*data[:-1])
        stop = timeit.default_timer()
        solve_times.append(stop - start)

        # get size of the estimate:
        if isinstance(estimate, tuple):
            # print("Estimate is a tuple!")
            accuracy_list.append(algo_in.eval_function(data, estimate[0]))
            try:
                algo_in.visualizer(
                    r, f"step-{algo_in.step_id}-{ex_num}", data, estimate[1]
                )
            except:
                algo_in.visualizer(r, f"step-{algo_in.step_id}-{ex_num}", data)
            r.text(
                f"{algo_in.str_id()}-{ex_num}",
                f"Ground Truth = {data[-1]} | Estimation = {estimate[0]} | Execution Time = {round(stop - start, 5)}",
            )

        else:

            accuracy_list.append(algo_in.eval_function(data, estimate))
            algo_in.visualizer(r, f"step-{algo_in.step_id}-{ex_num}", data)

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
            np.mean(accuracy_list),
            np.mean(solve_times),
            algo_in.eval_weights,
            algo_in.step_id,
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


def segment_eval_function(data, estimation):
    _, _, proj_seg = data
    cand_seg = estimation

    proj_seg_endpts = [
        np.array([proj_seg.p1.x, proj_seg.p1.y]),
        np.array([proj_seg.p2.x, proj_seg.p2.y]),
    ]
    cand_seg_endpts = [
        np.array([cand_seg.p1.x, cand_seg.p1.y]),
        np.array([cand_seg.p2.x, cand_seg.p2.y]),
    ]

    # norm distance
    dist_proj = np.linalg.norm(proj_seg_endpts[0] - proj_seg_endpts[1])
    dist_cand = np.linalg.norm(cand_seg_endpts[0] - cand_seg_endpts[1])

    dist_diff = np.abs(dist_proj - dist_cand)
    tol = 1e-2
    # Check that endpts are the same
    proj_seg_shapely = LineString(
        [
            [proj_seg_endpts[0][0], proj_seg_endpts[0][1]],
            [proj_seg_endpts[1][0], proj_seg_endpts[1][1]],
        ]
    )
    cand_seg_shapely = LineString(
        [
            [cand_seg_endpts[0][0], cand_seg_endpts[0][1]],
            [cand_seg_endpts[1][0], cand_seg_endpts[1][1]],
        ]
    )
    cand_seg_shapely_rev = LineString(
        [
            [cand_seg_endpts[1][0], cand_seg_endpts[1][1]],
            [cand_seg_endpts[0][0], cand_seg_endpts[0][1]],
        ]
    )
    return dist_diff < tol and (
        proj_seg_shapely.equals_exact(cand_seg_shapely, tolerance=tol)
        or proj_seg_shapely.equals_exact(cand_seg_shapely_rev, tolerance=tol)
    )


def collision_check_robot_frame_loop(
    poses: list[SE2Transform],
    r: float,
    observed_obstacles_list: list[list[Polygon]],
    map: list[Polygon],
) -> list[int]:
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

    # Generate Test Data
    test_values = [
        TestCollisionCheck(
            5,
            1,
            "Project Polygon Check",
            DataGenerator.generate_axis_polygon,
            visualize_axis_poly,
            CollisionPrimitives_SeparateAxis.proj_polygon,
            segment_eval_function,
            eval_weights=(5, 5),
        ),  # Task 1: proj polygon.
        TestCollisionCheck(
            10,
            2,
            "Separating Axis Thm",
            DataGenerator.generate_SAT_poly,
            viusalize_SAT_poly,
            CollisionPrimitives_SeparateAxis.separating_axis_thm,
            float_eval_function,
            eval_weights=(20, 20),
        ),  # Task 2: Separate Axis Theorem.
        TestCollisionCheck(
            6,
            3,
            "Separating Axis Thm with Circles",
            DataGenerator.generate_SAT_poly_circle,
            viusalize_SAT_poly_circle,
            CollisionPrimitives_SeparateAxis.separating_axis_thm,
            float_eval_function,
            eval_weights=(20, 20),
        ),  # Task 3: Extended Separate Axis Theorem for circles.
        TestCollisionCheck(
            5,
            4,
            "Path Collision Check",
            lambda x: DataGenerator().generate_random_robot_map_and_path(8, x),
            visualize_map_path,
            CollisionChecker().path_collision_check,
            idx_list_eval_function,
            (20, 20),
        ),  # Task 4
        TestCollisionCheck(
            5,
            5,
            "Path Collision Check - Occupancy Grid",
            lambda x: DataGenerator().generate_random_robot_map_and_path(9, x),
            visualize_map_path,
            CollisionChecker().path_collision_check_occupancy_grid,
            idx_list_eval_function,
            (20, 20),
        ),  # Task 5
        TestCollisionCheck(
            5,
            6,
            "Path Collision Check - R-Tree",
            lambda x: DataGenerator().generate_random_robot_map_and_path(10, x),
            visualize_map_path,
            CollisionChecker().path_collision_check_r_tree,
            idx_list_eval_function,
            (30, 30),
        ),  # Task 6
        TestCollisionCheck(
            5,
            7,
            "Collision Check - Rigid Body Transformation",
            DataGenerator().generate_robot_frame_data,
            visualize_robot_frame_map,
            collision_check_robot_frame_loop,
            idx_list_eval_function,
            (20, 20),
        ),  # Task 7
        TestCollisionCheck(
            5,
            8,
            "Path Collision Check - Safety Certificates",
            lambda x: DataGenerator().generate_random_robot_map_and_path(12, x),
            visualize_map_path,
            CollisionChecker().path_collision_check_safety_certificate,
            idx_list_eval_function,
            (30, 30),
        ),  # Task 8
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

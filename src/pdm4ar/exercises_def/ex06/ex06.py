import random
import timeit
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Optional, Sequence

import numpy as np
from dg_commons import SE2Transform
from pdm4ar.exercises.ex06.collision_checker import CollisionChecker
from pdm4ar.exercises_def.ex06.data import DataGenerator
from pdm4ar.exercises_def.ex06.sampling_data import (
    PUBLIC_PRM_CASES,
    PUBLIC_RRT_STAR_CASES,
    SamplingDataGenerator,
)
from pdm4ar.exercises_def.ex06.structures import Circle, Path, Point, Polygon, Triangle
from pdm4ar.exercises_def.ex06.visualization import (
    visualize_map_path,
    visualize_planning_problem,
    visualize_prm_problem,
    visualize_robot_frame_map,
)
from pdm4ar.exercises_def.structures import Exercise, ExIn, PerformanceResults
from reprep import Report
from shapely import geometry

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
    impl_validator: Optional[Callable[[Callable, Any], tuple[bool, str]]] = None

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
    ) -> CollisionCheckWeightedPerformance:

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

    # Validate implementation
    if algo_in.impl_validator is not None:
        data = algo_in.sample_generator(0)
        is_valid, error_msg = algo_in.impl_validator(algo_in.ex_function, *data[:-1])
        if not is_valid:
            raise RuntimeError(error_msg)

        # Validators execute student code and may consume random state. Restore the
        # seed so that every implementation is evaluated on the same test data.
        set_random_seed(RANDOM_SEED)

    accuracy_list = []
    solve_times = []
    test_data = [
        algo_in.sample_generator(ex_num)
        for ex_num in range(algo_in.number_of_test_cases)
    ]

    for ex_num, data in enumerate(test_data):
        start = timeit.default_timer()
        estimate = algo_in.ex_function(*data[:-1])
        stop = timeit.default_timer()
        solve_times.append(stop - start)

        accuracy_list.append(algo_in.eval_function(data, estimate))
        try:
            algo_in.visualizer(
                r, f"step-{algo_in.step_id}-{ex_num}", data, estimate
            )
        except TypeError:
            try:
                algo_in.visualizer(r, f"step-{algo_in.step_id}-{ex_num}", data)
            except Exception:
                pass
        except Exception:
            pass

        r.text(
            f"{algo_in.str_id()}-{ex_num}",
            f"Ground Truth = {data[-1]} | Estimation = {_summarize_estimate(estimate)} | Execution Time = {round(stop - start, 5)}",
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
            float(np.mean(accuracy_list)),
            float(np.mean(solve_times)),
            algo_in.eval_weights,
            algo_in.step_id,
        ),
        r,
    )


def _summarize_estimate(estimate: Any) -> Any:
    if isinstance(estimate, Path):
        return f"Path with {len(estimate.waypoints)} waypoints"
    if isinstance(estimate, list) and (not estimate or isinstance(estimate[0], Path)):
        return f"{len(estimate)} query paths"
    if isinstance(estimate, list) and estimate and isinstance(estimate[0], Point):
        return f"{len(estimate)} sampled points"
    return estimate


def idx_list_eval_function(data, estimation):
    if not isinstance(estimation, list):
        return 0.0

    segment_count = max(0, len(data[0]) - 1)
    if any(
        isinstance(index, bool)
        or not isinstance(index, Integral)
        or not 0 <= int(index) < segment_count
        for index in estimation
    ):
        return 0.0

    estimation_indices = {int(index) for index in estimation}
    if len(estimation_indices) != len(estimation):
        return 0.0
    if segment_count == 0:
        return 1.0

    ground_truth_indices = set(data[-1])
    ground_truth_bool = np.array(
        [i in ground_truth_indices for i in range(segment_count)]
    )
    estimation_bool = np.array(
        [i in estimation_indices for i in range(segment_count)]
    )

    return float((ground_truth_bool == estimation_bool).mean())


def _path_cost(path: Path) -> float:
    return float(
        sum(
            np.hypot(second.x - first.x, second.y - first.y)
            for first, second in zip(path.waypoints[:-1], path.waypoints[1:])
        )
    )


def _evaluation_path_has_collision(path: Path, radius: float, obstacles) -> bool:
    path_geometry = geometry.LineString(
        [(point.x, point.y) for point in path.waypoints]
    )
    for obstacle in obstacles:
        if isinstance(obstacle, Circle):
            obstacle_geometry = geometry.Point(obstacle.center.x, obstacle.center.y)
            clearance = radius + obstacle.radius
        elif isinstance(obstacle, Polygon):
            obstacle_geometry = geometry.Polygon(
                [(vertex.x, vertex.y) for vertex in obstacle.vertices]
            )
            clearance = radius
        elif isinstance(obstacle, Triangle):
            obstacle_geometry = geometry.Polygon(
                [
                    (obstacle.v1.x, obstacle.v1.y),
                    (obstacle.v2.x, obstacle.v2.y),
                    (obstacle.v3.x, obstacle.v3.y),
                ]
            )
            clearance = radius
        else:
            raise TypeError(f"Unsupported obstacle type: {type(obstacle).__name__}")
        if path_geometry.distance(obstacle_geometry) <= clearance:
            return True
    return False


def _valid_path(
    path: Path,
    start: Point,
    goal: Point,
    bounds,
    radius: float,
    obstacles,
) -> bool:
    if not isinstance(path, Path) or len(path.waypoints) < 2:
        return False
    if path.waypoints[0] != start or path.waypoints[-1] != goal:
        return False
    if any(
        not isinstance(point, Point)
        or not np.isfinite(point.x)
        or not np.isfinite(point.y)
        or not (
            bounds.p_min.x <= point.x <= bounds.p_max.x
            and bounds.p_min.y <= point.y <= bounds.p_max.y
        )
        for point in path.waypoints
    ):
        return False
    return not _evaluation_path_has_collision(path, radius, obstacles)


def prm_eval_function(data, estimation):
    samples, queries, bounds, radius, obstacles, connection_radius, expected = data
    if not isinstance(estimation, list) or len(estimation) != len(queries):
        return 0.0

    allowed = set(samples)
    scores = []
    for query_index, ((start, goal), path) in enumerate(zip(queries, estimation)):
        reference_cost = expected[query_index] if expected else float("nan")
        if np.isinf(reference_cost):
            scores.append(float(isinstance(path, Path) and not path.waypoints))
            continue
        if not _valid_path(path, start, goal, bounds, radius, obstacles):
            scores.append(0.0)
            continue
        if any(point not in allowed for point in path.waypoints):
            scores.append(0.0)
            continue
        edge_lengths = [
            np.hypot(second.x - first.x, second.y - first.y)
            for first, second in zip(path.waypoints[:-1], path.waypoints[1:])
        ]
        if any(length > connection_radius + 1e-9 for length in edge_lengths):
            scores.append(0.0)
            continue
        candidate_cost = float(sum(edge_lengths))
        scores.append(
            1.0
            if not np.isfinite(reference_cost)
            else min(1.0, reference_cost / max(candidate_cost, 1e-12))
        )
    return float(np.mean(scores))


def rrt_star_eval_function(data, estimation):
    start, goal, bounds, radius, obstacles = data[:5]
    reference_cost = float(data[-1])
    if np.isinf(reference_cost):
        return float(isinstance(estimation, Path) and not estimation.waypoints)
    if not _valid_path(estimation, start, goal, bounds, radius, obstacles):
        return 0.0
    candidate_cost = _path_cost(estimation)
    return (
        1.0
        if not np.isfinite(reference_cost)
        else min(1.0, reference_cost / max(candidate_cost, 1e-12))
    )


def collision_check_robot_frame_loop(
    poses: list[SE2Transform],
    r: float,
    observed_obstacles_list: list[list[Polygon]],
    map: list[Polygon],
) -> list[int]:
    from pdm4ar.exercises.ex06.collision_checker import CollisionChecker

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


def disallowed_validator(func: Callable, *args, **kwargs) -> tuple[bool, str]:
    import inspect  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    disallowed_dependencies = {
        "shapely",
        "Polygon3D",
        "motion_planners",
        "ompl",
        "pybullet_planning",
        "roboticstoolbox",
        "rrt_algorithms",
        "scipy.spatial",
        "sympy.geometry",
        "sys",
        "ctypes",
        "_ctypes",
    }

    called_funcs = []
    detected_libs = set()  # Track already detected libraries

    top_frame_id = id(inspect.currentframe())

    def trace_calls(frame, event, arg):  # pylint: disable=unused-argument
        import traceback  # pylint: disable=import-outside-toplevel

        if (id(frame) == top_frame_id) or (event not in ("call", "c_call")):
            return
        module = frame.f_globals.get("__name__", "")
        c_module = getattr(arg, "__module__", "") or ""  # For C functions
        for lib in disallowed_dependencies:
            if lib not in detected_libs and (
                module.startswith(lib) or c_module.startswith(lib)
            ):
                # Only record each library once
                detected_libs.add(lib)

                traces = traceback.extract_stack(frame)
                for trace in reversed(traces):
                    if trace.name == func.__name__:
                        called_funcs.append(
                            {
                                "library": lib,
                                "func_name": trace.name,
                                "lineno": trace.lineno,
                                "filename": trace.filename.split("/")[
                                    -1
                                ],  # Get the filename only
                            }
                        )
                        break
        return None

    sys.setprofile(trace_calls)
    try:
        func(*args, **kwargs)
    finally:
        sys.setprofile(None)

    # If no disallowed libraries were detected, the implementation is valid
    if len(called_funcs) == 0:
        return True, ""
    else:
        validation_details = []
        validation_details.append(
            "Implementation validation failed. Disallowed dependencies detected:"
        )
        for record in called_funcs:
            validation_details.append(
                f"  - Library: {record['library']}, "
                f"Function: {record['func_name']}, Line: {record['lineno']}, File: {record['filename']}"
            )
        error_msg = "\n".join(validation_details)
        return False, error_msg


def sampling_planner_validator(func: Callable, *args, **kwargs) -> tuple[bool, str]:
    """Reject direct planner libraries and non-deterministic implementations."""
    import sys  # pylint: disable=import-outside-toplevel

    planner_libraries = {
        "motion_planners",
        "ompl",
        "pybullet_planning",
        "roboticstoolbox",
        "rrt_algorithms",
    }
    detected: set[str] = set()

    def trace_calls(frame, event, arg):  # pylint: disable=unused-argument
        if event not in ("call", "c_call"):
            return None
        module = frame.f_globals.get("__name__", "")
        c_module = getattr(arg, "__module__", "") or ""
        for library in planner_libraries:
            if module.startswith(library) or c_module.startswith(library):
                detected.add(library)
        return None

    sys.setprofile(trace_calls)
    try:
        first_result = func(*args, **kwargs)
    finally:
        sys.setprofile(None)

    if detected:
        libraries = ", ".join(sorted(detected))
        return False, f"Direct motion-planning library call detected: {libraries}."
    if first_result != func(*args, **kwargs):
        return False, "The same input and seed must produce the same samples or path."
    return True, ""


def get_exercise6() -> Exercise:
    from pdm4ar.exercises.ex06.sampling_planners import SamplingBasedPlanner

    # Generate Test Data
    test_values = [
        TestCollisionCheck(
            5,
            1,
            "Path Collision Check",
            lambda x: DataGenerator().generate_random_robot_map_and_path(8, x),
            visualize_map_path,
            CollisionChecker().path_collision_check,
            idx_list_eval_function,
            (20, 20),
            impl_validator=disallowed_validator,
        ),  # Task 1 - Path Collision Check
        TestCollisionCheck(
            5,
            2,
            "Path Collision Check - Occupancy Grid",
            lambda x: DataGenerator().generate_random_robot_map_and_path(9, x),
            visualize_map_path,
            CollisionChecker().path_collision_check_occupancy_grid,
            idx_list_eval_function,
            (20, 20),
        ),  # Task 2 - Path Collision Check - Occupancy Grid
        TestCollisionCheck(
            5,
            3,
            "Path Collision Check - R-Tree",
            lambda x: DataGenerator().generate_random_robot_map_and_path(10, x),
            visualize_map_path,
            CollisionChecker().path_collision_check_r_tree,
            idx_list_eval_function,
            (30, 30),
        ),  # Task 3 - Path Collision Check - R-Tree
        TestCollisionCheck(
            5,
            4,
            "Collision Check - Rigid Body Transformation",
            DataGenerator().generate_robot_frame_data,
            visualize_robot_frame_map,
            collision_check_robot_frame_loop,
            idx_list_eval_function,
            (20, 20),
        ),  # Task 4 - Collision Check - Rigid Body Transformation
        TestCollisionCheck(
            5,
            5,
            "Path Collision Check - Optimization-based Collision Detection",
            lambda x: DataGenerator().generate_random_robot_map_and_path(12, x),
            visualize_map_path,
            CollisionChecker().path_collision_check_opt,
            idx_list_eval_function,
            (30, 30),
        ),  # Task 5 - Path Collision Check - Optimization-based Collision Detection
        TestCollisionCheck(
            PUBLIC_PRM_CASES,
            6,
            "Probabilistic Roadmap (PRM)",
            SamplingDataGenerator.generate_prm,
            visualize_prm_problem,
            SamplingBasedPlanner.prm,
            prm_eval_function,
            eval_weights=(20, 20),
            impl_validator=sampling_planner_validator,
        ),
        TestCollisionCheck(
            PUBLIC_RRT_STAR_CASES,
            7,
            "Rapidly-exploring Random Tree Star (RRT*)",
            SamplingDataGenerator.generate_rrt_star,
            visualize_planning_problem,
            SamplingBasedPlanner.rrt_star,
            rrt_star_eval_function,
            eval_weights=(20, 20),
            impl_validator=sampling_planner_validator,
        ),
    ]

    total_weights = (
        np.sum([t.eval_weights[0] for t in test_values]),
        np.sum([t.eval_weights[1] for t in test_values]),
    )

    return Exercise[TestCollisionCheck, Any](
        desc="This exercise covers collision checking and sampling-based planning.",
        evaluation_fun=_collision_check_rep,
        perf_aggregator=lambda x: CollisionCheckPerformance.perf_aggregator(
            x, total_weights
        ),
        test_values=test_values,
        expected_results=None,
    )

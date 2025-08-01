from dataclasses import dataclass, fields
from abc import ABC
import math
from typing import Callable, Tuple, Any, List
from dg_commons import SE2Transform
from reprep import MIME_PDF

from pdm4ar.exercises_def import PerformanceResults
from pdm4ar.exercises.ex05.structures import Curve
from pdm4ar.exercises_def.ex05.comparison import *
from pdm4ar.exercises.ex05.algo import calculate_dubins_path

PASSED_STR = "PASSED"
FAILED_STR = "FAILED!"
SOL_UNAVAILABLE = "Solution not available"

DubinsQuery = Tuple[SE2Transform, SE2Transform, float]
RadiusQuery = Tuple[float, float]


@dataclass(frozen=True)
class DubinsProblem:
    queries: List[Any]
    eval_weight: float
    id_num: int
    id_str: str
    algo_fun: Callable
    eval_fun: Callable
    pre_tf_fun: Callable = None
    plot_fun: Callable = None


@dataclass(frozen=True)
class DubinsPerformance(PerformanceResults):
    accuracy: float
    id_: int
    weight: float = 1.0
    """Percentage of correct queries"""

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1
        assert 0 < self.weight <= 1


@dataclass(frozen=True)
class DubinsFinalPerformance(PerformanceResults):
    accuracy_combined: float
    individual_accuracies: dict
    """Final performance"""

    def __post_init__(self):
        assert 0 <= self.accuracy_combined <= 1


def path_to_str(path: Path):
    return "".join([str(segment) for segment in path])


def ex1_radius_eval(algo_out, algo_out_tf, expected):
    radius = algo_out.min_radius
    gt_radius = expected.min_radius
    correct = math.isclose(gt_radius, radius)
    result_str = PASSED_STR if correct else FAILED_STR
    return correct, result_str


def ex2_turning_circle_eval(algo_out, algo_out_tf, expected):
    gt_turning_circle = expected
    are_equal = curves_are_equal(algo_out.left, gt_turning_circle.left) and curves_are_equal(
        algo_out.right, gt_turning_circle.right
    )
    result_str = PASSED_STR if are_equal else FAILED_STR
    return int(are_equal), result_str


def ex3_tangent_construct_eval(algo_out, algo_out_tf, expected):
    gt_tangent = expected
    correct_answers = 0
    result_str = ""
    if len(algo_out) == 0 and len(gt_tangent) == 0:
        correct_answers += 1
        result_str = PASSED_STR
    elif len(algo_out) == 0:
        result_str = FAILED_STR
    else:
        for tangent_gt in gt_tangent:
            found = False
            for tangent in algo_out:
                are_eq = start_end_configurations_are_equal(tangent_gt, tangent)
                if are_eq:
                    found = True
                    break
            if found:
                correct_answers += 1.0 / len(gt_tangent)
                result_str = PASSED_STR
    return correct_answers, result_str


def ex4_path_eval(algo_out, algo_out_tf, expected):
    gt_sol_dict = expected
    correct_answers = 0
    result_str = ""
    if algo_out_tf[0]:
        correct_answers, success, result_str = compare_paths(correct_answers, algo_out, algo_out_tf[1], gt_sol_dict)
    return correct_answers, result_str


def ex5_spline_eval(algo_out, algo_out_tf, expected):
    correct = 0
    result_str = ""

    if not isinstance(algo_out, tuple) or len(algo_out) != 7:
        return correct, FAILED_STR

    dubins_length, spline_length, is_feasible = algo_out[:3]
    dubins_gt, spline_gt, feasible_gt = expected[:3]

    lengths_match = math.isclose(dubins_length, dubins_gt, rel_tol=1e-6) and math.isclose(
        spline_length, spline_gt, rel_tol=1e-6
    )
    feasibility_match = is_feasible == feasible_gt

    if lengths_match and feasibility_match:
        correct = 1
        result_str = PASSED_STR
    else:
        result_str = FAILED_STR

    return correct, result_str


def ex4_pre_tf_fun(algo_out):
    algo_nonempty = bool(len(algo_out)) and all([isinstance(seg, Segment) for seg in algo_out])
    pre_msg = ""
    if algo_nonempty:
        algo_se2_path = extract_path_points(algo_out)
        algo_se2_np = se2_points_to_np_array(algo_se2_path)
        success = True
    else:
        pre_msg = f"{FAILED_STR}, returned list is empty or contains elements which are not of type Segment \n"
        success = False
        algo_se2_path = None
        algo_se2_np = None
    return success, (algo_se2_path, algo_se2_np), pre_msg


def ex3_tangent_plot_fun(rfig, query, algo_out, algo_out_tf, expected, sucess):
    with rfig.plot(nid="Graph", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        plot_circle_tangents(circle1=query[0], circle2=query[1], tan_list=algo_out, ax=ax)
        if expected is not None and not sucess:
            plot_circle_tangents(circle1=query[0], circle2=query[1], tan_list=expected, ax=ax)
        ax.axis("equal")


def ex4_path_plot_fun(rfig, query, algo_out, algo_out_tf, expected, sucess):
    with rfig.plot(nid="Graph", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        if algo_out_tf[0]:
            plot_2d_path(algo_out_tf[1], ax=ax)
        if expected is not None and not sucess:
            for gt_opt_path_np in expected["opt_np_points_list"]:
                plot_2d_path(gt_opt_path_np, ax=ax)
        plot_configuration(query[0], ax=ax, color="blue")
        plot_configuration(query[1], ax=ax)
        ax.title.set_text(path_to_str(algo_out))
        ax.axis("equal")


def ex5_pre_tf_fun(algo_out):
    try:
        # Extract Dubins path again to plot
        _, _, _, t0, t1, p0, p1 = algo_out

        # Reconstruct SE2Transform for plotting convenience
        dummy_start = SE2Transform(p0, 0.0)
        dummy_end = SE2Transform(p1, 0.0)
        radius = 1.0  # any valid radius; not relevant for plotting t0/t1

        dubins_path = calculate_dubins_path(dummy_start, dummy_end, radius=radius)
        if not dubins_path or not isinstance(dubins_path, list):
            return False, None, f"{FAILED_STR}, Dubins path is empty or invalid\n"

        se2_path = extract_path_points(dubins_path)
        np_path = se2_points_to_np_array(se2_path)

        return True, (se2_path, np_path), ""
    except Exception as e:
        return False, None, f"{FAILED_STR}, exception in pre_tf_fun: {e}\n"


def ex5_spline_plot_fun(rfig, query, algo_out, algo_out_tf, expected, success):
    with rfig.plot(nid="Graph", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()

        start, end, radius = query

        # Recompute Dubins path for visualization
        dubins_path = calculate_dubins_path(start, end, radius=radius)
        if not dubins_path or not isinstance(dubins_path, list):
            ax.set_title("No valid Dubins path")
            return
        dubins_pts = extract_path_points(dubins_path)
        dubins_np = np.array([[p.p[0], p.p[1], p.theta] for p in dubins_pts])

        plot_2d_path(dubins_np, ax=ax)

        # Recompute spline for plotting
        _, _, _, t0, t1, p0, p1 = algo_out

        def hermite(t):
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1

        ts = np.linspace(0, 1, 100)
        spline_pts = np.array([hermite(t) for t in ts])
        ax.plot(spline_pts[:, 0], spline_pts[:, 1], "r--", label="Spline")

        plot_configuration(start, ax=ax, color="blue")
        plot_configuration(end, ax=ax)
        ax.axis("equal")
        ax.legend()
        ax.set_title("Spline vs Dubins")

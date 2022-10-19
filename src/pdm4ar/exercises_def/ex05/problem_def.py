from dataclasses import dataclass
from abc import ABC
import math
from typing import Callable, Tuple, Any, List
from dg_commons import SE2Transform
from reprep import MIME_PDF

from pdm4ar.exercises_def import PerformanceResults
from pdm4ar.exercises.ex05.structures import Curve
from pdm4ar.exercises_def.ex05.comparison import *


PASSED_STR = "PASSED"
FAILED_STR = "FAILED!"
SOL_UNAVAILABLE = "Solution not available"

DubinsQuery = Tuple[SE2Transform, SE2Transform, float]
RadiusQuery = Tuple[float, float]

@dataclass(frozen=True)
class DubinsProblem():
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
    weight: float = 1.0
    """Percentage of correct queries"""

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1
        assert 0 < self.weight <= 1



def path_to_str(path: Path):
    return ''.join([str(segment) for segment in path])


def ex1_radius_eval(algo_out, algo_out_tf, expected):
    radius = algo_out.min_radius
    gt_radius = expected.min_radius
    correct = math.isclose(gt_radius, radius)
    result_str = PASSED_STR if correct else FAILED_STR
    return correct, result_str

def ex2_turning_circle_eval(algo_out, algo_out_tf, expected):
    gt_turning_circle = expected
    are_equal = curves_are_equal(algo_out.left, gt_turning_circle.left) and curves_are_equal(algo_out.right,
                                                                                   gt_turning_circle.right)
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
    return correct_answers ,result_str

def ex4_path_eval(algo_out, algo_out_tf, expected):
    gt_sol_dict = expected
    correct_answers = 0
    result_str = ""
    if algo_out_tf[0]:
        correct_answers, success, result_str = compare_paths(correct_answers, algo_out, algo_out_tf[1], gt_sol_dict)
    return correct_answers, result_str


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
        ax.axis('equal')


def ex4_path_plot_fun(rfig, query, algo_out, algo_out_tf, expected, sucess):
    with rfig.plot(nid="Graph", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        if algo_out_tf[0]:
            plot_2d_path(algo_out_tf[1], ax=ax)
        if expected is not None and not sucess:
            for gt_opt_path_np in expected["opt_np_points_list"] :
                plot_2d_path(gt_opt_path_np, ax=ax)
        plot_configuration(query[0], ax=ax, color='blue')
        plot_configuration(query[1], ax=ax)
        ax.title.set_text(path_to_str(algo_out))
        ax.axis('equal')
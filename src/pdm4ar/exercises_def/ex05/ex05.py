# TODO this file should be avaiable to student move to other repo
from numpy import empty
import pdm4ar.exercises.ex05.algo as algo
import math
from pdm4ar.exercises_def import Exercise, PerformanceResults
from pdm4ar.exercises_def.ex05.comparison import *
from dataclasses import dataclass
from abc import ABC
from typing import Tuple
from reprep import Report, MIME_PDF
from zuper_commons.text import remove_escapes

DubinsQuery = Tuple[SE2Transform, SE2Transform, float]
RadiusQuery = Tuple[float, float]

SEGMENT_LEVEL_CHECK = False  # Compare paths on the object segment level for each paths
SE2_LEVEL_CHECK = True  # Compare paths using list of points on both paths
assert SEGMENT_LEVEL_CHECK != SE2_LEVEL_CHECK

PASSED_STR = "PASSED"
FAILED_STR = "FAILED!"
SOL_UNAVAILABLE = "Solution not available"

def path_to_str(path: Path):
    return ''.join([str(segment) for segment in path])

@dataclass(frozen=True)
class DubinsProblem(ABC):
    weight: float

@dataclass(frozen=True)
class DubinsPathRadiusProblem(DubinsProblem):
    queries: List[RadiusQuery]

@dataclass(frozen=True)
class TurningCircleProblem(DubinsProblem):
    queries: List[Tuple[SE2Transform, float]]

@dataclass(frozen=True)
class TangentConstructionProblem(DubinsProblem):
    queries: List[Tuple[Curve, Curve]]

@dataclass(frozen=True)
class DubinsPathSearchProblem(DubinsProblem):
    queries: List[DubinsQuery]

@dataclass(frozen=True)
class ReedsSheppPathSearchProblem(DubinsProblem):
    queries: List[DubinsQuery]

@dataclass(frozen=True)
class DubinsPerformance(PerformanceResults):
    accuracy: float
    weight: float = 1.0
    """Percentage of correct queries"""

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1
        assert 0 < self.weight <= 1


def exercise_dubins_eval(prob: DubinsProblem,
                         expected: List[DubinsProblem],
                         ) -> Tuple[DubinsPerformance, Report]:
    
    test_queries = prob.queries
    correct_answers = 0
   
    if isinstance(prob, DubinsPathRadiusProblem):
        r = Report(f"Radius Calculation Test")
        for i, query in enumerate(test_queries):
            msg = ""
            algo_radius = algo.calculate_car_turning_radius(*query).min_radius  
        
            if expected is not None:
                gt_radius = expected[i]
                gt_radius_str = f"{gt_radius: .3f}"
                correct = math.isclose(gt_radius, algo_radius)
                correct_answers += correct
                result_str = PASSED_STR if correct else FAILED_STR
            else:
                gt_radius = SOL_UNAVAILABLE
                result_str = ""
            
            msg += f"Input:\nwheel base:\t{query[0]}\tmax steering angle:\t{query[1]}" f"\nOutput:\t{algo_radius:.3f}" 
            msg += f"\nExpectedOutput:\t {gt_radius_str} \n"
            msg += result_str
            r.text(f"Query: {i + 1}", text=remove_escapes(msg))

    if isinstance(prob, TurningCircleProblem):
        r = Report(f"Turning Circle Test")
        for i, query in enumerate(test_queries):
            # Your method TODO replace with student's version
            algo_turning_circle = algo.calculate_turning_circles(query[0], radius=query[1])
                        
            if expected is not None:
                gt_turning_circle = expected[i]

                are_equal = curves_are_equal(algo_turning_circle.left, gt_turning_circle.left) and curves_are_equal(algo_turning_circle.right,
                                                                                                gt_turning_circle.right)
                gt_str = f'Ground truth: left: {str(gt_turning_circle.left.center)}, \n right: {str(gt_turning_circle.right.center)} \n'
                if are_equal:
                    correct_answers += 1
                    result_str = PASSED_STR
                else:
                    result_str = FAILED_STR
            else:
                gt_str = SOL_UNAVAILABLE
                result_str = ""
                       
            msg = f"Configuration:\t{query[0]} with turning radius:\t{query[1]} \n"
            msg += f"Computed circles: left: {str(algo_turning_circle.left.center)}, \n right: {str(algo_turning_circle.right.center)} \n"
            msg += gt_str
            msg += result_str
            r.text(f"Query: {i + 1}", text=remove_escapes(msg))

    if isinstance(prob, TangentConstructionProblem):
        r = Report(f"Tangent Construction Test")
        for i, query in enumerate(test_queries):
           
            algo_tangent = algo.calculate_tangent_btw_curves(curve1=query[0],
                                                                 curve2=query[1]) 
            algo_str = f"Computed tangent(s) {str(*algo_tangent)} \n"
 
            if expected is not None:
                gt_tangent = expected[i]
                gt_str = f"Ground truth tangent(s): {str(*gt_tangent)} \n"
                success = False
                if len(algo_tangent) == 0 and len(gt_tangent) == 0:
                    correct_answers += 1
                    result_str = PASSED_STR
                    success = True
                elif len(algo_tangent) == 0:
                    result_str = FAILED_STR
                else:
                    for tangent_gt in gt_tangent:
                        found = False
                        for tangent in algo_tangent:
                            are_eq = start_end_configurations_are_equal(tangent_gt, tangent)
                            if are_eq:
                                found = True
                                break
                        if found:
                            correct_answers += 1.0 / len(gt_tangent)
                            result_str = PASSED_STR
            else:
                gt_str = SOL_UNAVAILABLE
                result_str = ""

            msg = f"Start Circle at :\t{query[0]} {query[0].center}, \n Goal Circle at:\t{query[1]} {query[1].center}\n"
            msg += algo_str
            msg += gt_str
            msg += result_str
            r.text(f"Query: {i + 1}", text=remove_escapes(msg))
            
            figsize = None
            rfig = r.figure(cols=1)
            with rfig.plot(nid="Graph", mime=MIME_PDF, figsize=figsize) as _:
                ax = plt.gca()
                plot_circle_tangents(circle1=query[0], circle2=query[1], tan_list=algo_tangent, ax=ax)
                if not success and expected is not None:
                    plot_circle_tangents(circle1=query[0], circle2=query[1], tan_list=gt_tangent, ax=ax)
                ax.axis('equal')
            

    if isinstance(prob, DubinsPathSearchProblem) or isinstance(prob, ReedsSheppPathSearchProblem):
        if isinstance(prob, DubinsPathSearchProblem):
            r = Report(f"Dubin's Path Test")
        else:
            r = Report(f"Reeds-Shepp Path Test")
            
        for i, query in enumerate(test_queries):
            # TODO replace with students version
            algo_path = algo.calculate_dubins_path(start_config=query[0], end_config=query[1], radius=query[2], return_all_valid_dubins_paths=True)
            
            algo_nonempty = bool(len(algo_path)) and all([isinstance(seg, Segment) for seg in algo_path])

            if algo_nonempty:
                algo_str = f"Computed path: {path_to_str(algo_path)} \n"
                algo_se2_path = extract_path_points(algo_path)
                algo_se2_np = se2_points_to_np_array(algo_se2_path)
            else:
                algo_str = f"{FAILED_STR}, returned list is empty or contains elements which are not of type Segment \n"

            if expected is not None:       
                gt_sol_dict = expected[i] 
                gt_opt = gt_sol_dict["opt_paths"]
                gt_str = f"Ground Truth Dubins' Path:  {*[path_to_str(opt_path[1]) for opt_path in gt_opt], } \n"
                if algo_nonempty:
                    correct_answers, success, result_str = compare_paths(correct_answers, algo_path, algo_se2_np, gt_sol_dict)
                else:
                    success = False
                    result_str = ""
            else:
                gt_str = SOL_UNAVAILABLE
                result_str = ""
            
            msg = f"Start: {query[0]},\nGoal: {query[1]} with radius: {query[2]} \n"
            msg += algo_str
            msg += gt_str
            msg += result_str
            r.text(f"Query: {i + 1}", text=remove_escapes(msg))

            figsize = None
            rfig = r.figure(cols=1)            
            # Plot path
            with rfig.plot(nid="Graph", mime=MIME_PDF, figsize=figsize) as _:
                ax = plt.gca()
                if algo_nonempty:
                    plot_2d_path(algo_se2_np, ax=ax)
                if not success and expected is not None:
                    for gt_opt_path_np in gt_sol_dict["opt_np_points_list"] :
                        plot_2d_path(gt_opt_path_np, ax=ax)
                plot_configuration(query[0], ax=ax, color='blue')
                plot_configuration(query[1], ax=ax)
                ax.title.set_text(algo_str)
                ax.axis('equal')
            

    msg = f"You got {correct_answers: .3f}/{len(test_queries)} correct results!"
    perf = DubinsPerformance(accuracy=float(correct_answers) / len(test_queries), weight=prob.weight)
    r.text("ResultsInfo", text=remove_escapes(msg))
    return perf, r


def compare_paths(correct_answers, algo_path, algo_se2_np, gt_sol_dict):
    result_str = ""
    success = False
    if SE2_LEVEL_CHECK: 
        # Check if path is a valid dubins path
        for gt_points_np in gt_sol_dict["valid_np_points_list"]:
            if points_are_close(gt_points_np, algo_se2_np):
                correct_answers += 0.2
                result_str += 'Path is a valid dubins path\n'
                break
        # Check if path is among optimal paths
        for gt_points_np in gt_sol_dict["opt_np_points_list"]:
            if points_are_close(gt_points_np, algo_se2_np):
                success = True
                correct_answers += 0.8
                result_str += f'Path is an optimal dubins path \n {PASSED_STR} \n'
                break
    if SEGMENT_LEVEL_CHECK: 
        # Check if path is a valid dubins path
        for path in gt_sol_dict["valid_paths"]:
            if paths_are_equal(algo_path, path[1]):
                if not success:
                    correct_answers += 0.2  # TODO TBD
                result_str += 'Path is a valid dubins path \n'
                break
        # Check if path is among optimal paths
        for path in gt_sol_dict["opt_paths"]:
            if paths_are_equal(algo_path, path[1]):
                if not success:
                    correct_answers += 0.8  # TODO TBD
                result_str += f'Path is optimal \n {PASSED_STR}'
                success = True
                break
    if not success:
        result_str += FAILED_STR + ", not optimal path found"
    return correct_answers, success, result_str


def exercise_dubins_perf_aggregator(perf_outs: List[DubinsPerformance]) -> DubinsPerformance:
    return DubinsPerformance(sum([el.accuracy*el.weight for el in perf_outs]) / len(perf_outs), weight = 1)
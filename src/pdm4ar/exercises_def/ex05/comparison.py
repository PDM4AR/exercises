from dg_commons import get_distance_SE2
from pdm4ar.exercises.ex05.structures import Segment
from pdm4ar.exercises_def.ex05.utils import *

NUM_TOL = 1e-5
SEGMENT_LEVEL_CHECK = False  # Compare paths on the object segment level for each paths
SE2_LEVEL_CHECK = True  # Compare paths using list of points on both paths
assert SEGMENT_LEVEL_CHECK != SE2_LEVEL_CHECK
PASSED_STR = "PASSED"
FAILED_STR = "FAILED!"

def curves_are_equal(curve1: Curve, curve2: Curve) -> bool:
    return get_distance_SE2(curve1.center.as_SE2(), curve2.center.as_SE2()) < NUM_TOL and \
           np.allclose(curve1.radius, curve2.radius)  and \
           np.allclose(curve1.arc_angle, curve2.arc_angle) 

def start_end_configurations_are_equal(seg1: Segment, seg2: Segment) -> bool:
    return get_distance_SE2(seg1.start_config.as_SE2(), seg2.start_config.as_SE2()) < NUM_TOL and \
           get_distance_SE2(seg1.end_config.as_SE2(), seg2.end_config.as_SE2()) < NUM_TOL

def points_are_close(point_list_1: np.array, point_list_2: np.array) -> bool:
    if point_list_1.shape != point_list_2.shape:
        return False
    return np.all(np.linalg.norm(point_list_1-point_list_2, axis=1) < NUM_TOL)

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


def paths_are_equal(path1: Path, path2: Path) -> bool:
    if len(path1) != len(path2):
        return False
    for seg1, seg2 in zip(path1, path2):
        if type(seg1) is not type(seg2):
            return False
        if seg1.type is not seg2.type:
            return False
        else:
            if not start_end_configurations_are_equal(seg1, seg2):
                return False
            if not seg1.gear is seg2.gear:
                return False
            if seg1.type is not DubinsSegmentType.STRAIGHT:
                if not curves_are_equal(seg1, seg2):
                    return False
    return True

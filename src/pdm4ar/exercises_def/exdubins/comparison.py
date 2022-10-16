from dg_commons import get_distance_SE2
from pdm4ar.exercises.exdubins.structures import Segment
from pdm4ar.exercises_def.exdubins.utils import *

NUM_TOL = 1e-5

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

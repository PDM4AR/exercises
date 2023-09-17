from abc import ABC
import itertools
from turtle import circle
from typing import Tuple, List
import math
import numpy as np
from dg_commons import SE2Transform
from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.problem_def import *
import pdm4ar.exercises.ex05.algo as algo

# TBD
# TODO TBD
EX_1_RADIUS_WEIGHT = 0.05
EX_2_CURVES_WEIGHT = 0.05
EX_3_TANGENT_WEIGHT = 0.20
EX_4_DUBINS_WEIGHT = 0.60
EX_5_REEDS_WEIGHT = 0.10

assert math.isclose(EX_1_RADIUS_WEIGHT + EX_2_CURVES_WEIGHT + EX_3_TANGENT_WEIGHT + EX_4_DUBINS_WEIGHT + EX_5_REEDS_WEIGHT,1)


def get_ex1_radius_test_values() -> DubinsProblem:
    wheel_base_np = 4
    max_steering_angle_np = np.pi/4
    combined_list = [(wheel_base_np, max_steering_angle_np)]
    return DubinsProblem(queries=combined_list, id_num=1, id_str="Radius Computation Test", algo_fun=algo.calculate_car_turning_radius, eval_fun=ex1_radius_eval, eval_weight=EX_1_RADIUS_WEIGHT)


def get_ex2_turning_circles_test_values() -> DubinsProblem:
    config_list = []
    config_list.append(SE2Transform([1., 2.5], 0.))
    radius = 3
    queries = [(el, radius) for el in config_list]
    return DubinsProblem(queries=queries, id_num=2, id_str="Turning Circle Test", algo_fun=algo.calculate_turning_circles, eval_fun=ex2_turning_circle_eval, eval_weight=EX_2_CURVES_WEIGHT)

def get_ex3_tangent_start_end_test_values() -> DubinsProblem:
    queries = []
    center1 = SE2Transform([0, 0], 0)
    center2 = SE2Transform([4, 3], 0)

    circle1 = Curve.create_circle(center=center1, radius=2, config_on_circle=SE2Transform([0,-2],0.), curve_type=DubinsSegmentType.LEFT)
    circle2 = Curve.create_circle(center=center2, radius=2, config_on_circle=SE2Transform([4, 5], 0.), curve_type=DubinsSegmentType.RIGHT)

    circle3 = Curve.create_circle(center=center1, radius=2, config_on_circle=SE2Transform([0, 2], 0.), curve_type=DubinsSegmentType.RIGHT)
    circle4 = Curve.create_circle(center=center2, radius=2, config_on_circle=SE2Transform([4, 5], 0.), curve_type=DubinsSegmentType.RIGHT)

    circle5 = Curve.create_circle(center=center1, radius=4, config_on_circle=SE2Transform([0, -4], 0.), curve_type=DubinsSegmentType.LEFT)
    circle6 = Curve.create_circle(center=center2, radius=4, config_on_circle=SE2Transform([4, 7], 0.), curve_type=DubinsSegmentType.RIGHT)

    queries = [(circle1, circle2)]#[(circle1, circle2), (circle3, circle4), (circle5, circle6)]
    return DubinsProblem(queries=queries, id_num=3, id_str="Tangent Construction Test", algo_fun=algo.calculate_tangent_btw_circles, eval_fun=ex3_tangent_construct_eval, eval_weight=EX_3_TANGENT_WEIGHT, plot_fun=ex3_tangent_plot_fun)

def get_ex4_start_end_test_values() -> DubinsProblem:
    config_list = []
    queries = []
    radius = 3.5
    config_list.append((SE2Transform([1., 2.5], -np.pi/4), SE2Transform([12, 1.0], -np.pi/3))) 
    config_list.append((SE2Transform([0., 0.], -np.pi/6), SE2Transform([8, 4.], -np.pi/3))) 
    for config in config_list:
        queries += [(*config, radius) ]

    return DubinsProblem(queries=queries, id_num=4, id_str="Dubins' Path Test", algo_fun=algo.calculate_dubins_path, eval_fun=ex4_path_eval, eval_weight=EX_4_DUBINS_WEIGHT, plot_fun=ex4_path_plot_fun, pre_tf_fun=ex4_pre_tf_fun)


def get_ex5_start_end_test_values() -> DubinsProblem:
    config_list = []
    queries = []
    radius = 3.5
    config_list.append((SE2Transform([1., 2.5], 0.), SE2Transform([-8, 2.5], 0))) 
    config_list.append((SE2Transform([0., 0.], np.pi/6), SE2Transform([8, 4.], -np.pi/2))) 
    for config in config_list:
        queries += [(*config, radius) ]
    return DubinsProblem(queries=queries, id_num=5, id_str="Reeds' Path Test", algo_fun=algo.calculate_reeds_shepp_path, eval_fun=ex4_path_eval, eval_weight=EX_5_REEDS_WEIGHT, plot_fun=ex4_path_plot_fun, pre_tf_fun=ex4_pre_tf_fun)



def get_example_test_values() -> List[DubinsProblem]:
    test_values = [get_ex1_radius_test_values(), get_ex2_turning_circles_test_values(), get_ex3_tangent_start_end_test_values(),
                   get_ex4_start_end_test_values(), get_ex5_start_end_test_values()]
    return test_values

def ex5_get_expected_results():
    expected_results = []
    expected_results[0] = DubinsParam(min_radius=4)
    expected_results[1] = TurningCircle(left=Curve.create_circle(center=SE2Transform([1.0, 5.5], 0), config_on_circle=SE2Transform.identity(), radius=3, curve_type=DubinsSegmentType.LEFT),right=Curve.create_circle(center=SE2Transform([1.0, -0.5], 0), config_on_circle=SE2Transform.identity(), radius=3, curve_type=DubinsSegmentType.RIGHT))
    expected_results[2] = [[Line(SE2Transform([2.0, 0], math.pi/2), SE2Transform([2.0, 3.0], math.pi/2), Gear.FORWARD)],
                           [Line(SE2Transform([-1.2, 1.6], 0.6435011087932844), SE2Transform([2.8, 4.6], 0.6435011087932844), Gear.FORWARD)],
                            []]
    expected_results[3] = [[Curve(SE2Transform([1.0, 2.5], -0.7853981633974483), SE2Transform([4.42379809, 1.60596552], 0.2745577143223241), SE2Transform([3.47487373, 4.97487373], 0), 3.5, DubinsSegmentType.LEFT, 1.059955877719772),
                            Line(SE2Transform([4.42379809, 1.60596552], 0.2745577143223241), SE2Transform([8.01998673, 2.61890822], 0.2745577143223241)),
                            Curve(SE2Transform([8.01998673, 2.61890822], 0.2745577143223241), SE2Transform([12.0, 1.0], -1.0471975511965976), SE2Transform([8.96891109, -0.75], 0), 3.5, DubinsSegmentType.RIGHT, 1.3217552655189215)],
                           [Curve(SE2Transform([0, 0], -0.5235987755982988), SE2Transform([2.73397724, -0.3277485], 0.2849780173805057), SE2Transform([1.75, 3.03108891], 0), 3.5, DubinsSegmentType.LEFT, 0.8085767929788048),
                            Line(SE2Transform([2.73397724, -0.3277485], 0.2849780173805057), SE2Transform([12.01506616, 2.39116258], 0.2849780173805057)),
                            Curve(SE2Transform([12.01506616, 2.39116258], 0.2849780173805057), SE2Transform([8.0, 4.0], -1.0471975511965976), SE2Transform([11.03108891, 5.75], 0), 3.5, DubinsSegmentType.LEFT, 4.951009738602483)]]
    expected_results[4] = [[Curve(SE2Transform([1.0, 2.5], 0), SE2Transform([1.0, 2.5], 0), SE2Transform([1.0, -1.0], 0), 3.5, DubinsSegmentType.RIGHT, 0, Gear.REVERSE),
                            Line(SE2Transform([1.0, 2.5], 0), SE2Transform([-8, 2.5], 0), Gear.REVERSE),
                            Curve(SE2Transform([-8, 2.5], 0), SE2Transform([-8, 2.5], 0), SE2Transform([-8, -1], 0), 3.5, DubinsSegmentType.RIGHT, 0, Gear.REVERSE)],
                           [Curve(SE2Transform([0, 0], 0.5235987755982988), SE2Transform([1.48198942, 4.37431056], 1.964680031035629), SE2Transform([-1.75, 3.03108891], 0), 3.5, DubinsSegmentType.LEFT, 1.44108125543733),
                            Curve(SE2Transform([1.48198942, 4.37431056], 1.964680031035629), SE2Transform([8.10698942, 4.8587661], -1.8186891508705159), SE2Transform([4.71397884, 5.7175322], 0), 3.5, DubinsSegmentType.RIGHT, 3.783369181906145),
                            Curve(SE2Transform([8.10698942, 4.8587661], -1.8186891508705159), SE2Transform([8.0, 4.0], -1.5707963267948966), SE2Transform([11.5, 4.0], 0), 3.5, DubinsSegmentType.LEFT, 0.24789282407561952)]]
    return expected_results
from abc import ABC
import itertools
from turtle import circle
from typing import Tuple, List
import math
import numpy as np
from dg_commons import SE2Transform
from pdm4ar.exercises.ex05.structures import Curve, DubinsSegmentType
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
    center2 = SE2Transform([8, 8.2], 0)
    circle1 = Curve.create_circle(center=center1, radius=3, config_on_circle=SE2Transform([0., 3], 0.), curve_type=DubinsSegmentType.LEFT)
    circle2 = Curve.create_circle(center=center2, radius=3, config_on_circle=SE2Transform([11., 8.2], 0.), curve_type=DubinsSegmentType.RIGHT)
    queries = [(circle1, circle2)]
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

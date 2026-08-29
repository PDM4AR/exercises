import pdm4ar.exercises.ex05.algo as algo
from pdm4ar.exercises_def.ex05.problem_def import *
import sympy as sp

# TBD
# TODO TBD
EX_1_RADIUS_WEIGHT = 0.05
EX_2_CURVES_WEIGHT = 0.05
EX_3_TANGENT_WEIGHT = 0.20
EX_4_DUBINS_WEIGHT = 0.45
EX_5_SPLINE_WEIGHT = 0.10
EX_6_REEDS_WEIGHT = 0.10
EX_7_CHOW_WEIGHT = 0.05

assert math.isclose(
    EX_1_RADIUS_WEIGHT
    + EX_2_CURVES_WEIGHT
    + EX_3_TANGENT_WEIGHT
    + EX_4_DUBINS_WEIGHT
    + EX_6_REEDS_WEIGHT
    + EX_5_SPLINE_WEIGHT
    + EX_7_CHOW_WEIGHT,
    1,
)


def get_ex1_radius_test_values() -> DubinsProblem:
    wheel_base_np = 4
    max_steering_angle_np = np.pi / 4
    combined_list = [(wheel_base_np, max_steering_angle_np)]
    return DubinsProblem(
        queries=combined_list,
        id_num=1,
        id_str="Radius Computation Test",
        algo_fun=algo.calculate_car_turning_radius,
        eval_fun=ex1_radius_eval,
        eval_weight=EX_1_RADIUS_WEIGHT,
    )


def get_ex2_turning_circles_test_values() -> DubinsProblem:
    config_list = []
    config_list.append(SE2Transform([1.0, 2.5], 0.0))
    radius = 3
    queries = [(el, radius) for el in config_list]
    return DubinsProblem(
        queries=queries,
        id_num=2,
        id_str="Turning Circle Test",
        algo_fun=algo.calculate_turning_circles,
        eval_fun=ex2_turning_circle_eval,
        eval_weight=EX_2_CURVES_WEIGHT,
    )


def get_ex3_tangent_start_end_test_values() -> DubinsProblem:
    queries = []
    center1 = SE2Transform([0, 0], 0)
    center2 = SE2Transform([4, 3], 0)

    circle1 = Curve.create_circle(
        center=center1, radius=2, config_on_circle=SE2Transform([0, -2], 0.0), curve_type=DubinsSegmentType.LEFT
    )
    circle2 = Curve.create_circle(
        center=center2, radius=2, config_on_circle=SE2Transform([4, 5], 0.0), curve_type=DubinsSegmentType.RIGHT
    )

    circle3 = Curve.create_circle(
        center=center1, radius=2, config_on_circle=SE2Transform([0, 2], 0.0), curve_type=DubinsSegmentType.RIGHT
    )
    circle4 = Curve.create_circle(
        center=center2, radius=2, config_on_circle=SE2Transform([4, 5], 0.0), curve_type=DubinsSegmentType.RIGHT
    )

    circle5 = Curve.create_circle(
        center=center1, radius=4, config_on_circle=SE2Transform([0, -4], 0.0), curve_type=DubinsSegmentType.LEFT
    )
    circle6 = Curve.create_circle(
        center=center2, radius=4, config_on_circle=SE2Transform([4, 7], 0.0), curve_type=DubinsSegmentType.RIGHT
    )

    queries = [(circle1, circle2), (circle3, circle4), (circle5, circle6)]
    return DubinsProblem(
        queries=queries,
        id_num=3,
        id_str="Tangent Construction Test",
        algo_fun=algo.calculate_tangent_btw_circles,
        eval_fun=ex3_tangent_construct_eval,
        eval_weight=EX_3_TANGENT_WEIGHT,
        plot_fun=ex3_tangent_plot_fun,
    )


def get_ex4_start_end_test_values() -> DubinsProblem:
    config_list = []
    queries = []
    radius = 3.5
    config_list.append((SE2Transform([1.0, 2.5], -np.pi / 4), SE2Transform([12, 1.0], -np.pi / 3)))
    config_list.append((SE2Transform([0.0, 0.0], 0), SE2Transform([0.0, 7], -np.pi)))
    config_list.append((SE2Transform([0.0, 0.0], -np.pi / 6), SE2Transform([8, 4.0], -np.pi / 3)))
    config_list.append((SE2Transform([0.0, 0.0], np.pi / 2), SE2Transform([7, 0.0], np.pi / 2)))
    for config in config_list:
        queries += [(*config, radius)]

    return DubinsProblem(
        queries=queries,
        id_num=4,
        id_str="Dubins' Path Test",
        algo_fun=algo.calculate_dubins_path,
        eval_fun=ex4_path_eval,
        eval_weight=EX_4_DUBINS_WEIGHT,
        plot_fun=ex4_path_plot_fun,
        pre_tf_fun=ex4_pre_tf_fun,
    )


def get_ex5_spline_comparison_values() -> DubinsProblem:
    config_list = []
    queries = []
    radius = 3.5
    config_list.append((SE2Transform([0.0, 0.0], np.pi / 6), SE2Transform([8.0, 4.0], -np.pi / 2)))
    config_list.append((SE2Transform([0.0, 0.0], 0.0), SE2Transform([10.0, 2.0], np.pi / 8)))
    config_list.append((SE2Transform([-3.0, 7.0], 1.05 * np.pi), SE2Transform([-4.0, 2.5], 1.83 * np.pi)))
    config_list.append((SE2Transform([0.0, 0.0], np.pi / 2), SE2Transform([3.0, 7.0], np.pi / 2)))

    for config in config_list:
        queries += [(*config, radius)]

    return DubinsProblem(
        queries=queries,
        id_num=5,
        id_str="Spline vs Dubins Comparison",
        algo_fun=algo.compare_spline_to_dubins,
        eval_fun=ex5_spline_eval,
        eval_weight=EX_5_SPLINE_WEIGHT,
        plot_fun=ex5_spline_plot_fun,
        pre_tf_fun=ex5_pre_tf_fun,
    )


def get_ex6_start_end_test_values() -> DubinsProblem:
    config_list = []
    queries = []
    radius = 3.5
    config_list.append((SE2Transform([1.0, 2.5], 0.0), SE2Transform([-8, 2.5], 0)))
    config_list.append((SE2Transform([0.0, 0.0], np.pi / 6), SE2Transform([8, 4.0], -np.pi / 2)))
    config_list.append((SE2Transform([0.0, 0.0], np.pi / 2), SE2Transform([-7, 0.0], -np.pi / 2)))
    config_list.append((SE2Transform([-3.0, 7.0], 1.05 * np.pi), SE2Transform([-4, 2.5], 1.83 * np.pi)))

    for config in config_list:
        queries += [(*config, radius)]
    return DubinsProblem(
        queries=queries,
        id_num=6,
        id_str="Reeds' Path Test",
        algo_fun=algo.calculate_reeds_shepp_path,
        eval_fun=ex4_path_eval,
        eval_weight=EX_6_REEDS_WEIGHT,
        plot_fun=ex4_path_plot_fun,
        pre_tf_fun=ex4_pre_tf_fun,
    )


def get_ex7_chow_rank_test_values() -> DubinsProblem:
    queries = []

    x, y, theta = sp.symbols("x y theta", real=True)
    unicycle_fields = [
        sp.Matrix([sp.cos(theta), sp.sin(theta), 0]),
        sp.Matrix([0, 0, 1]),
    ]
    queries.extend(
        [
            (
                unicycle_fields,
                [x, y, theta],
                {x: sp.Rational(1, 2), y: -1, theta: sp.pi / 2},
                1,
            ),
        ]
    )

    x, y, theta, phi = sp.symbols("x y theta phi", real=True)
    bicycle_fields = [
        sp.Matrix([sp.cos(theta), sp.sin(theta), sp.tan(phi) / 2, 0]),
        sp.Matrix([0, 0, 0, 1]),
    ]
    queries.append((bicycle_fields, [x, y, theta, phi], {x: 0, y: 0, theta: 0, phi: 0}, 2))

    x, y, z = sp.symbols("x y z", real=True)
    depth_fields = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, x**2])]
    queries.extend(
        [
            (depth_fields, [x, y, z], {x: 0, y: 0, z: 0}, 1),
            (depth_fields, [x, y, z], {x: 1, y: -2, z: 4}, 1),
        ]
    )

    translation_fields = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0])]
    queries.append((translation_fields, [x, y, z], {x: -3, y: 4, z: 7}, 2))

    return DubinsProblem(
        queries=queries,
        id_num=7,
        id_str="Chow Controllability Rank Test",
        algo_fun=algo.compute_chow_closure,
        eval_fun=ex7_chow_rank_eval,
        eval_weight=EX_7_CHOW_WEIGHT,
        plot_fun=ex7_chow_rank_plot_fun,
    )


def get_example_test_values() -> List[DubinsProblem]:
    test_values = [
        get_ex1_radius_test_values(),
        get_ex2_turning_circles_test_values(),
        get_ex3_tangent_start_end_test_values(),
        get_ex4_start_end_test_values(),
        get_ex5_spline_comparison_values(),
        get_ex6_start_end_test_values(),
        get_ex7_chow_rank_test_values(),
    ]
    return test_values

import random
from typing import Optional, Tuple, Mapping

from dg_commons import apply_SE2_to_shapely_geo, X, PlayerName, fd
from dg_commons.maps.road_bounds import build_road_boundary_obstacle
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.maps.shapes_generator import create_random_starshaped_polygon
from dg_commons.sim.goals import PolygonGoal, PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator_visualisation import ZOrders
from geometry import SE2_from_xytheta
from matplotlib import pyplot as plt
from numpy import deg2rad
from shapely.geometry import Polygon

PDM4AR_1 = PlayerName("PDM4AR_1")
PDM4AR_2 = PlayerName("PDM4AR_2")
PDM4AR_3 = PlayerName("PDM4AR_3")
PDM4AR_4 = PlayerName("PDM4AR_4")


Goals = Mapping[PlayerName, PlanningGoal]


def get_dgscenario(seed: Optional[int] = None) -> Tuple[DgScenario, Goals, X]:
    scenario_name = "USA_Lanker-1_1_T-1"
    cm_scenario, _ = load_commonroad_scenario(scenario_name, scenarios_dir=".")

    _, gates = build_road_boundary_obstacle(cm_scenario)

    shapes = []
    poly1 = Polygon([[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]])
    poly2 = apply_SE2_to_shapely_geo(poly1, SE2_from_xytheta((7, 15, deg2rad(30))))
    shapes += [poly2, ]
    if seed is not None:
        random.seed(seed)

    positions = [(-5, 5), (-30, -20), (-10, -15), (-20, 0), (10, 60), (20, 45), ]
    for pos in positions:
        poly = Polygon(create_random_starshaped_polygon(*pos, 2, 0.3, 0.5, 10))
        shapes.append(poly)

    obstacles = list(map(StaticObstacle, shapes))
    static_obstacles = dict(zip(range(len(obstacles)), obstacles))

    x0 = VehicleState(x=-20, y=-40, psi=deg2rad(65), vx=2, delta=-0.02)
    goal_poly = random.sample(gates, 1)[0]
    goals: Goals = fd({PDM4AR_1: PolygonGoal(goal_poly.buffer(1.5))})
    dg_scenario = DgScenario(scenario=cm_scenario, use_road_boundaries=True, static_obstacles=static_obstacles)

    return dg_scenario, goals, x0


if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    dg_scenario, goal, _ = get_dgscenario(seed=98)
    ax = plt.gca()
    shapely_viz = ShapelyViz(ax)

    for s_obstacle in dg_scenario.static_obstacles.values():
        shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
    shapely_viz.add_shape(goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
    ax = shapely_viz.ax
    ax.autoscale()
    ax.set_aspect("equal")
    plt.savefig("tmp.png", dpi=300)

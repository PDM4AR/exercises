import random
from typing import Optional, Mapping

from dg_commons import apply_SE2_to_shapely_geo, PlayerName
from dg_commons.maps.shapes_generator import create_random_starshaped_polygon
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.structures import DgScenario
from geometry import SE2_from_xytheta
from numpy import deg2rad
from shapely.geometry import Polygon

PDM4AR_1 = PlayerName("PDM4AR_1")
PDM4AR_2 = PlayerName("PDM4AR_2")
PDM4AR_3 = PlayerName("PDM4AR_3")
PDM4AR_4 = PlayerName("PDM4AR_4")

Goals = Mapping[PlayerName, PlanningGoal]


def get_dgscenario(seed: Optional[int] = None) -> DgScenario:
    scenario_name = "USA_Lanker-1_1_T-1"
    cm_scenario, _ = load_commonroad_scenario(scenario_name, scenarios_dir=".")

    shapes = []
    poly1 = Polygon([[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]])
    poly2 = apply_SE2_to_shapely_geo(poly1, SE2_from_xytheta((7, 15, deg2rad(30))))
    shapes += [poly2, ]
    if seed is not None:
        random.seed(seed)

    positions = [(-5, 10), (-30, -20), (-10, -15), (-22, 0), (10, 60), (20, 45), ]
    for pos in positions:
        poly = Polygon(create_random_starshaped_polygon(*pos, 2, 0.3, 0.5, 10))
        shapes.append(poly)

    obstacles = list(map(StaticObstacle, shapes))
    static_obstacles = dict(zip(range(len(obstacles)), obstacles))

    dg_scenario = DgScenario(scenario=cm_scenario, use_road_boundaries=True, static_obstacles=static_obstacles)

    return dg_scenario

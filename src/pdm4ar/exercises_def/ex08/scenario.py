import random
from pathlib import Path
from typing import Optional, Mapping

from dg_commons.maps.shapes_generator import create_random_starshaped_polygon
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.structures import DgScenario
from shapely.geometry import Polygon


def get_dgscenario(config_dict: Mapping, seed: Optional[int] = None) -> DgScenario:
    scenario_name = "USA_Lanker-1_1_T-1"
    scenarios_dir = Path(__file__).parent
    cm_scenario, _ = load_commonroad_scenario(scenario_name, scenarios_dir=str(scenarios_dir))

    if seed is not None:
        random.seed(seed)

    shapes = []
    avg_radius: float = config_dict["static_obstacles"]["avg_radius"]
    irregularity: float = config_dict["static_obstacles"]["irregularity"]
    spikiness: float = config_dict["static_obstacles"]["spikiness"]
    n_vertices: float = config_dict["static_obstacles"]["n_vertices"]
    for pos in config_dict["static_obstacles"]["centers"]:
        poly = Polygon(create_random_starshaped_polygon(*pos, avg_radius, irregularity, spikiness, n_vertices))
        shapes.append(poly)

    obstacles = list(map(StaticObstacle, shapes))

    dg_scenario = DgScenario(scenario=cm_scenario, static_obstacles=obstacles, use_road_boundaries=True)

    return dg_scenario

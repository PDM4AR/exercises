from collections import defaultdict
from copy import deepcopy
from decimal import Decimal as D
from typing import Any, Mapping

import yaml
from dg_commons import PlayerName, fd
from dg_commons.perception.sensor import VisRangeSensor
from dg_commons.sim import SimParameters
from dg_commons.sim.goals import PolygonGoal
from dg_commons.sim.models.diff_drive import DiffDriveModel, DiffDriveState
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.scenarios import DgScenario
from dg_commons.sim.sim_perception import ObsFilter, FovObsFilter
from dg_commons.sim.simulator import SimContext
from shapely import Polygon, LinearRing, Point

from pdm4ar.exercises.ex10.agent import Pdm4arAgent


def _load_config(file_path: str) -> Mapping[str, Any]:
    with open(file_path, "r") as file:
        config: dict[str, Any] = yaml.safe_load(file)
    return fd(config)


def sim_context_from_yaml(file_path: str):
    config = _load_config(file_path=file_path)

    #  obstacles
    shapes = list(map(Polygon, config["static_obstacles"]))
    obstacles = list(map(StaticObstacle, shapes))
    boundary = [StaticObstacle(LinearRing(config["boundary"])), ]
    static_obstacles = obstacles + boundary

    # add agents
    agents_dict = config["agents"]
    players, models, missions = {}, {}, {}
    for pn, p_attr in agents_dict.items():
        x0 = DiffDriveState(**p_attr["state"])
        color = p_attr["color"]
        model = DiffDriveModel(
                x0=x0,
                vg=DiffDriveGeometry.default(color=color),
                vp=DiffDriveParameters.default(omega_limits=(-10, 10))
        )
        models[pn] = model
        player = Pdm4arAgent()
        players[pn] = player
        goal_poly = Point(p_attr["goal"]).buffer(p_attr["goal_radius"])
        goal = PolygonGoal(goal_poly)
        missions[pn] = goal
    # sensing
    lidar2d = VisRangeSensor(range=30)
    sensors: dict[PlayerName, ObsFilter] = defaultdict(lambda: FovObsFilter(deepcopy(lidar2d)))
    return SimContext(
            dg_scenario=DgScenario(static_obstacles=static_obstacles),
            models=models,
            players=players,
            missions=missions,
            sensors=sensors,
            param=SimParameters(
                    dt=D("0.01"),
                    dt_commands=D("0.1"),
                    sim_time_after_collision=D(3),
                    max_sim_time=D(90),
            ),
            seed=config["seed"],
            description=file_path.split("/")[-1].split(".")[0],
    )


# if __name__ == "__main__":
#     from pathlib import Path
#     from pprint import pprint
#
#     configs = ["config_1.yaml", "config_2.yaml"]
#     for c in configs:
#         config_file = Path(__file__).parent / c
#         config = _load_config(str(config_file))
#         pprint(config)
#
#         # test actual sim context creation
#         sim_context = sim_context_from_yaml(str(config_file))
#         pprint(sim_context)

from collections import defaultdict
from copy import deepcopy
from decimal import Decimal as D
from typing import Any, Mapping, List

import yaml
from dg_commons import PlayerName, fd
from dg_commons.perception.sensor import VisRangeSensor
from dg_commons.sim import SimParameters
from dg_commons.sim.goals import PolygonGoal
from dg_commons.sim.models.diff_drive import DiffDriveModel, DiffDriveState
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.scenarios import DgScenario
from dg_commons.sim.shared_goals import CollectionPoint, SharedPolygonGoal, SharedPolygonGoalsManager
from dg_commons.sim.sim_perception import FovObsFilter, ObsFilter
from dg_commons.sim.simulator import SimContext
from pdm4ar.exercises.ex14.agent import Pdm4arAgent
from pdm4ar.exercises_def.ex14.agent_process import AgentProcess
from shapely import LinearRing, Point, Polygon


def load_config(file_path: str) -> Mapping[str, Any]:
    with open(file_path, "r") as file:
        config: dict[str, Any] = yaml.safe_load(file)
        if "config_name" not in config:
            config["config_name"] = file_path.split("/")[-1].split(".")[0]
    return fd(config)


def sim_context_from_config(config: Mapping[str, Any]) -> SimContext:
    #  obstacles
    obstacles = None
    if "static_obstacles" in config:
        shapes = list(map(Polygon, config["static_obstacles"]))
        obstacles = list(map(StaticObstacle, shapes))

    # boundaries
    boundary = [
        StaticObstacle(LinearRing(config["boundary"])),
    ]
    static_obstacles = obstacles + boundary if obstacles is not None else boundary

    # add agents
    agents_dict = config["agents"]
    players, models, missions = {}, {}, {}
    for pn, p_attr in agents_dict.items():
        x0 = DiffDriveState(**p_attr["state"])
        color = p_attr["color"]
        model = DiffDriveModel(
            x0=x0, vg=DiffDriveGeometry.default(color=color), vp=DiffDriveParameters.default(omega_limits=(-10, 10))
        )
        models[pn] = model
        # player_capacity = p_attr["capacity"]
        player = AgentProcess(Pdm4arAgent)
        # player.set_capacity(player_capacity)
        players[pn] = player
        goal_poly = Point(p_attr["goal"]).buffer(p_attr["goal_radius"])
        goal = PolygonGoal(goal_poly)
        missions[pn] = goal
    # sensing
    lidar2d = VisRangeSensor(range=30)
    sensors: dict[PlayerName, ObsFilter] = defaultdict(lambda: FovObsFilter(deepcopy(lidar2d)))

    # shared goals manager (optional)
    shared_goals_manager = None
    if "shared_goals" in config and "collection_points" in config:
        # Parse shared goals
        shared_goals: List[SharedPolygonGoal] = []
        for goal_data in config["shared_goals"]:
            goal_poly = Point(goal_data["center"]).buffer(goal_data["radius"])
            shared_goal = SharedPolygonGoal(goal_id=goal_data["id"], polygon=goal_poly)
            shared_goals.append(shared_goal)

        # Parse collection points
        collection_points: List[CollectionPoint] = []
        for cp_data in config["collection_points"]:
            cp_poly = Point(cp_data["center"]).buffer(cp_data["radius"])
            collection_point = CollectionPoint(point_id=cp_data["id"], polygon=cp_poly)
            collection_points.append(collection_point)

        # Create the manager
        shared_goals_manager = SharedPolygonGoalsManager(shared_goals=shared_goals, collection_points=collection_points)

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
        description=config["config_name"],
        shared_goals_manager=shared_goals_manager,
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

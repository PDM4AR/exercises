from decimal import Decimal as D
from math import cos, sin, pi
from typing import Any

import yaml
from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.obstacles import (
    StaticObstacle,
    DynObstacleParameters,
    ObstacleGeometry,
)
from dg_commons.sim.models.obstacles_dyn import (
    DynObstacleModel,
    DynObstacleState,
    DynObstacleCommands,
)
from dg_commons.sim.models.rocket import RocketState, RocketModel
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext
from numpy import arctan2
from shapely import LineString, Point
from shapely.geometry.base import BaseGeometry

from pdm4ar.exercises.ex09.agent import RocketAgent
from pdm4ar.exercises_def.ex09.goal import RocketTarget, SatelliteTarget


def _load_config(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as file:
        config: dict[str, Any] = yaml.safe_load(file)
    return config


def _parse_planets(
        c: dict,
) -> tuple[
    list[BaseGeometry], dict[PlayerName, DynObstacleModel], dict[PlayerName, NPAgent]
]:
    planets = []
    satellites = {}
    satellites_players = {}

    for pn, p in c["planets"].items():
        planet = Point(p["center"]).buffer(p["radius"])
        planets.append(planet)
        for sname, s in p["satellites"].items():
            s, ps = _parse_satellite(
                    planet, tau=s["tau"], orbit_r=s["orbit_r"], omega=s["omega"], radius=s["radius"]
            )
            satellite_id = pn + "/" + sname
            satellites[satellite_id] = s
            satellites_players[satellite_id] = ps

    return planets, satellites, satellites_players


def _parse_satellite(
        planet: Point, tau: float, orbit_r, omega, radius
) -> tuple[DynObstacleModel, NPAgent]:
    """
    orbit_r:
    omega:
    tau: initial angle of satellite w.r.t. mother planet
    """
    x = planet.centroid.x + orbit_r * cos(tau)
    y = planet.centroid.y + orbit_r * sin(tau)
    curr_psi = pi / 2 + arctan2(y - planet.centroid.y, x - planet.centroid.x)

    satellite_1 = DynObstacleState(
            x=x, y=y, psi=curr_psi, vx=omega * orbit_r, vy=0, dpsi=omega
    )
    satellite_1_shape = Point(0, 0).buffer(radius)
    dyn_obstacle = DynObstacleModel(
            satellite_1,
            shape=satellite_1_shape,
            og=ObstacleGeometry(m=5, Iz=50, e=0.5),
            op=DynObstacleParameters(vx_limits=(-100, 100), acc_limits=(-10, 10)),
    )
    centripetal_acc = omega ** 2 * orbit_r
    # keep sequence of commands constant
    cmds_seq = DgSampledSequence[DynObstacleCommands](
            timestamps=[0],
            values=[
                DynObstacleCommands(acc_x=0, acc_y=centripetal_acc, acc_psi=0),
            ],
    )
    player = NPAgent(cmds_seq)
    return dyn_obstacle, player


def sim_context_from_yaml(file_path: str):
    config = _load_config(file_path=file_path)

    # rocket
    assert len(config["agents"].keys()) == 1, "Only one player today"
    name = list(config["agents"])[0]
    playername = PlayerName(name)
    x0 = RocketState(**config["agents"][name]["state"])

    # obstacles (planets + satellites)
    planets, satellites, satellites_npagents = _parse_planets(config)
    env_limits = LineString(config["boundary"]["corners"])
    planets.append(env_limits)

    static_obstacles: list[StaticObstacle] = [StaticObstacle(shape=s) for s in planets]

    # load goal
    conf_goal = config["agents"][name]["goal"]
    conf_goal_type = conf_goal["type"]
    if conf_goal_type == "static":
        x0_target = DynObstacleState(**conf_goal["state"])
        goal = RocketTarget(
                target=x0_target, pos_tol=conf_goal["pos_tolerance"], vel_tol=conf_goal["vel_tolerance"]
        )
    elif conf_goal_type == "satellite":
        satellite_name = conf_goal["name"]
        target_x0 = satellites[satellite_name].get_state()
        satellite_config = config["planets"][satellite_name.split("/")[0]]["satellites"][
            satellite_name.split("/")[1]]
        goal = SatelliteTarget(target=target_x0,
                               omega=satellite_config["omega"],
                               tau=satellite_config["tau"],
                               orbit_r=satellite_config["orbit_r"],
                               radius=satellite_config["radius"],
                               pos_tol=conf_goal["pos_tolerance"],
                               vel_tol=conf_goal["vel_tolerance"],
                               )
        pass
    else:
        raise ValueError(f"Unrecognized goal type: {conf_goal_type}")
    missions = {playername: goal}

    # models & players
    players = {playername: RocketAgent()}
    models = {playername: RocketModel.default(x0)}
    for p, s in satellites.items():
        models[p] = s

    for p, sagent in satellites_npagents.items():
        players[p] = sagent

    return SimContext(
            dg_scenario=DgScenario(static_obstacles=static_obstacles),
            models=models,
            players=players,
            missions=missions,
            param=SimParameters(
                    dt=D("0.01"),
                    dt_commands=D("0.1"),
                    sim_time_after_collision=D(4),
                    max_sim_time=D(10),
            ),
            seed=config["seed"],
            description=file_path.split("/")[-1].split(".")[0],
    )


if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    configs = ["config_planet.yaml", "config_satellites.yaml", "config_mov_target.yaml"]
    for c in configs:
        config_file = Path(__file__).parent / c
        config = _load_config(str(config_file))
        pprint(config)

        # test actual sim context creation
        sim_context = sim_context_from_yaml(str(config_file))
        pprint(sim_context)

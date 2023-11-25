from pprint import pprint
from decimal import Decimal as D
from math import cos, sin, pi
from numpy import arctan2

from typing import Any
import yaml
from shapely import LineString, Point
from shapely.geometry.base import BaseGeometry
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.rocket import RocketState, RocketCommands, RocketModel
from dg_commons.sim.simulator import SimContext
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

from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons import PlayerName, DgSampledSequence
from pdm4ar.exercises.ex09.agent import RocketAgent
from pdm4ar.exercises_def.ex09.goal import RocketTarget


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

    for p in c["planets"]:
        planet = Point(p["center"]).buffer(p["radius"])
        planets.append(planet)
        for sname, s in p["satellites"].items():
            s, ps = _parse_satellite(
                planet, tau=s["tau"], orbit_r=s["orbit_r"], omega=s["omega"]
            )
            satellites[sname] = s
            satellites_players[sname] = ps

    return planets, satellites, satellites_players


def _parse_satellite(
    planet: Point, tau: float, orbit_r, omega
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
    satellite_1_shape = Point(0, 0).buffer(1)
    dyn_obstacle = DynObstacleModel(
        satellite_1,
        shape=satellite_1_shape,
        og=ObstacleGeometry(m=5, Iz=50, e=0.5),
        op=DynObstacleParameters(vx_limits=(-100, 100), acc_limits=(-10, 10)),
    )
    centripetal_acc = omega**2 * orbit_r
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
    assert len(config["agents"].keys()) == 0, "Only one player today"
    name = config["agents"].keys()[0]
    playername = PlayerName(name)
    x0 = RocketState(**config["agents"][name]["state"])

    # obstacles (planets + satellites)
    planets, satellites, satellites_npagents = _parse_planets(config)
    env_limits = LineString(config["boundary"]["corners"])
    planets.append(env_limits)

    static_obstacles: list[StaticObstacle] = [StaticObstacle(shape=s) for s in planets]

    # load goal
    conf_goal = config["agents"][name]["goal"]
    if conf_goal == "static":
        x0_target = conf_goal["state"]
        goal = RocketTarget(
            target=x0_target, pos_tol=conf_goal["pos_tol"], vel_tol=conf_goal["vel_tol"]
        )
    elif conf_goal == "satellite":
        # TODO goal from satellite
        pass
    else:
        raise ValueError("Unrecognized goal type")
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
    )


if __name__ == "__main__":
    configs = ["config_planet.yaml", "config_satellites.yaml", "config_mov_target.yaml"]
    for c in configs:
        config_file = "src/pdm4ar/exercises_def/ex09/" + c
        config = _load_config(config_file)
        pprint(config)

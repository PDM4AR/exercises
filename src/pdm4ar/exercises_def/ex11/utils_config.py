from decimal import Decimal as D
from math import cos, sin, pi
from typing import Any
from copy import deepcopy

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
from dg_commons.sim.models.spaceship import SpaceshipState, SpaceshipModel
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext
from numpy import arctan2
from shapely import LineString, Point
from shapely.geometry.base import BaseGeometry

from pdm4ar.exercises.ex11.agent import SpaceshipAgent
from pdm4ar.exercises_def.ex11.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex11.utils_params import SatelliteParams, PlanetParams


def _load_config(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as file:
        config: dict[str, Any] = yaml.safe_load(file)
    return config


def _parse_planets(
    c: dict,
) -> tuple[
    list[BaseGeometry],
    dict[PlayerName, PlanetParams],
    dict[PlayerName, DynObstacleModel],
    dict[PlayerName, SatelliteParams],
    dict[PlayerName, NPAgent],
]:
    planets = []
    planet_params = {}
    satellites = {}
    satellites_params = {}
    satellites_players = {}

    for pn, p in c["planets"].items():
        planet = Point(p["center"]).buffer(p["radius"])

        planet_params[pn] = PlanetParams(
            center=p["center"],
            radius=p["radius"],
        )

        planets.append(planet)
        for sname, sat in p["satellites"].items():
            s, ps = _parse_satellite(
                planet, tau=sat["tau"], orbit_r=sat["orbit_r"], omega=sat["omega"], radius=sat["radius"]
            )
            satellite_id = pn + "/" + sname
            satellites[satellite_id] = s
            satellites_players[satellite_id] = ps
            satellites_params[satellite_id] = SatelliteParams(
                orbit_r=sat["orbit_r"],
                omega=sat["omega"],
                tau=sat["tau"],
                radius=sat["radius"],
            )

    return planets, planet_params, satellites, satellites_params, satellites_players


def _parse_satellite(planet: Point, tau: float, orbit_r, omega, radius) -> tuple[DynObstacleModel, NPAgent]:
    """
    orbit_r:
    omega:
    tau: initial angle of satellite w.r.t. mother planet
    """
    x = planet.centroid.x + orbit_r * cos(tau)
    y = planet.centroid.y + orbit_r * sin(tau)
    curr_psi = pi / 2 + arctan2(y - planet.centroid.y, x - planet.centroid.x)

    v = omega * orbit_r

    satellite_1 = DynObstacleState(x=x, y=y, psi=curr_psi, vx=v, vy=0, dpsi=omega)
    satellite_1_shape = Point(0, 0).buffer(radius)
    dyn_obstacle = DynObstacleModel(
        satellite_1,
        shape=satellite_1_shape,
        og=ObstacleGeometry(m=500, Iz=50, e=0.5),
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

    # Spaceship
    assert len(config["agents"].keys()) == 1, "Only one player today"
    name = list(config["agents"])[0]
    playername = PlayerName(name)
    x0 = SpaceshipState(**config["agents"][name]["state"])

    # obstacles (planets + satellites)
    planets, planets_params, satellites, satellites_params, satellites_npagents = _parse_planets(config)

    env_limits = LineString(config["boundary"]["corners"])
    planets.append(env_limits)
    obsgeo = ObstacleGeometry.default_static(color="saddlebrown")
    static_obstacles: list[StaticObstacle] = [StaticObstacle(shape=s, geometry=obsgeo) for s in planets]

    # load goal
    conf_goal = config["agents"][name]["goal"]
    conf_goal_type = conf_goal["type"]
    if conf_goal_type == "static":
        x0_target = DynObstacleState(**conf_goal["state"])
        goal = SpaceshipTarget(
            target=x0_target,
            pos_tol=conf_goal["pos_tolerance"],
            vel_tol=conf_goal["vel_tolerance"],
            dir_tol=conf_goal["dir_tolerance"],
        )
    elif conf_goal_type == "dock":
        x0_target = DynObstacleState(**conf_goal["state"])
        goal = DockingTarget(
            target=x0_target,
            pos_tol=conf_goal["pos_tolerance"],
            vel_tol=conf_goal["vel_tolerance"],
            dir_tol=conf_goal["dir_tolerance"],
            add_land_space=conf_goal["add_land_space"],
            arms_length=conf_goal["arms_length"],
            offset=conf_goal["offset"],
        )
        landing_shape = goal.get_landing_base()
        obsgeo = ObstacleGeometry.default_static(color="blue")
        static_obstacles += [StaticObstacle(shape=landing_shape, geometry=obsgeo)]
    else:
        raise ValueError(f"Unrecognized goal type: {conf_goal_type}")
    missions = {playername: goal}

    # models & players
    initstate = SpaceshipState(**config["agents"][name]["state"])
    players = {
        playername: SpaceshipAgent(
            init_state=deepcopy(initstate), satellites=deepcopy(satellites_params), planets=deepcopy(planets_params)
        )
    }

    models = {playername: SpaceshipModel.default(x0)}
    for p, s in satellites.items():
        models[p] = s

    for p, sagent in satellites_npagents.items():
        players[p] = sagent

    return SimContext(
        dg_scenario=DgScenario(static_obstacles=static_obstacles),  # need satellites
        models=models,
        players=players,
        missions=missions,
        param=SimParameters(
            dt=D("0.01"),
            dt_commands=D("0.1"),
            sim_time_after_collision=D(4),
            max_sim_time=D(60),
        ),
        seed=config["seed"],
        description=file_path.split("/")[-1].split(".")[0],
    )


if __name__ == "__main__":
    from pathlib import Path
    from pprint import pprint

    configs = ["config_planet.yaml", "config_satellites.yaml", "config_staellites_diff.yaml"]
    for c in configs:
        config_file = Path(__file__).parent / c
        config = _load_config(str(config_file))
        pprint(config)

        # test actual sim context creation
        sim_context = sim_context_from_yaml(str(config_file))
        pprint(sim_context)

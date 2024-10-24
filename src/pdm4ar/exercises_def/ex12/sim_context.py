from collections import defaultdict
from copy import deepcopy
from decimal import Decimal as D
from pathlib import Path
from typing import Dict, Optional, Mapping
import numpy as np
from geometry import SE2_from_xytheta

from commonroad.prediction.prediction import TrajectoryPrediction
from dg_commons import PlayerName, apply_SE2_to_shapely_geo
from dg_commons.maps.road_bounds import build_road_boundary_obstacle
from dg_commons.perception.sensor import VisRangeSensor
from dg_commons.sim import SimParameters, SimTime, SimModel, SimLog, logger
from dg_commons.sim.goals import RefLaneGoal
from dg_commons.sim.models.vehicle import VehicleModel, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.scenarios import DgScenario, load_commonroad_scenario
from dg_commons.sim.sim_perception import ObsFilter, FovObsFilter
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.scenarios.model_from_dyn_obstacle import (
    infer_model_from_cr_dyn_obstacle,
)
from dg_commons.sim.scenarios.utils_dyn_obstacle import (
    infer_lane_from_dyn_obs,
)
from dg_commons.sim.agents.idm_agent.idm_agent import IDMAgent
from dg_commons.sim.simulator_visualisation import ZOrders, SimRenderer
from matplotlib import pyplot as plt

from pdm4ar.exercises.ex12.agent import Pdm4arAgent
from pdm4ar.exercises_def.ex12.scenario import get_dgscenario

__all__ = ["get_sim_contexts"]


def commonroad_scenario_to_simcontext(
    scenario_name: str,
    ego_player: Optional[PlayerName] = None,
    other_players: Optional[list[PlayerName]] = None,
    sim_param: Optional[SimParameters] = None,
    seed: int = 0,
) -> SimContext:
    """
    This function loads a CommonRoad scenario and tries to convert the dynamic
    obstacles into the Model/Agent paradigm used by the driving-game simulator.
    :param scenario_name:
    :param scenarios_dir:
    :param ego_player:
    :param ego_player_type:
    :param ego_player_params:
    :param others_player_type:
    :param others_player_params:
    :param sim_param:
    :param seed:
    :param description:
    :return:
    """
    scenarios_dir = str(Path(__file__))
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, scenarios_dir)
    players, models, missions = {}, {}, {}
    static_obstacles: list[StaticObstacle] = []

    for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
        assert isinstance(dyn_obs.prediction, TrajectoryPrediction), "Only trajectory predictions are supported"
        p_name = PlayerName(f"P{i}")
        if p_name == ego_player:
            p_name = PlayerName("Ego")
            model: SimModel = infer_model_from_cr_dyn_obstacle(dyn_obs, color="firebrick")
            agent = Pdm4arAgent()
        else:
            model: SimModel = infer_model_from_cr_dyn_obstacle(dyn_obs, color="royalblue")
            agent = IDMAgent()
            dglane = infer_lane_from_dyn_obs(dyn_obs=dyn_obs, network=scenario.lanelet_network)
            players.update({p_name: agent})
            models.update({p_name: model})
            goal_progress = dglane.along_lane_from_beta(len(dglane.control_points))
            goal = RefLaneGoal(dglane, goal_progress=goal_progress)
            missions.update({p_name: goal})
        logger.info(f"Managed to load {len(players)}")

    sim_param = SimParameters() if sim_param is None else sim_param
    # _, gates = build_road_boundary_obstacle(scenario)
    # gates_united = unary_union(gates)
    # missions = {p: PolygonGoal(gates_united) for p in players.keys()}
    return SimContext(
        dg_scenario=DgScenario(
            scenario, static_obstacles=static_obstacles, use_road_boundaries=True, road_boundaries_buffer=1.5
        ),
        models=models,
        players=players,
        missions=missions,
        log=SimLog(),
        param=sim_param,
        seed=seed,
    )


def get_sim_contexts(config_dict: Mapping) -> list[SimContext]:
    sim_context_all = []
    sim_param = SimParameters(
        dt=SimTime("0.05"),
        dt_commands=SimTime("0.25"),  # IDM unstable if set to 0.5
        max_sim_time=SimTime(10),
        sim_time_after_collision=SimTime("1.0"),
    )
    seed = config_dict["seed"]
    np.random.seed(seed)
    for test_config in config_dict["Testcases"]:
        sim_context: SimContext = commonroad_scenario_to_simcontext(
            scenario_name=test_config["Scenario"],
            sim_param=sim_param,
            seed=seed,
        )
        sim_context = _set_ego_in_sim_context(sim_context, seed)
        sim_context_all.append(sim_context)

    return sim_context_all


if __name__ == "__main__":
    # matplotlib.use('TkAgg')
    from .ex12 import load_config_ex12

    config_path = Path(__file__).parent / "config_1.yaml"
    config_dict = load_config_ex12(config_path)
    sim_context = get_sim_contexts(config_dict)
    sim_renderer = SimRenderer(sim_context)
    shapely_viz = sim_renderer.shapely_viz
    ax = sim_renderer.commonroad_renderer.ax

    with sim_renderer.plot_arena(ax):
        for pn, goal in sim_context.missions.items():
            shapely_viz.add_shape(
                goal.get_plottable_geometry(), color=config_dict["agents"][pn]["color"], zorder=ZOrders.GOAL, alpha=0.5
            )
        for pn, model in sim_context.models.items():
            footprint = model.get_footprint()
            shapely_viz.add_shape(footprint, color=config_dict["agents"][pn]["color"], zorder=ZOrders.MODEL, alpha=0.5)

    ax = shapely_viz.ax
    ax.autoscale()
    ax.set_aspect("equal")
    plt.savefig("tmp.png", dpi=300)

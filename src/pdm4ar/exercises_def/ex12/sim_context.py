from collections import defaultdict
from decimal import Decimal as D
from pathlib import Path
from typing import Dict, Optional, Mapping
import numpy as np

from commonroad.prediction.prediction import TrajectoryPrediction
from dg_commons import PlayerName
from dg_commons.perception.sensor import VisRangeSensor
from dg_commons.sim import SimParameters, SimTime, SimModel, SimLog, logger
from dg_commons.sim.goals import RefLaneGoal
from dg_commons.maps.lanes import DgLanelet
from dg_commons.sim.scenarios import DgScenario, load_commonroad_scenario
from dg_commons.sim.sim_perception import ObsFilter, FovObsFilter, IdObsFilter
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


__all__ = ["get_sim_contexts"]


def commonroad_scenario_to_simcontext(
    scenario_name: str,
    ego_player: PlayerName,
    ego_goal_lane_id: int,
    other_players: list[PlayerName],
    sim_param: Optional[SimParameters] = None,
    seed: int = 0,
) -> SimContext:
    """
    This function loads a CommonRoad scenario and tries to convert the dynamic
    obstacles into the Model/Agent paradigm used by the driving-game simulator.
    :param scenario_name:
    :param ego_player:
    :param other_players:
    :param sim_param:
    :param seed:
    :return:
    """
    scenarios_dir = str(Path(__file__).parent)
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, scenarios_dir)
    players, models, missions = {}, {}, {}

    for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
        assert isinstance(dyn_obs.prediction, TrajectoryPrediction), "Only trajectory predictions are supported"
        p_name = PlayerName(f"P{i}")
        if p_name == ego_player:
            p_name = PlayerName("Ego")
            model: SimModel = infer_model_from_cr_dyn_obstacle(dyn_obs, color="firebrick")
            agent = Pdm4arAgent()
            goal_lane = scenario.lanelet_network.find_lanelet_by_id(ego_goal_lane_id)
            dglane = DgLanelet.from_commonroad_lanelet(goal_lane)
        elif p_name in other_players:
            model: SimModel = infer_model_from_cr_dyn_obstacle(dyn_obs, color="royalblue")
            agent = IDMAgent()
            dglane = infer_lane_from_dyn_obs(dyn_obs=dyn_obs, network=scenario.lanelet_network)
        else:
            continue
        players.update({p_name: agent})
        models.update({p_name: model})
        goal_progress = dglane.along_lane_from_beta(len(dglane.control_points))
        goal = RefLaneGoal(dglane, goal_progress=goal_progress)
        missions.update({p_name: goal})
        logger.info(f"Managed to load {len(players)}")

    sim_param = SimParameters() if sim_param is None else sim_param

    lidar2d = VisRangeSensor(range=40)
    sensors: dict[PlayerName, ObsFilter] = defaultdict(lambda: IdObsFilter())
    sensors[PlayerName("Ego")] = FovObsFilter(lidar2d)

    return SimContext(
        dg_scenario=DgScenario(scenario, use_road_boundaries=True, road_boundaries_buffer=1.5),
        models=models,
        players=players,
        missions=missions,
        sensors=sensors,
        log=SimLog(),
        param=sim_param,
        seed=seed,
        description="Highway Driving",
    )


def get_sim_contexts(config_dict: Mapping) -> list[SimContext]:
    sim_context_all = []
    sim_param = SimParameters(
        dt=SimTime("0.05"),
        dt_commands=SimTime("0.1"),
        max_sim_time=SimTime(10),
        sim_time_after_collision=SimTime("1.0"),
    )
    seed = config_dict["seed"]
    np.random.seed(seed)
    for test_config in config_dict["testcases"]:
        sim_context: SimContext = commonroad_scenario_to_simcontext(
            scenario_name=test_config["scenario"],
            ego_player=test_config["ego"]["original_name"],
            ego_goal_lane_id=test_config["ego"]["goal_lane_id"],
            other_players=test_config["others"]["names"],
            sim_param=sim_param,
            seed=seed,
        )
        sim_context_all.append(sim_context)

    return sim_context_all


if __name__ == "__main__":
    # matplotlib.use('TkAgg')
    # from .ex12 import load_config_ex12
    from pdm4ar.exercises_def.ex12.ex12 import load_config_ex12

    config_path = Path(__file__).parent / "config_1.yaml"
    config_dict = load_config_ex12(config_path)
    sim_contexts = get_sim_contexts(config_dict)
    fig, axes = plt.subplots(len(sim_contexts), 1, figsize=(10, 10))
    axes = [axes] if len(sim_contexts) == 1 else axes
    for idx, sim_context in enumerate(sim_contexts):
        sim_renderer = SimRenderer(sim_context, ax=axes[idx])
        shapely_viz = sim_renderer.shapely_viz
        ax = sim_renderer.commonroad_renderer.ax

        with sim_renderer.plot_arena(ax):
            for pn, goal in sim_context.missions.items():
                shapely_viz.add_shape(
                    goal.get_plottable_geometry(),
                    color=sim_context.models[pn].model_geometry.color,
                    zorder=ZOrders.GOAL,
                    alpha=0.5,
                )
            for pn, model in sim_context.models.items():
                footprint = model.get_footprint()
                shapely_viz.add_shape(
                    footprint, color=sim_context.models[pn].model_geometry.color, zorder=ZOrders.MODEL, alpha=0.5
                )

        ax = shapely_viz.ax
        ax.autoscale()
        ax.set_aspect("equal")
    plt.savefig("tmp.png", dpi=300)

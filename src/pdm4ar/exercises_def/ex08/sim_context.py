from collections import defaultdict
from copy import deepcopy
from decimal import Decimal as D
from pathlib import Path
from typing import Dict, Optional, Mapping

from dg_commons import PlayerName
from dg_commons.maps.road_bounds import build_road_boundary_obstacle
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.perception.sensor import VisRangeSensor
from dg_commons.sim import SimParameters
from dg_commons.sim.goals import PolygonGoal, PlanningGoal
from dg_commons.sim.models.vehicle import VehicleModel, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.scenarios import DgScenario
from dg_commons.sim.sim_perception import ObsFilter, FovObsFilter
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_visualisation import ZOrders
from matplotlib import pyplot as plt

from pdm4ar.exercises.ex08.agent import Pdm4arAgent
from pdm4ar.exercises_def.ex08.scenario import get_dgscenario

__all__ = ["get_sim_context"]


def _get_empty_sim_context(scenario: DgScenario) -> SimContext:
    # sensing
    lidar2d = VisRangeSensor(range=30)
    sensors: Dict[PlayerName, ObsFilter] = defaultdict(lambda: FovObsFilter(lidar2d))
    # sim parameters
    sim_params = SimParameters(dt=D("0.01"),
                               dt_commands=D("0.1"),
                               sim_time_after_collision=D(1),
                               max_sim_time=D(50))

    return SimContext(
            dg_scenario=scenario,
            models={},
            players={},
            sensors=sensors,
            param=sim_params)


def get_sim_context(config_dict: Mapping, seed: Optional[int] = None, config_name: str = "") -> SimContext:
    dgscenario = get_dgscenario(config_dict, seed)
    simcontext = _get_empty_sim_context(dgscenario)
    simcontext.description = f"Environment-{config_name}"

    #
    _, gates = build_road_boundary_obstacle(simcontext.dg_scenario.scenario)

    # add embodied clones of the nominal agent
    agents_dict = config_dict["agents"]
    for pn in agents_dict.keys():
        player_name = PlayerName(pn)
        x0 = VehicleState(**agents_dict[pn]["state"])
        goal_n = agents_dict[pn]["goal"]
        goal = PolygonGoal(gates[goal_n].buffer(1))
        color = agents_dict[pn]["color"]
        _add_player(simcontext, x0, player_name, goal=goal, color=color)
    return simcontext


def _add_player(simcontext: SimContext, x0: VehicleState, new_name: PlayerName, goal: PlanningGoal,
                color: str = "royalblue"):
    model = VehicleModel(x0=x0, vg=VehicleGeometry.default_car(color=color), vp=VehicleParameters.default_car())

    new_models: Dict[PlayerName, VehicleModel] = {new_name: model}
    new_players = {new_name: Pdm4arAgent(
            sg=deepcopy(model.model_geometry),
            sp=deepcopy(model.model_params)
    )
    }
    # update
    simcontext.models.update(new_models)
    simcontext.players.update(new_players)
    simcontext.missions[new_name] = goal
    return


if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    from ex08 import load_config_ex08

    config_dict = load_config_ex08(Path("config_4.yaml"))
    sim_context = get_sim_context(config_dict, seed=98)
    ax = plt.gca()
    shapely_viz = ShapelyViz(ax)

    for s_obstacle in sim_context.dg_scenario.static_obstacles.values():
        shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
    for pn, goal in sim_context.missions.items():
        shapely_viz.add_shape(goal.get_plottable_geometry(), color=config_dict[pn]["color"], zorder=ZOrders.GOAL,
                              alpha=0.5)
    for pn, model in sim_context.models.items():
        footprint = model.get_footprint()
        shapely_viz.add_shape(footprint, color=config_dict[pn]["color"], zorder=ZOrders.GOAL, alpha=0.5)

    ax = shapely_viz.ax
    ax.autoscale()
    ax.set_aspect("equal")
    plt.savefig("tmp.png", dpi=300)

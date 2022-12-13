from collections import defaultdict
from copy import deepcopy
from decimal import Decimal as D
from typing import Dict

from dg_commons.maps.road_bounds import build_road_boundary_obstacle
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.perception.sensor import VisRangeSensor
from dg_commons.sim import SimParameters
from dg_commons.sim.goals import PolygonGoal
from dg_commons.sim.models.vehicle import VehicleModel, VehicleState
from dg_commons.sim.sim_perception import ObsFilter, FovObsFilter
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_visualisation import ZOrders
from matplotlib import pyplot as plt

from pdm4ar.exercises.ex08.agent import Pdm4arAgent
from pdm4ar.exercises_def.ex08.scenario import *

__all__ = ["get_sim_context"]


def _get_empty_sim_context(scenario: DgScenario) -> SimContext:
    # sensing
    lidar2d = VisRangeSensor(range=40)
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


def get_sim_context(seed: Optional[int] = None, number_of_clones: int = 0) -> SimContext:
    dgscenario = get_dgscenario(seed)
    simcontext = _get_empty_sim_context(dgscenario)
    simcontext.description = f"Environment-{number_of_clones}-clones"

    #
    _, gates = build_road_boundary_obstacle(simcontext.dg_scenario.scenario)
    # goal_poly = random.sample(gates, 1)[0]
    # goals: Goals = fd({PDM4AR_1: PolygonGoal(goal_poly.buffer(1.5))})

    # add embodied clones of the nominal agent
    x0 = VehicleState(x=-20, y=-40, psi=deg2rad(65), vx=2, delta=-0.02)
    goal_1 = PolygonGoal(gates[16].buffer(1))
    _add_player(simcontext, x0, PDM4AR_1, goal=goal_1)
    if number_of_clones >= 1:
        x0 = VehicleState(x=40, y=0, psi=deg2rad(150), vx=3, delta=-0.02)
        goal_2 = PolygonGoal(gates[7].buffer(1))
        _add_player(simcontext, x0, PDM4AR_2, goal=goal_2)
    if number_of_clones >= 2:
        x0 = VehicleState(x=-40, y=30, psi=deg2rad(-30), vx=2, delta=+0.02)
        goal_3 = PolygonGoal(gates[12].buffer(1))
        _add_player(simcontext, x0, PDM4AR_3, goal=goal_3)
    if number_of_clones >= 3:
        x0 = VehicleState(x=20, y=65, psi=deg2rad(-120), vx=4, delta=0)
        goal_4 = PolygonGoal(gates[15].buffer(1))
        _add_player(simcontext, x0, PDM4AR_4, goal=goal_4)

    return simcontext


def _add_player(simcontext: SimContext, x0: VehicleState, new_name: PlayerName, goal: PlanningGoal):
    model = VehicleModel.default_car(x0)
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
    sim_context = get_sim_context(seed=98, number_of_clones=3)
    ax = plt.gca()
    shapely_viz = ShapelyViz(ax)

    for s_obstacle in sim_context.dg_scenario.static_obstacles.values():
        shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
    for pn, goal in sim_context.missions.items():
        shapely_viz.add_shape(goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
    for pn, model in sim_context.models.items():
        footprint = model.get_footprint()
        shapely_viz.add_shape(footprint, color="blue", zorder=ZOrders.GOAL, alpha=0.5)

    ax = shapely_viz.ax
    ax.autoscale()
    ax.set_aspect("equal")
    plt.savefig("tmp.png", dpi=300)

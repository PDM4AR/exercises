from copy import deepcopy
from decimal import Decimal as D
from typing import Optional

from dg_commons import PlayerName, DgSampledSequence
from dg_commons.maps.shapes_generator import create_random_starshaped_polygon
from dg_commons.planning import PlanningGoal
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.obstacles import ObstacleGeometry, DynObstacleParameters
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel, DynObstacleState, DynObstacleCommands
from dg_commons.sim.models.vehicle import VehicleModel, VehicleState
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext
from numpy import deg2rad
from shapely.geometry import Polygon

from pdm4ar.exercises.ex08.agent import Pdm4arAgent
from pdm4ar.exercises_def.ex08.scenario import get_dgscenario

__all__ = ["get_sim_context_static", "get_sim_context_dynamic"]

PDM4AR = PlayerName("PDM4AR")


def _get_sim_context_static(scenario: DgScenario, goal: PlanningGoal, x0: VehicleState) -> SimContext:
    model = VehicleModel.default_car(x0)
    models = {PDM4AR: model}
    missions = {PDM4AR: goal}
    players = {PDM4AR: Pdm4arAgent(
            goal=goal,
            static_obstacles=deepcopy(list(scenario.static_obstacles.values())),
            sg=deepcopy(model.get_geometry()),
            sp=deepcopy(model.sp))
    }

    sim_params = SimParameters(dt=D("0.01"),
                               dt_commands=D("0.1"),
                               sim_time_after_collision=D(1),
                               max_sim_time=D(50))
    return SimContext(
            dg_scenario=scenario,
            models=models,
            players=players,
            missions=missions,
            param=sim_params)


def get_sim_context_static(seed: Optional[int] = None) -> SimContext:
    dgscenario, goal, x0 = get_dgscenario(seed)
    simcontext = _get_sim_context_static(dgscenario, goal, x0)
    simcontext.description = "static-environment"
    return simcontext


def get_sim_context_dynamic(seed: Optional[int] = None) -> SimContext:
    dgscenario, goal, x0 = get_dgscenario(seed)
    simcontext = _get_sim_context_static(dgscenario, goal, x0)
    simcontext.description = "dynamic-environment"
    # add a couple of dynamic obstacles to the environment
    DObs1 = PlayerName("DObs1")
    poly = Polygon(create_random_starshaped_polygon(0, 0, 3, 0.2, 0.2, 6))
    x0_dobs1: DynObstacleState = DynObstacleState(x=60, y=80, psi=deg2rad(-70), vx=6, vy=0, dpsi=0)
    og_dobs1: ObstacleGeometry = ObstacleGeometry(m=1000, Iz=1000, e=0.2)
    op_dops1: DynObstacleParameters = DynObstacleParameters(vx_limits=(-10, 10), acc_limits=(-1, 1))

    models = {DObs1: DynObstacleModel(x0_dobs1, poly, og_dobs1, op_dops1)}
    dyn_obstacle_commands = DgSampledSequence[DynObstacleCommands](
            timestamps=[0],
            values=[DynObstacleCommands(acc_x=0, acc_y=0, acc_psi=0)],
    )
    players = {DObs1: NPAgent(dyn_obstacle_commands)}
    simcontext.models.update(models)
    simcontext.players.update(players)

    return simcontext

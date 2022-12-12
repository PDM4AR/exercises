from collections import defaultdict
from copy import deepcopy
from decimal import Decimal as D
from typing import Dict

from dg_commons import DgSampledSequence
from dg_commons.perception.sensor import VisRangeSensor
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.obstacles import ObstacleGeometry, DynObstacleParameters
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel, DynObstacleState, DynObstacleCommands
from dg_commons.sim.models.vehicle import VehicleModel
from dg_commons.sim.sim_perception import ObsFilter, IdObsFilter, FovObsFilter
from dg_commons.sim.simulator import SimContext

from pdm4ar.exercises.ex08.agent import Pdm4arAgent
from pdm4ar.exercises_def.ex08.scenario import *

__all__ = ["get_sim_context_static", "get_sim_context_dynamic"]


def _get_sim_context_static(scenario: DgScenario, goals: Goals, x0: VehicleState) -> SimContext:
    model = VehicleModel.default_car(x0)
    models: Dict[PlayerName, VehicleModel] = {PDM4AR_1: model}
    # missions = goals
    players = {PDM4AR_1: Pdm4arAgent(
            sg=deepcopy(model.model_geometry),
            sp=deepcopy(model.model_params)
    )
    }
    # sensing
    lidar2d = VisRangeSensor(range=40)
    sensors: Dict[PlayerName, ObsFilter] = defaultdict(lambda: IdObsFilter())
    sensors[PDM4AR_1] = FovObsFilter(lidar2d)
    # sim parameters
    sim_params = SimParameters(dt=D("0.01"),
                               dt_commands=D("0.1"),
                               sim_time_after_collision=D(1),
                               max_sim_time=D(50))

    return SimContext(
            dg_scenario=scenario,
            models=models,
            players=players,
            missions=goals,
            sensors=sensors,
            param=sim_params)


def get_sim_context_static(seed: Optional[int] = None) -> SimContext:
    dgscenario, goals, x0 = get_dgscenario(seed)
    simcontext = _get_sim_context_static(dgscenario, goals, x0)
    simcontext.description = "static-environment"
    return simcontext


def get_sim_context_dynamic(seed: Optional[int] = None) -> SimContext:
    dgscenario, goal, x0 = get_dgscenario(seed)
    simcontext = _get_sim_context_static(dgscenario, goal, x0)
    simcontext.description = "dynamic-environment"

    # add a couple of non-reactive agents to the environment
    DObs1 = PlayerName("DObs1")
    poly = Polygon(create_random_starshaped_polygon(0, 0, 2, 0.1, 0.1, 6))
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

    # add embodied clones of the nominal agent
    # todo
    return simcontext

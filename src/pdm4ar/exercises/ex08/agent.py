import random
from dataclasses import dataclass
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def __init__(self,
                 sg: VehicleGeometry,
                 sp: VehicleParameters
                 ):
        self.sg = sg
        self.sp = sp
        self.name: PlayerName = None
        self.goal: PlanningGoal = None
        self.lanelet_network: LaneletNetwork = None
        self.static_obstacles: Sequence[StaticObstacle] = None
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator at the beginning of each episode."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.static_obstacles = list(init_obs.dg_scenario.static_obstacles.values())

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """ This method is called by the simulator at each time step.
        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """

        # todo implement here some better planning
        rnd_acc = random.random() * self.params.param1
        rnd_ddelta = (random.random()-.5) * self.params.param1

        return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)

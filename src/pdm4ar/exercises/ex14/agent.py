import random
from dataclasses import dataclass
from typing import Sequence, Optional, cast, Mapping

from dg_commons import PlayerName
from dg_commons.sim import SimObservations, InitSimGlobalObservations, SharedGoalObservation
from dg_commons.sim.agents import Agent, GlobalPlanner
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.diff_drive import DiffDriveCommands
from dg_commons.sim.models.diff_drive_structures import DiffDriveGeometry, DiffDriveParameters
from dg_commons.sim.models.obstacles import StaticObstacle


@dataclass(frozen=True)
class Pdm4arGlobalPlannerParams:
    param1: float = 10


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 10


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: DiffDriveGeometry
    sp: DiffDriveParameters

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: DiffDriveState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """

        # todo implement here your planning stack
        omega1 = random.random() * self.params.param1
        omega2 = random.random() * self.params.param1

        return DiffDriveCommands(omega_l=omega1, omega_r=omega2)


class Pdm4arGlobalPlanner(GlobalPlanner):
    """
    This is the Global Planner for PDM4AR
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task
    """

    players: Mapping[PlayerName, Pdm4arAgent]
    static_obstacles: Optional[Sequence[StaticObstacle]]
    shared_goals: Optional[Mapping[str, SharedGoalObservation]] = None

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arGlobalPlannerParams()

    def on_episode_init(self, init_sim_obs: InitSimGlobalObservations, players: Mapping[PlayerName, Pdm4arAgent]):
        self.players = players
        for player_name, player in self.players.items():
            player.name = player_name
            player.goal = init_sim_obs.players_obs[player_name].goal
            player.static_obstacles = list(init_sim_obs.dg_scenario.static_obstacles)
            player.sg = cast(DiffDriveGeometry, init_sim_obs.players_obs[player_name].model_geometry)
            player.sp = cast(DiffDriveParameters, init_sim_obs.players_obs[player_name].model_params)

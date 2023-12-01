from dataclasses import dataclass
from typing import Sequence
import numpy as np

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.rocket import RocketCommands, RocketState
from dg_commons.sim.models.rocket_structures import RocketGeometry, RocketParameters

from .planner import RocketPlanner


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class RocketAgent(Agent):
    "Do not modify this class name"
    cmds_plan: DgSampledSequence[RocketCommands]
    state_traj: DgSampledSequence[RocketState]
    myname: PlayerName
    planner: RocketPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: RocketGeometry
    sp: RocketParameters

    def __init__(self, satellites, planets, params: Pdm4arAgentParams):
        self.satellites = satellites
        self.planets =  planets
        self.params = params # modify

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This is the PDM4AR agent.
        Do *NOT* modify the naming of the existing methods and the input/output types.
        Feel free to add additional methods, objects and functions that help you to solve the task
        """
        self.myname = init_sim_obs.my_name
        self.planner = RocketPlanner()
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
       
        # given x0
        # either given goal.x0_target
        # or given goal

        # agent = Rocket(x0=x0, x1=x1)
        # planets = planets_test_case(case=TEST_CASE, agent=agent)
        # map = Map(planets=planets)
        # problem = SCProblem(agent=agent, map=map)
        
        # compute a plan
        self.cmds_plan, self.state_traj = self.planner.compute_trajectory()

    def get_commands(self, sim_obs: SimObservations) -> RocketCommands:
        """
        This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        expected_state = self.state_traj.at_interp(sim_obs.time)
        # todo a possible scheme to update the plan
        # if np.linalg.norm(current_state.as_ndarray() - expected_state.as_ndarray()) > 0.1:
        #     # update initial state/time horizon for the planner
        #     self.cmds_plan, self.state_traj = self.planner.compute_trajectory()

        # ZOH
        cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        return cmds

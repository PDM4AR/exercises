from dataclasses import dataclass
from typing import Sequence
import numpy as np

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.rocket import RocketCommands, RocketState
from dg_commons.sim.models.rocket_structures import RocketGeometry, RocketParameters

from pdm4ar.exercises.ex11.planner import RocketPlanner
from pdm4ar.exercises_def.ex11.goal import RocketTarget, SatelliteTarget
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

@dataclass(frozen=True)
class Pdm4arAgentParams:
    """
    Definition space for additional agent parameters.
    """
    pos_tol: 0.5
    dir_tol: 0.5
    vel_tol: 1.0

class RocketAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """
    init_state: RocketState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[RocketCommands]
    state_traj: DgSampledSequence[RocketState]
    myname: PlayerName
    planner: RocketPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: RocketGeometry
    sp: RocketParameters

    def __init__(self, init_state: RocketState, satellites: dict[PlayerName, SatelliteParams], planets: dict[PlayerName, PlanetParams]):
        """
        Initializes the agent.
        This method is called by the simulator only at the beginning of each simulation.
        Provides the RocketAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        Feel free to add additional methods, objects and functions that help you to solve the task
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.planner = RocketPlanner(planets=self.planets, satellites=self.satellites, sg=self.sg, sp=self.sp)

        # Get goal from Targets (either moving (SatelliteTarget) or static (RocketTarget))
        if isinstance(init_sim_obs.goal, SatelliteTarget):
            self.goal_state = init_sim_obs.goal.get_target_state_at(0.0)
        elif isinstance(init_sim_obs.goal, RocketTarget):
            self.goal_state = init_sim_obs.goal.target

        #
        # TODO: Implement Compute Trajectory
        #

        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)        

    def get_commands(self, sim_obs: SimObservations) -> RocketCommands:
        """
        This is called by the simulator at every time step. (0.1 sec)
        Do not modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        expected_state = self.state_traj.at_interp(sim_obs.time)

        #
        # TODO: Implement scheme to replan
        #
        
        # ZeroOrderHold
        cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # FirstOrderHold
        # cmds = self.cmds_plan.at_interp(sim_obs.time)

        return cmds
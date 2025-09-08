from dataclasses import dataclass
from typing import Sequence

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters

from pdm4ar.exercises.ex13.planner import SpaceshipPlanner
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, SatelliteParams

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Config:
    PLOT = True
    VERBOSE = False


def plot_traj(computed: DgSampledSequence, actual: list = None):
    """
    Example of simple plotting function to help you debug your code.
    Feel free to modify it or create your own plotting functions.
    Note that the plot is overwritten at each call which means that only the last simulation is saved.
    """

    timestamps = list(computed._timestamps)  # sequence.get_sampling_points()
    values = list(computed._values)

    df = pd.DataFrame(values)

    plt.plot(df["x"], df["y"], label="Computed Trajectory")
    for i in range(len(df)):
        plt.arrow(
            df["x"][i],
            df["y"][i],
            np.cos(df["psi"][i]),
            np.sin(df["psi"][i]),
            head_width=0.1,
            head_length=0.1,
            fc="k",
            ec="k",
        )

    if actual is not None:
        actual_positions = np.array([[state.x, state.y] for state in actual])
        actual_orientations = np.array([state.psi for state in actual])
        plt.scatter(actual_positions[:, 0], actual_positions[:, 1], label="Actual Trajectory")
        for i in range(len(actual_positions)):
            plt.arrow(
                actual_positions[i, 0],
                actual_positions[i, 1],
                np.cos(actual_orientations[i]),
                np.sin(actual_orientations[i]),
                head_width=0.1,
                head_length=0.05,
                fc="r",
                ec="r",
            )

    plt.grid(True)
    plt.legend()
    file_path = "src/pdm4ar/exercises/ex13/final_traj.png"  # feel free to change path
    plt.savefig(file_path)
    plt.close()


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    my_tol: float = 0.1


class SpaceshipAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SpaceshipState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SpaceshipCommands]
    state_traj: DgSampledSequence[SpaceshipState]
    myname: PlayerName
    planner: SpaceshipPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    def __init__(
        self,
        init_state: SpaceshipState,
        satellites: dict[PlayerName, SatelliteParams],
        planets: dict[PlayerName, PlanetParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SpaceshipAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.actual_trajectory = []
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.

        the time spent in this method is **not** considered in the score.
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.planner = SpaceshipPlanner(planets=self.planets, satellites=self.satellites, sg=self.sg, sp=self.sp)
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        # make sure you consider both types of goals accordingly
        # (Docking is a subclass of SpaceshipTarget and may require special handling
        # to take into account the docking structure)
        self.goal_state = init_sim_obs.goal.target

        #
        # TODO: Implement Compute Initial Trajectory
        #

        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)

    def get_commands(self, sim_obs: SimObservations) -> SpaceshipCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)

        NOTE: this function is not run in real time meaning that the simulation is stopped when the function is called.
        Thus the time efficiency of the replanning is not critical for the simulation.
        However the average time spent in get_commands is still considered in the score.

        Do **not** modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        self.actual_trajectory.append(current_state)
        expected_state = self.state_traj.at_interp(sim_obs.time)

        if Config.PLOT and int(10 * sim_obs.time) % 25 == 0:
            plot_traj(self.state_traj, self.actual_trajectory)

        #
        # TODO: Implement scheme to replan
        #

        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # FirstOrderHold
        cmds = self.cmds_plan.at_interp(sim_obs.time)

        return cmds

from dataclasses import dataclass
from re import S
from typing import Sequence
import numpy as np
from IPython import embed

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters

from pdm4ar.exercises.ex13.planner import SatellitePlanner
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, SatelliteParams, AsteroidParams
from pdm4ar.exercises_def.ex13.utils_plot import plot_traj


# HINT: as a good practice we suggest to use the config class to centralise activation of the debugging options
class Config:
    PLOT = True
    VERBOSE = False


@dataclass(frozen=True)
class Pdm4arAgentParams:
    """
    Definition space for additional agent parameters.
    """

    pos_tol: 0.5
    dir_tol: 0.5
    vel_tol: 1.0


class SatelliteAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SatelliteState
    satellites: dict[PlayerName, SatelliteParams]
    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SatelliteCommands]
    state_traj: DgSampledSequence[SatelliteState]
    myname: PlayerName
    planner: SatellitePlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SatelliteGeometry
    sp: SatelliteParameters

    def __init__(
        self,
        init_state: SatelliteState,
        satellites: dict[PlayerName, SatelliteParams],
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SatelliteAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.actual_trajectory = []
        self.init_state = init_state
        self.satellites = satellites
        self.planets = planets
        self.asteroids = asteroids

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
        self.planner = SatellitePlanner(planets=self.planets, satellites=self.satellites, sg=self.sg, sp=self.sp)

        # Get goal from Targets (either moving (SatelliteTarget) or static (SpaceshipTarget))
        # if isinstance(init_sim_obs.goal, SatelliteTarget):
        #     self.goal_state = init_sim_obs.goal.get_target_state_at(0.0)
        # elif isinstance(init_sim_obs.goal, SpaceshipTarget):
        #     self.goal_state = init_sim_obs.goal.target

        #
        # TODO: Implement Compute Trajectory
        #
        self.goal = init_sim_obs.goal
        goal = init_sim_obs.goal
        start = self.init_state

        K = 50

        X = np.zeros((6, K))
        U = np.zeros((2, K))
        p = np.zeros(1)

        # isinstance(goal, DockingTarget):
        #     phi = goal.target.as_ndarray()[3]
        #     goal_state = np.concatenate([goal.target.as_ndarray()-, np.array([0, start_state[7]])], axis=0)

        feasible = False
        i = 0
        while not feasible and i < 20:
            p[0] = 10 + i
            start_state = start.as_ndarray()

            # if isinstance(goal, SatelliteTarget):
            #     # Forcast goal position at time p
            #     goal_state = np.concatenate(
            #         [
            #             goal.get_target_state_at(p[0]).as_ndarray(),
            #             np.array([0, start_state[7]]),
            #         ],
            #         axis=0,
            #     )
            if isinstance(goal, SpaceshipTarget):
                goal_state = np.concatenate([goal.target.as_ndarray(), np.array([0, start_state[5]])], axis=0)

            for k in range(K):
                X[:, k] = (1 - (k / (K - 1))) * start_state  # + (k / (K - 1)) * goal_state

            U[0, :] = (self.sp.F_limits[1]) / 2
            U[1, :] = (self.sp.F_limits[1]) / 2

            U[:, 0] = 0
            U[:, -1] = 0

            self.cmds_plan, self.state_traj, feasible = self.planner.compute_trajectory(
                self.init_state, init_sim_obs.goal, X, U, p
            )
            i += 1
        if feasible:
            traj = [state[1].as_ndarray()[0:3] for state in self.state_traj]
            self.plot_trajectory(traj)

    def plot_trajectory(self, traj):
        import matplotlib.pyplot as plt

        dock = False

        # Define the environment boundaries

        x_min, x_max = -13, 26

        y_min, y_max = -13, 13

        # Plot the trajectory

        plt.figure(figsize=(6, 6))

        # plt.plot(*zip(*traj), marker="o", color="b", linestyle="-", linewidth=1.5, markersize=5)
        print(f"len traj = {len(traj)}")

        for i, (x, y, theta) in enumerate(traj):
            # Compute line endpoints for the orientation line
            line_length = 1.3  # Length of the orientation line
            dx = line_length * np.cos(theta)
            dy = line_length * np.sin(theta)
            plt.plot([x, x + dx], [y, y + dy], color="r", linewidth=1.5)  # Draw orientation line
            if i >= len(traj) - 3:
                plt.plot(
                    x,
                    y,
                    "go",
                )  # Draw the point itself
            else:
                plt.plot(
                    x,
                    y,
                    "bo",
                )  # Draw the point itself
        if dock:
            x = np.linspace(-11, 25, 50)
            y = np.linspace(-11, 11, 50)

            # Create a meshgrid
            X, Y = np.meshgrid(x, y)

            # Stack into an array of shape (2500, 2) with all (x, y) pairs
            points = np.vstack([X.ravel(), Y.ravel()]).T
            np.savetxt("/workspaces/exercises/out/11/index.html_resources/Al.txt", self.planner.Al, fmt="%d")
            np.savetxt("/workspaces/exercises/out/11/index.html_resources/bl.txt", self.planner.bl, fmt="%d")

            bl_expanded = self.planner.bl[:, np.newaxis]
            print(f"boolean shape {((self.planner.Al[0,:] @ points.T <= bl_expanded[0]).T).shape}")
            filter = (np.all(self.planner.Al @ points.T <= bl_expanded, axis=0)).reshape((points.shape[0], 1))
            bool_filter = np.hstack((filter, filter))
            points = points[bool_filter]
            points = points.reshape(int(points.shape[0] / 2), 2)
            print(f"points dim {points.shape}")
            plt.scatter(points[:, 0], points[:, 1], color="r", alpha=0.3)

            self.plot_circle(5.0, 4.0, 2.5)
            self.plot_circle(-5.0, -4.0, 1.6)
            self.plot_circle(0.0, 0.0, 2.0)

        # Set plot boundaries

        plt.xlim(x_min, x_max)

        plt.ylim(y_min, y_max)

        # Labeling the axes and setting grid

        plt.xlabel("X")

        plt.ylabel("Y")

        plt.title("2D Trajectory in a Square Environment")

        plt.grid(True)

        # Display the plot

        plt.savefig("/workspaces/exercises/out/11/index.html_resources/traj.jpg")

    def plot_circle(self, cx, cy, radius):
        import matplotlib.pyplot as plt

        theta = np.linspace(0, 2 * np.pi, 100)  # 100 points around the circle
        x = radius * np.cos(theta) + cx  # x = r * cos(theta)
        y = radius * np.sin(theta) + cy  # y = r * sin(theta)

        plt.plot(x, y)  # Plot the circle

    def get_commands(self, sim_obs: SimObservations) -> SatelliteCommands:
        """
        This is called by the simulator at every time step. (0.1 sec)
        Do not modify the signature of this method.
        """
        print(f"getting commands at time {sim_obs.time:.2f}")
        # current_state = sim_obs.players[self.myname].state
        # self.actual_trajectory.append(current_state)
        # expected_state = self.state_traj.at_interp(sim_obs.time)

        # if Config.PLOT and int(10 * sim_obs.time) % 25 == 0:
        #     plot_traj(self.state_traj, self.actual_trajectory)

        # start = current_state
        # goal = self.goal

        # K = 50

        # X = np.zeros((6, K))
        # U = np.zeros((2, K))
        # p = np.zeros(1)

        # if float(sim_obs.time) >= float(self.state_traj.get_end()):
        #     feasible = False
        #     i = 1
        #     while not feasible and i < 10:
        #         p[0] = 9 + i
        #         start_state = start.as_ndarray()

        #         # if isinstance(goal, SatelliteTarget):
        #         #     # Forcast goal position at time p
        #         #     goal_state = np.concatenate(
        #         #         [
        #         #             goal.get_target_state_at(p[0]).as_ndarray(),
        #         #             np.array([0, start_state[7]]),
        #         #         ],
        #         #         axis=0,
        #         #     )
        #         if isinstance(goal, SpaceshipTarget):
        #             goal_state = np.concatenate([goal.target.as_ndarray(), np.array([0, start_state[5]])], axis=0)

        #         for k in range(K):
        #             X[:, k] = (1 - (k / (K - 1))) * start_state  # + (k / (K - 1)) * goal_state

        #         U[0, :] = (self.sp.F_limits[1]) / 2
        #         U[1, :] = (self.sp.F_limits[1]) / 2

        #         U[:, 0] = 0
        #         U[:, -1] = 0

        #         (
        #             self.cmds_plan,
        #             self.state_traj,
        #             feasible,
        #         ) = self.planner.compute_trajectory(start, goal, X, U, p)
        #         i += 1

        # else:
        #     REPLAN_THRESH = 1
        #     POS_THRESH = 0.3
        #     VEL_THRESH = 0.3
        #     DIR_THRESH = 0.5
        #     if np.linalg.norm(current_state.as_ndarray() - expected_state.as_ndarray(), ord=2) > REPLAN_THRESH:
        #         X_init = np.zeros((8, K))
        #         U_init = np.zeros((2, K))
        #         p_bar = np.zeros(1)

        #         old_p = self.cmds_plan.get_end() - float(sim_obs.time)

        #         state_traj = self.state_traj.get_subsequence(sim_obs.time, self.state_traj.get_end()).shift_timestamps(
        #             -float(sim_obs.time)
        #         )
        #         old_X = np.zeros_like(X_init)
        #         for k in range(K):
        #             f = k / (K - 1)
        #             old_X[:, k] = state_traj.at_interp(f * p_bar).as_ndarray()
        #         cmds_plan = self.cmds_plan.get_subsequence(sim_obs.time, self.cmds_plan.get_end()).shift_timestamps(
        #             -float(sim_obs.time)
        #         )
        #         old_U = np.zeros_like(U_init)
        #         for k in range(K):
        #             f = k / (K - 1)
        #             old_U[:, k] = cmds_plan.at_interp(f * p_bar).as_ndarray()

        #         X_init[:, 1:] = old_X[:, 1:]
        #         X_init[:, 0] = current_state.as_ndarray()
        #         U_init = old_U
        #         p_init = np.array([old_p])

        #         (
        #             self.cmds_plan,
        #             self.state_traj,
        #             feasible,
        #         ) = self.planner.compute_trajectory(current_state, self.goal, X_init, U_init, p_init)
        #         t0 = float(sim_obs.time)
        #         self.cmds_plan = self.cmds_plan.shift_timestamps(t0)
        #         self.state_traj = self.state_traj.shift_timestamps(t0)
        #         # ***** DEBUG *****
        #         print("New state trajectory")
        #         print(self.state_traj.at_interp(sim_obs.time))
        #         # *****************

        # #
        # # TODO: Implement scheme to replan
        # #

        # # ZeroOrderHold
        # # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # # FirstOrderHold
        cmds = self.cmds_plan.at_interp(sim_obs.time)

        return cmds

from dataclasses import dataclass
import os
from re import S
from typing import List, Sequence
from cvxpy import pos
import numpy as np
from IPython import embed

# for plot_LQR
import matplotlib.pyplot as plt
import os
from pdm4ar.exercises_def.structures import out_dir

from shapely.geometry import Polygon, LineString, Point

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
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams
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

    pos_tol: float = 0.4
    dir_tol: float = 0.4
    vel_tol: float = 0.4
    max_replannings: int = int(os.getenv("EX13_MAX_REPLANNINGS", "1"))
    correction_thrsh: float = 0.4


class SatelliteAgent(Agent):
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SatelliteState
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
    ap: Pdm4arAgentParams

    def __init__(
        self,
        init_state: SatelliteState,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SatelliteAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.actual_trajectory = []
        self.norm2_errors = []
        self.max_errors = []
        self.LQR_components = []
        self.command_times = []
        self.replanning_times = []
        self.planned_trajectories = []
        self.ap = Pdm4arAgentParams()
        self.init_state = init_state
        self.planets = planets
        self.asteroids = asteroids
        self.nb_replannings = 0

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

        # Get borders
        assert init_sim_obs.dg_scenario is not None
        map_borders = self.get_border_coordinates(init_sim_obs.dg_scenario.static_obstacles)

        # Get goal
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        self.target_type = type(init_sim_obs.goal)
        self.goal_state = init_sim_obs.goal.target

        landing_constraints_points = {}
        if isinstance(init_sim_obs.goal, DockingTarget):
            A, B, C, A1, A2, p = init_sim_obs.goal.get_landing_constraint_points()
            print(f"A: {A}, B: {B}, C: {C}, A1: {A1}, A2: {A2}, p: {p}")
            landing_constraints_points = {"A": A, "B": B, "C": C, "A1": A1, "A2": A2, "p": p}

        self.planner = SatellitePlanner(
            sg=self.sg,
            sp=self.sp,
            map_borders=map_borders,
            init_state=self.init_state,
            goal_state=self.goal_state,
            lc=landing_constraints_points,
            planets=self.planets,
            asteroids=self.asteroids,
        )

        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)
        self.planned_trajectories.append(self.state_traj)

    def get_commands(self, sim_obs: SimObservations) -> SatelliteCommands:
        """
        This is called by the simulator at every time step. (0.1 sec)
        Do not modify the signature of this method.
        """
        print(f"getting commands at time {sim_obs.time:.2f}")

        # # ZeroOrderHold
        # # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # # FirstOrderHold

        current_state = sim_obs.players[self.myname].state
        self.actual_trajectory.append(current_state)
        self.command_times.append(float(sim_obs.time))
        expected_state = self.state_traj.at_interp(sim_obs.time)

        if self.check_replan_needed(current_state, expected_state):
            self.replanning_times.append(float(sim_obs.time))
            print("Replanning...")
            init_cmd = self.cmds_plan.at_interp(sim_obs.time)
            # init_input = np.array([init_cmd.F_left, init_cmd.F_right])
            init_input = np.array([0, 0])
            self.cmds_plan, self.state_traj = self.planner.compute_trajectory(
                current_state, self.goal_state, init_input=init_input, sim_time=sim_obs.time
            )
            # The new plan starts from the current state at the current global time.
            # Do not feed the LQR a reference from the discarded trajectory.
            expected_state = self.state_traj.at_interp(sim_obs.time)
            self.planned_trajectories.append(self.state_traj)

        # print(self.cmds_plan)
        cmds = self.cmds_plan.at_interp(sim_obs.time)
        print(cmds)

        lqr_component, error = self.planner.lqr_controller(
            current_state=current_state, current_input=cmds, expected_state=expected_state
        )
        ##################### For LQR analysis and plotting #######################
        error_2norm = np.linalg.norm(error)
        max_error = np.max(np.abs(error))

        self.norm2_errors.append(error_2norm)
        self.LQR_components.append(lqr_component)
        self.max_errors.append(max_error)

        if Config.PLOT and int(10 * sim_obs.time) % 25 == 0:
            plot_traj(self.state_traj, self.actual_trajectory)
            self.plot_lqr(self.LQR_components, self.norm2_errors, self.max_errors)
        ##########################################################################

        cmds_array = np.array([cmds.F_left, cmds.F_right])

        # Add LQR component (if needed)
        if error_2norm >= self.ap.correction_thrsh:
            print(f"Applying LQR correction, error norm: {error_2norm}")
            cmds_array += lqr_component

        cmds_array = np.clip(cmds_array, self.sp.F_limits[0], self.sp.F_limits[1])

        # Convert in SatelliteCommands
        cmds_with_lqr = SatelliteCommands(F_left=cmds_array[0], F_right=cmds_array[1])

        return cmds_with_lqr

    def check_replan_needed(self, current_state: SatelliteState, expected_state: SatelliteState) -> bool:
        if self.nb_replannings >= self.ap.max_replannings:
            return False

        pos_error = np.linalg.norm(np.array([current_state.x - expected_state.x, current_state.y - expected_state.y]))
        angle_error = current_state.psi - expected_state.psi
        dir_error = abs(np.arctan2(np.sin(angle_error), np.cos(angle_error)))
        vel_error = np.linalg.norm(
            np.array([current_state.vx - expected_state.vx, current_state.vy - expected_state.vy])
        )
        pos_replan = pos_error > self.ap.pos_tol
        dir_replan = dir_error > self.ap.dir_tol
        vel_replan = vel_error > self.ap.vel_tol
        replan = pos_replan or dir_replan or vel_replan

        if pos_replan:
            print(f"Position error {pos_error:.3f} exceeds tolerance {self.ap.pos_tol}")
        if dir_replan:
            print(f"Direction error {dir_error:.3f} exceeds tolerance {self.ap.dir_tol}")
        if vel_replan:
            print(f"Velocity error {vel_error:.3f} exceeds tolerance {self.ap.vel_tol}")

        if replan:
            self.nb_replannings += 1
        return replan

    def get_border_coordinates(self, static_obstacles: Sequence[StaticObstacle]) -> dict:
        borders = None
        for obstacle in static_obstacles:
            if isinstance(obstacle.shape, LineString):
                borders = obstacle
        if borders is None:
            raise ValueError("No borders found in static obstacles")

        # Extract coordinates from the LineString
        coords = list(borders.shape.coords)

        # Determine xmin, xmax, ymin, ymax
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        xmin = min(x_coords)
        xmax = max(x_coords)
        ymin = min(y_coords)
        ymax = max(y_coords)

        dict_borders = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }

        return dict_borders

    # not in utilis_plot.py not to let students have access to LQR analysis
    def plot_lqr(self, lqr_component: List, error_2norm: List, max_error: List):
        """
        Plots LQR components and error norm over simulation time.
        Updates the same figure at each simulation step.
        """
        t = np.arange(len(lqr_component))

        lqr_component = np.array(lqr_component)
        error_2norm = np.array(error_2norm)
        max_error = np.array(max_error)

        data_top_1 = lqr_component[:, 0]
        data_top_2 = lqr_component[:, 1]
        data_middle = error_2norm
        data_bottom = max_error

        width = 0.5  # bar width

        # Create vertically stacked subplots with shared x-axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

        # ----- TOP subplot (two datasets, side-by-side) -----
        ax1.bar(t - width / 2, data_top_1, width, label="left delta-thrust")
        ax1.bar(t + width / 2, data_top_2, width, label="right delta-thrust")
        ax1.set_title("LQR Components Over Time")
        ax1.legend()
        ax1.grid(True, axis="y")

        # ----- MIDDLE subplot (single dataset) -----
        ax2.bar(t, data_middle, width, color="tab:orange", label="norm2 of state error")
        ax2.set_title("Delta state Error Norm Over Time")
        ax2.legend()
        ax2.grid(True, axis="y")

        # ----- BOTTOM subplot (single dataset) -----
        ax3.bar(t, data_bottom, width, color="tab:green", label="max error encountered at this time")
        ax3.set_xlabel("discrete time step")
        ax3.set_title("Max state Error Encountered Over Time")
        ax3.legend()
        ax3.grid(True, axis="y")

        plt.tight_layout()

        # Save figure
        output_dir = os.path.join(out_dir("13"), "index.html_resources")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "lqr_analysis.png"), dpi=300)
        plt.close()

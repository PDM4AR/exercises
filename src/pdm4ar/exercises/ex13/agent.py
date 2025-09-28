from dataclasses import dataclass
from re import S
from typing import Sequence
import numpy as np
from IPython import embed

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
            planets=self.planets,
            satellites=self.satellites,
            sg=self.sg,
            sp=self.sp,
            map_borders=map_borders,
            init_state=self.init_state,
            goal_state=self.goal_state,
            lc=landing_constraints_points,
        )

        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)

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
        expected_state = self.state_traj.at_interp(sim_obs.time)

        if Config.PLOT and int(10 * sim_obs.time) % 25 == 0:
            plot_traj(self.state_traj, self.actual_trajectory)

        cmds = self.cmds_plan.at_interp(sim_obs.time)

        lqr_component = self.planner.lqr_controller(
            current_state=current_state, current_input=cmds, expected_state=expected_state
        )
        cmds_array = np.array([cmds.F_left, cmds.F_right])
        cmds_array += lqr_component
        cmds_with_lqr = SatelliteCommands(F_left=cmds_array[0], F_right=cmds_array[1])

        return cmds_with_lqr

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

import ast
from dataclasses import dataclass, field
import glob
from typing import Any, Union
from matplotlib.dates import SA
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.seq.sequence import Sequence, Timestamp
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)
import scipy.linalg

from pdm4ar.exercises.ex13.discretization import *
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, SatelliteParams, AsteroidParams

PLOT = True


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "CLARABEL"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 10  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 0 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant

    tf_min_weight: float = 1.5  # weight for final time
    mult_padding: float = 1  # multiplicative padding for obstacles
    add_padding: float = 1.5  # additive padding for obstacles
    n_last: int = 5  # number of last points to consider for docker constraints
    weight_thrust: float = 0  # weight for thrust
    weight_ddelta: float = 0  # weight for angular velocity
    weight_thrust_rate: float = 0  # weight for thrust rate
    weight_ddelta_rate: float = 0  # weight for angular velocity rate
    thrust_limit_factor: float = 1  # factor to limit thrust


class SatellitePlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    asteroids: dict[PlayerName, AsteroidParams]
    satellite: SatelliteDyn
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: SolverParameters
    map_borders: dict[str, float]
    init_state: SatelliteState
    goal_state: DynObstacleState
    lc: dict[str, Any]

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
        map_borders: dict[str, float],
        init_state: SatelliteState,
        goal_state: DynObstacleState,
        lc: dict[str, Any] = {},
        planets: dict[PlayerName, PlanetParams] = {},
        satellites: dict[PlayerName, SatelliteParams] = {},
        asteroids: dict[PlayerName, AsteroidParams] = {},
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp

        self.map_borders = map_borders
        self.init_state = init_state
        self.goal_state = goal_state
        self.dim_pos = 2
        self.J_bar = None
        self.docking_mode = False
        self.sim_time = 0.0
        self.border_padding = (self.sg.l / 2) * 1.5
        self.n_last = 5

        self.landing_constraints_points = lc
        # if lc:
        #     mid = (lc["A1"] + lc["A2"]) / 2
        #     docker_name = PlayerName("Docker")
        #     self.planets[docker_name] = PlanetParams([mid[0], mid[1]], -0.966)

        # print(f"planets: {self.planets}")
        # print(f"LC : {self.landing_constraints_points}")
        # print(f"l_c: {self.sg.l_c}")
        # print(f"l_f: {self.sg.l_f}")
        # Solver Parameters
        self.params = SolverParameters()

        # Satellite Dynamics
        self.satellite = SatelliteDyn(self.sg, self.sp)
        # Satellite Dynamics
        self.satellite = SatelliteDyn(self.sg, self.sp)

        # Discretization Method
        self.foh = True
        if self.foh:
            self.integrator_foh = FirstOrderHold(self.satellite, self.params.K, self.params.N_sub)
        else:
            self.integrator_zoh = ZeroOrderHold(self.satellite, self.params.K, self.params.N_sub)

        # Numbers of constraints
        self.nb_ic = self.satellite.n_x
        self.nb_tc = self.satellite.n_x  # -2
        self.nb_planets = len(self.planets)
        self.nb_satellites = len(self.satellites)
        self.nb_asteroids = len(self.asteroids)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        (
            self.problem_parameters["X_bar"].value,
            self.problem_parameters["U_bar"].value,
            self.problem_parameters["p_bar"].value,
        ) = self.initial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState, init_input: NDArray = np.array([0, 0])
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.problem_parameters["tr_radius"].value = self.params.tr_radius
        self.init_state = init_state  # Only place where this is changed
        self.goal_state = goal_state  # Only place where this is changed
        self.problem_parameters["init_input"].value = init_input
        self.problem_parameters["init_state"].value = np.array(
            [
                self.init_state.x,
                self.init_state.y,
                self.init_state.psi,
                self.init_state.vx,
                self.init_state.vy,
                self.init_state.dpsi,
                # self.init_state.delta,
                # self.init_state.m,
            ]
        )

        self.problem_parameters["goal_state"].value = np.array(
            [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                self.goal_state.dpsi,
            ]
        )

        (
            self.problem_parameters["X_bar"].value,
            self.problem_parameters["U_bar"].value,
            self.problem_parameters["p_bar"].value,
        ) = self.initial_guess()

        self.J_bar = self._compute_j(
            self.problem_parameters["X_bar"].value,
            self.problem_parameters["U_bar"].value,
            self.problem_parameters["p_bar"].value,
        )

        for i in range(self.params.max_iterations):
            print("-" * 25, f"Iteration {i}", "-" * 25)
            self._convexification()
            try:
                L_star = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
            print(f"Problem status {self.problem.status}")
            if PLOT:
                self.plot_trajectory(i)
                self.plot_heatmap()
                # self.plot_linearized_obstacles(0)

            # Obtain J_star, J_bar, L_star
            J_star = self._compute_j(self.variables["X"].value, self.variables["U"].value, self.variables["p"].value)

            actual_improvement = self.J_bar - J_star
            predicted_improvement = self.J_bar - L_star  # type: ignore
            rho = actual_improvement / predicted_improvement  # type: ignore

            # L_star_debug = self._compute_l()
            print(f"    J_bar: {self.J_bar}")
            print(f"    J_star: {J_star}")
            print(f"    L_star: {L_star}")
            # print(f"    L_star_debug: {L_star_debug}")
            print("-------")
            print(f"    actual_improvement: {actual_improvement}")
            print(f"    predicted_improvement: {predicted_improvement}")
            print("-------")
            print(f"    rho: {rho}")
            print(f"    trust region: {self.problem_parameters['tr_radius'].value}")
            print("-------")
            print(f"    v_ic: {np.linalg.norm(self.variables['v_ic'].value, 1)}")
            print(f"    v_tc: {np.linalg.norm(self.variables['v_tc'].value, 1)}")
            print(f"    v: {np.linalg.norm(self.variables['v'].value.flatten(), 1)}")
            print(f"    v_s: {self.variables['v_s'].value}")

            self._update_trust_region(rho, J_star)  # type: ignore

            if predicted_improvement < self.params.stop_crit:  # type: ignore
                break

        cmd_sequence, state_sequence = self._extract_optimal_cmds_and_traj()

        return cmd_sequence, state_sequence

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K
        X = np.zeros((self.satellite.n_x, K))
        U = np.zeros((self.satellite.n_u, K))
        # U[0, 0] = self.problem_parameters["init_input"][0].value
        # U[1, 0] = self.problem_parameters["init_input"][1].value

        p = np.zeros((self.satellite.n_p))

        X[0, :] = np.linspace(
            self.problem_parameters["init_state"][0].value, self.problem_parameters["goal_state"][0].value, K
        )
        X[1, :] = np.linspace(
            self.problem_parameters["init_state"][1].value, self.problem_parameters["goal_state"][1].value, K
        )
        X[2, :] = np.linspace(
            self.problem_parameters["init_state"][2].value, self.problem_parameters["goal_state"][2].value, K
        )
        X[3, :] = np.linspace(
            self.problem_parameters["init_state"][3].value, self.problem_parameters["goal_state"][3].value, K
        )
        X[4, :] = np.linspace(
            self.problem_parameters["init_state"][4].value, self.problem_parameters["goal_state"][4].value, K
        )
        X[5, :] = np.linspace(
            self.problem_parameters["init_state"][5].value, self.problem_parameters["goal_state"][5].value, K
        )

        goal_pos = np.array(
            [self.problem_parameters["goal_state"][0].value, self.problem_parameters["goal_state"][1].value]
        )
        init_pos = np.array(
            [self.problem_parameters["init_state"][0].value, self.problem_parameters["init_state"][1].value]
        )

        p[0] = self.params.tf_min_weight * np.linalg.norm(goal_pos - init_pos) / self.sp.vx_limits[1]

        if PLOT:
            # Plot initial guess trajectory
            plt.plot(X[0], X[1], label="initial_guess")
            for j in range(self.params.K):
                plt.arrow(
                    X[0, j],
                    X[1, j],
                    1.0 * np.cos(X[2, j]),
                    1.0 * np.sin(X[2, j]),
                    color="blue",
                    head_width=0.25,
                    head_length=0.4,
                )
            plt.legend()
            plt.savefig("initial_guess.png")
            plt.close()

        return X, U, p

    def interpolate_angle(self, psi_start, psi_end, t_values):
        delta = (psi_end - psi_start) % (2 * np.pi)

        # Generate evenly spaced interpolation steps
        psi_interpolated = psi_start + t_values * delta

        # Wrap the result back to [-pi, pi]
        psi_interpolated = (psi_interpolated) % (2 * np.pi)

        return psi_interpolated

    def interpolate_with_midpoint(self, A, C, B, N):
        # Calculate the number of points in each segment
        N1 = N // 2 + 1  # Points from A to C (inclusive)
        N2 = N - N1 + 1  # Points from C to B (inclusive)

        # Interpolate points
        segment1 = np.linspace(A, C, N1, endpoint=False)  # A to just before C
        segment2 = np.linspace(C, B, N2)  # C to B

        # Combine the segments
        trajectory = np.concatenate((segment1, segment2))
        return trajectory

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.satellite.n_x, self.params.K), name="X"),
            "U": cvx.Variable((self.satellite.n_u, self.params.K), name="U"),
            "p": cvx.Variable(self.satellite.n_p, name="p"),
            "v": cvx.Variable((self.satellite.n_x, self.params.K - 1), name="v"),
            "v_ic": cvx.Variable((self.nb_ic), name="v_ic"),
            "v_tc": cvx.Variable((self.nb_tc), name="v_tc"),
            "v_s": cvx.Variable(nonneg=True, name="v_s"),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        init_state = np.array(
            [
                self.init_state.x,
                self.init_state.y,
                self.init_state.psi,
                self.init_state.vx,
                self.init_state.vy,
                self.init_state.dpsi,
            ]
        )

        goal_state = np.array(
            [
                self.goal_state.x,
                self.goal_state.y,
                self.goal_state.psi,
                self.goal_state.vx,
                self.goal_state.vy,
                self.goal_state.dpsi,
            ]
        )

        problem_parameters = {
            # Known at init
            "init_state": cvx.Parameter(self.satellite.n_x, value=init_state, name="init_state"),
            "init_input": cvx.Parameter(self.satellite.n_u, value=np.array([0, 0]), name="init_input"),
            "goal_state": cvx.Parameter(6, value=goal_state, name="goal_state"),
            "map_x": cvx.Parameter(
                2, value=np.array([self.map_borders["xmin"], self.map_borders["xmax"]]), name="map_x"
            ),
            "map_y": cvx.Parameter(
                2, value=np.array([self.map_borders["ymin"], self.map_borders["ymax"]]), name="map_y"
            ),
            "thrust_limits": cvx.Parameter(
                2, value=np.array(self.sp.F_limits) * self.params.thrust_limit_factor, name="thrust_limits"
            ),
            # "max_delta": cvx.Parameter(2, value=np.array(self.sp.delta_limits), name="max_delta"),
            # "max_ddelta": cvx.Parameter(2, value=np.array(self.sp.ddelta_limits), name="max_ddelta"),
            "tr_radius": cvx.Parameter(value=self.params.tr_radius, name="tr_radius"),
            "m_satellite": cvx.Parameter(value=self.sg.m, name="m_satellite"),
            "X_bar": cvx.Parameter((self.satellite.n_x, self.params.K), name="X_bar"),
            "U_bar": cvx.Parameter((self.satellite.n_u, self.params.K), name="U_bar"),
            "p_bar": cvx.Parameter(self.satellite.n_p, name="p_bar"),
            # Dynamics
            "A_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_x, self.params.K - 1), name="A_bar"),
            "F_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_p, self.params.K - 1), name="F_bar"),
            "r_bar": cvx.Parameter((self.satellite.n_x, self.params.K - 1), name="r_bar"),
        }

        if self.foh:
            problem_parameters["B_plus_bar"] = cvx.Parameter(
                (self.satellite.n_x * self.satellite.n_u, self.params.K - 1), name="B_plus_bar"
            )
            problem_parameters["B_minus_bar"] = cvx.Parameter(
                (self.satellite.n_x * self.satellite.n_u, self.params.K - 1), name="B_minus_bar"
            )
        else:
            problem_parameters["B_bar"] = cvx.Parameter(
                (self.satellite.n_x * self.satellite.n_u, self.params.K - 1), name="B_bar"
            )

        if self.landing_constraints_points:
            problem_parameters["docker_matrices"] = cvx.Parameter((3, 2), name="docker_matrices")
            problem_parameters["docker_b"] = cvx.Parameter(3, name="docker_b")

        # Obstacle
        if self.planets:
            problem_parameters["C_planet_bar"] = cvx.Parameter(
                (self.nb_planets * self.dim_pos, self.params.K), name="C_planet_bar"
            )
            problem_parameters["r_prime_planet_bar"] = cvx.Parameter(
                (self.nb_planets, self.params.K), name="r_prime_planet_bar"
            )
        if self.satellites:
            problem_parameters["C_sat_bar"] = cvx.Parameter(
                (self.nb_satellites * self.dim_pos, self.params.K), name="C_sat_bar"
            )
            problem_parameters["G_bar"] = cvx.Parameter((self.nb_satellites, self.params.K), name="G_bar")
            problem_parameters["r_prime_sat_bar"] = cvx.Parameter(
                (self.nb_satellites, self.params.K), name="r_prime_sat_bar"
            )
        if self.asteroids:
            problem_parameters["C_ast_bar"] = cvx.Parameter(
                (self.nb_asteroids * self.dim_pos, self.params.K), name="C_ast_bar"
            )
            problem_parameters["H_bar"] = cvx.Parameter((self.nb_asteroids, self.params.K), name="H_bar")
            problem_parameters["r_prime_ast_bar"] = cvx.Parameter(
                (self.nb_asteroids, self.params.K), name="r_prime_ast_bar"
            )

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        constraints = [
            # map constraints
            self.variables["X"][0, :] >= self.problem_parameters["map_x"][0] + self.border_padding,
            self.variables["X"][0, :] <= self.problem_parameters["map_x"][1] - self.border_padding,
            self.variables["X"][1, :] >= self.problem_parameters["map_y"][0] + self.border_padding,
            self.variables["X"][1, :] <= self.problem_parameters["map_y"][1] - self.border_padding,
            # # thruster angle constraint
            # self.variables["X"][6, :] >= self.problem_parameters["max_delta"][0],
            # self.variables["X"][6, :] <= self.problem_parameters["max_delta"][1],
            # # # mass consumption constraint
            # self.variables["X"][7, :] >= self.problem_parameters["m_satellite"],
            # thrust constraints
            self.variables["U"][0, :] >= self.problem_parameters["thrust_limits"][0],
            self.variables["U"][0, :] <= self.problem_parameters["thrust_limits"][1],
            # thruster angular velocity constraints
            self.variables["U"][1, :] >= self.problem_parameters["thrust_limits"][0],
            self.variables["U"][1, :] <= self.problem_parameters["thrust_limits"][1],
            self.variables["p"] >= 0.1,  # try with 0
            # initial equality constraints
            self.variables["X"][:, 0] - self.problem_parameters["init_state"] + self.variables["v_ic"] == 0,
            # self.variables["U"][:, 0] == self.problem_parameters["init_input"],  # add initial input when replanning
            self.variables["U"][:, 0] == 0,
            # terminal equality constraints
            self.variables["X"][:, -1] - self.problem_parameters["goal_state"] + self.variables["v_tc"] == 0,
            self.variables["U"][:, -1] == 0,
        ]

        # Trust region constraints
        trust_region_constraints = []
        for k in range(self.params.K):
            trust_region_constraints.append(
                cvx.norm(self.variables["X"][:, k] - self.problem_parameters["X_bar"][:, k], 2)
                + cvx.sum_squares(self.variables["U"][:, k] - self.problem_parameters["U_bar"][:, k])
                + cvx.norm(self.variables["p"] - self.problem_parameters["p_bar"], 2)
                <= self.problem_parameters["tr_radius"]
            )
        constraints.extend(trust_region_constraints)

        # Dynamics constraints
        dynamics_constraints = []
        if self.foh:
            for k in range(self.params.K - 1):
                dynamics_constraints.append(
                    self.variables["X"][:, k + 1]
                    == cvx.reshape(
                        self.problem_parameters["A_bar"][:, k], (self.satellite.n_x, self.satellite.n_x), order="F"
                    )
                    @ self.variables["X"][:, k]
                    + cvx.reshape(
                        self.problem_parameters["B_minus_bar"][:, k],
                        (self.satellite.n_x, self.satellite.n_u),
                        order="F",
                    )
                    @ self.variables["U"][:, k]
                    + cvx.reshape(
                        self.problem_parameters["B_plus_bar"][:, k], (self.satellite.n_x, self.satellite.n_u), order="F"
                    )
                    @ self.variables["U"][:, k + 1]
                    + cvx.reshape(
                        self.problem_parameters["F_bar"][:, k], (self.satellite.n_x, self.satellite.n_p), order="F"
                    )
                    @ self.variables["p"]
                    + self.problem_parameters["r_bar"][:, k]
                    + self.variables["v"][:, k]
                )
        else:
            for k in range(self.params.K - 1):
                dynamics_constraints.append(
                    self.variables["X"][:, k + 1]
                    == cvx.reshape(
                        self.problem_parameters["A_bar"][:, k], (self.satellite.n_x, self.satellite.n_x), order="F"
                    )
                    @ self.variables["X"][:, k]
                    + cvx.reshape(
                        self.problem_parameters["B_bar"][:, k], (self.satellite.n_x, self.satellite.n_u), order="F"
                    )
                    @ self.variables["U"][:, k]
                    + cvx.reshape(
                        self.problem_parameters["F_bar"][:, k], (self.satellite.n_x, self.satellite.n_p), order="F"
                    )
                    @ self.variables["p"]
                    + self.problem_parameters["r_bar"][:, k]
                    + self.variables["v"][:, k]
                )
        constraints.extend(dynamics_constraints)

        # Obstacle constraints
        if self.planets:
            planet_constraints = []
            for k in range(self.params.K):
                planet_constraints.append(
                    cvx.reshape(
                        self.problem_parameters["C_planet_bar"][:, k], (self.nb_planets, self.dim_pos), order="C"
                    )
                    @ self.variables["X"][:2, k]
                    + self.problem_parameters["r_prime_planet_bar"][:, k]
                    <= self.variables["v_s"]
                )
            constraints.extend(planet_constraints)
            if self.satellites:
                satellite_constraints = []
                for k in range(self.params.K):
                    satellite_constraints.append(
                        cvx.reshape(
                            self.problem_parameters["C_sat_bar"][:, k],
                            (self.nb_satellites, self.dim_pos),
                            order="C",
                        )
                        @ self.variables["X"][:2, k]
                        + self.problem_parameters["G_bar"][:, k] * self.variables["p"]
                        + self.problem_parameters["r_prime_sat_bar"][:, k]
                        <= self.variables["v_s"]  # why the same as for planets?
                    )
                constraints.extend(satellite_constraints)

        if self.asteroids:
            asteroid_constraints = []
            for k in range(self.params.K):
                asteroid_constraints.append(
                    cvx.reshape(
                        self.problem_parameters["C_ast_bar"][:, k], (self.nb_asteroids, self.dim_pos), order="C"
                    )
                    @ self.variables["X"][:2, k]
                    + self.problem_parameters["H_bar"][:, k] * self.variables["p"]
                    + self.problem_parameters["r_prime_ast_bar"][:, k]
                    <= self.variables["v_s"]
                )
            constraints.extend(asteroid_constraints)

        # Docker Constraints
        if self.landing_constraints_points:
            docker_constraints = []
            i = 1
            # for i in range(3):
            for j in range(self.n_last):
                docker_constraints.append(
                    self.problem_parameters["docker_matrices"][i] @ self.variables["X"][:2, -(j + 1)]
                    <= self.problem_parameters["docker_b"][i]
                )
            constraints.extend(docker_constraints)

        return constraints

    def _compute_j(self, X, U, p) -> float:
        """
        Compute defects, s_plus, gic, gtc and cost
        """
        total_cost = 0.0

        # Defects
        if self.foh:
            propagated_states = self.integrator_foh.integrate_nonlinear_piecewise(X, U, p)
        else:
            propagated_states = self.integrator_zoh.integrate_nonlinear_piecewise(X, U, p)

        defects = X - propagated_states

        # S_plus
        s_plus_planets = np.zeros((len(self.planets), self.params.K))
        s_plus_satellites = np.zeros((len(self.satellites), self.params.K))
        s_plus_asteroids = np.zeros((len(self.asteroids), self.params.K))
        for k in range(self.params.K):
            for i, planet in enumerate(self.planets):
                s_plus_planets[i, k] = self._s_plus_circle(
                    self.planets[planet].radius,
                    np.array(self.planets[planet].center),
                    X[:2, k],
                )
            for i, sat in enumerate(self.satellites):
                p_s_k, _, _ = self.satellite_pos_at_k(sat, k)
                s_plus_satellites[i, k] = self._s_plus_circle(self.satellites[sat].radius, p_s_k, X[:2, k])

            for i, ast in enumerate(self.asteroids):
                p_a_k, _ = self.asteroid_pos_at_k(ast, k)
                s_plus_asteroids[i, k] = self._s_plus_circle(self.asteroids[ast].radius, p_a_k, X[:2, k])

        # g_ic, g_tc
        g_ic = X[:, 0] - self.problem_parameters["init_state"].value
        g_tc = X[:6, -1] - self.problem_parameters["goal_state"].value

        s_plus_stacked = np.vstack((s_plus_planets, s_plus_satellites, s_plus_asteroids))

        # Cost
        terminal_cost = (
            (self.params.weight_p @ p)[0]
            + self.params.lambda_nu
            * (
                np.linalg.norm(g_ic, 1)
                + np.linalg.norm(g_tc, 1)
                + np.linalg.norm(np.array(defects), 1)
                + np.max(s_plus_stacked.flatten())
            )
            + self.params.weight_thrust * np.linalg.norm(U[0, :], 1)
            + self.params.weight_ddelta * np.linalg.norm(U[1, :], 1)
        )

        running_cost = 0.0
        for k in range(self.params.K - 1):
            running_cost += self.params.weight_thrust * np.abs(U[0, k] - U[0, k + 1])
            running_cost += self.params.weight_ddelta * np.abs(U[1, k] - U[1, k + 1])
        running_cost *= 1 / (2 * self.params.K)

        total_cost = terminal_cost + running_cost

        return total_cost

    def _compute_l(self) -> float:
        terminal_cost = (
            (self.params.weight_p @ self.variables["p"].value)[0]
            + self.params.lambda_nu
            * (
                np.linalg.norm(self.variables["v_ic"].value, 1)
                + np.linalg.norm(self.variables["v_tc"].value, 1)
                + np.linalg.norm(self.variables["v"].value, 1)
                + self.variables["v_s"].value
            )
            + self.params.weight_thrust * np.linalg.norm(self.variables["U"][0, :].value, 1)
            + self.params.weight_ddelta * np.linalg.norm(self.variables["U"][1, :].value, 1)
        )

        running_cost = 0.0
        for k in range(self.params.K - 1):
            running_cost += self.params.weight_thrust * np.abs(
                self.variables["U"][0, k].value - self.variables["U"][0, k + 1].value
            )
            running_cost += self.params.weight_ddelta * np.abs(
                self.variables["U"][1, k].value - self.variables["U"][1, k + 1].value
            )
        running_cost *= 1 / (2 * self.params.K)

        total_cost = terminal_cost + running_cost
        return total_cost

    def _s_plus_circle(self, radius: float, center: NDArray, pos: NDArray) -> float:
        safe_radius = self.params.mult_padding * (radius + self.params.add_padding)
        s_plus = -((pos[0] - center[0]) ** 2) - (pos[1] - center[1]) ** 2 + safe_radius**2
        if s_plus > 0:  # inside the enlarged circular obstacle
            return s_plus
        else:
            return 0

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        terminal_cost = (
            self.params.weight_p @ self.variables["p"]
            + self.params.lambda_nu
            * (
                cvx.norm(self.variables["v_ic"], 1)
                + cvx.norm(self.variables["v_tc"], 1)
                + cvx.norm(self.variables["v"], 1)
                + self.variables["v_s"]
            )
            + self.params.weight_thrust * cvx.norm(self.variables["U"][0, :], 1)
            + self.params.weight_ddelta * cvx.norm(self.variables["U"][1, :], 1)
        )

        running_cost = 0.0
        for k in range(self.params.K - 1):
            running_cost += self.params.weight_thrust_rate * cvx.abs(
                self.variables["U"][0, k] - self.variables["U"][0, k + 1]
            )
            running_cost += self.params.weight_ddelta_rate * cvx.abs(
                self.variables["U"][1, k] - self.variables["U"][1, k + 1]
            )
        running_cost *= 1 / (2 * self.params.K)

        objective = terminal_cost + running_cost
        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        if self.foh:
            (
                self.problem_parameters["A_bar"].value,
                self.problem_parameters["B_plus_bar"].value,
                self.problem_parameters["B_minus_bar"].value,
                self.problem_parameters["F_bar"].value,
                self.problem_parameters["r_bar"].value,
            ) = self.integrator_foh.calculate_discretization(
                self.problem_parameters["X_bar"].value,
                self.problem_parameters["U_bar"].value,
                self.problem_parameters["p_bar"].value,
            )
        else:
            (
                self.problem_parameters["A_bar"].value,
                self.problem_parameters["B_bar"].value,
                self.problem_parameters["F_bar"].value,
                self.problem_parameters["r_bar"].value,
            ) = self.integrator_zoh.calculate_discretization(
                self.problem_parameters["X_bar"].value,
                self.problem_parameters["U_bar"].value,
                self.problem_parameters["p_bar"].value,
            )

        if self.planets:
            (
                self.problem_parameters["C_planet_bar"].value,
                self.problem_parameters["r_prime_planet_bar"].value,
            ) = self._calculate_jacobian_planets()
            if self.satellites:
                (
                    self.problem_parameters["C_sat_bar"].value,
                    self.problem_parameters["r_prime_sat_bar"].value,
                    self.problem_parameters["G_bar"].value,
                ) = self.calculate_jacobian_satellites()

        if self.asteroids:
            (
                self.problem_parameters["C_ast_bar"].value,
                self.problem_parameters["r_prime_ast_bar"].value,
                self.problem_parameters["H_bar"].value,
            ) = self.calculate_jacobian_asteroids()

        if self.landing_constraints_points:
            docker_matrices, docker_b = self._compute_docker_constraints()
            self.problem_parameters["docker_matrices"].value = docker_matrices
            self.problem_parameters["docker_b"].value = docker_b

    def _calculate_jacobian_planets(self):
        # The goal is to generate a matrix C_bar that contains the jacobians of the obstacles,
        # C should be of size (n_x * nb_obstacles, K)
        # So one line of C_bar should be the jacobian of the obstacle with respect to the state at time k

        X_bar = self.problem_parameters["X_bar"].value

        C_planets = np.zeros((self.nb_planets * self.dim_pos, self.params.K))
        r_planets = np.zeros((self.nb_planets, self.params.K))

        for k in range(self.params.K):
            for i, planet in enumerate(self.planets):
                jacobian_planet, r_planet = self.jacobian_for_circle(
                    self.planets[planet].radius, np.array(self.planets[planet].center), X_bar[:2, k]
                )
                C_planets[i * self.dim_pos : (i + 1) * self.dim_pos, k] = jacobian_planet
                r_planets[i, k] = r_planet

        return C_planets, r_planets

    def calculate_jacobian_satellites(self):
        X_bar = self.problem_parameters["X_bar"].value
        p_bar = self.problem_parameters["p_bar"][0].value
        G_bar = np.zeros((self.nb_satellites, self.params.K))
        C_satellites = np.zeros((self.nb_satellites * self.dim_pos, self.params.K))
        r_satellites = np.zeros((self.nb_satellites, self.params.K))

        for k in range(self.params.K):
            for i, sat in enumerate(self.satellites):
                p_s_k, t_k, center = self.satellite_pos_at_k(sat, k)

                jacobian_satellite, r_satellite = self.jacobian_for_circle(
                    self.satellites[sat].radius, p_s_k, X_bar[:2, k]
                )

                G_k = self.temporal_jacobian_sat(
                    k,
                    X_bar[:2, k],
                    t_k,
                    self.satellites[sat].orbit_r,
                    self.satellites[sat].tau,
                    self.satellites[sat].omega,
                    p_s_k,
                )

                G_bar[i, k] = G_k

                r_satellite -= G_k * p_bar

                C_satellites[i * self.dim_pos : (i + 1) * self.dim_pos, k] = jacobian_satellite
                r_satellites[i, k] = r_satellite

        return C_satellites, r_satellites, G_bar

    def calculate_jacobian_asteroids(self):
        X_bar = self.problem_parameters["X_bar"].value
        p_bar = self.problem_parameters["p_bar"][0].value
        H_bar = np.zeros((self.nb_asteroids, self.params.K))
        C_asteroids = np.zeros((self.nb_asteroids * self.dim_pos, self.params.K))
        r_asteroids = np.zeros((self.nb_asteroids, self.params.K))

        for k in range(self.params.K):
            for i, ast in enumerate(self.asteroids):
                p_a_k, t_k = self.asteroid_pos_at_k(ast, k)

                jacobian_asteroid, r_asteroid = self.jacobian_for_circle(
                    self.asteroids[ast].radius, p_a_k, X_bar[:2, k]
                )

                H_k = self.temporal_jacobian_ast(k, X_bar[:2, k], p_a_k, ast)

                H_bar[i, k] = H_k

                r_asteroid -= H_k * p_bar

                C_asteroids[i * self.dim_pos : (i + 1) * self.dim_pos, k] = jacobian_asteroid
                r_asteroids[i, k] = r_asteroid

        return C_asteroids, r_asteroids, H_bar

    def satellite_pos_at_k(self, sat: PlayerName, k):
        p_bar = self.problem_parameters["p_bar"][0].value
        orbit_r = self.satellites[sat].orbit_r
        tau = self.satellites[sat].tau
        omega = self.satellites[sat].omega

        for planet in self.planets:
            if sat.startswith(planet):
                center = np.array(self.planets[planet].center)
                break
        t_k = (p_bar + self.sim_time) * k / (self.params.K - 1)
        x_s_k = center[0] + orbit_r * np.cos(tau + omega * t_k)
        y_s_k = center[1] + orbit_r * np.sin(tau + omega * t_k)
        p_s_k = np.array([x_s_k, y_s_k])
        return p_s_k, t_k, center

    def asteroid_pos_at_k(self, ast: PlayerName, k):
        p_bar = self.problem_parameters["p_bar"][0].value

        p_a_0 = self.asteroids[ast].start
        glob_vel = self.get_glob_vel_asteroid(ast)

        t_k = (p_bar + self.sim_time) * k / (self.params.K - 1)
        x_a_k = p_a_0[0] + glob_vel[0] * t_k
        y_a_k = p_a_0[1] + glob_vel[1] * t_k
        p_a_k = np.array([x_a_k, y_a_k])
        return p_a_k.ravel(), t_k

    def get_glob_vel_asteroid(self, ast: PlayerName):
        lin_vel = self.asteroids[ast].velocity
        orientation = self.asteroids[ast].orientation
        transf_matrix = np.array(
            [[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]]
        )
        glob_vel = transf_matrix @ np.array(lin_vel).reshape((2, 1))
        return glob_vel

    def temporal_jacobian_sat(self, k, pos_agent, t_k, orbit_r, tau, omega, current_sat_pos):
        a = 2 * omega * k * orbit_r / (self.params.K - 1)
        alpha = tau + omega * t_k
        return a * (
            (pos_agent[1] - current_sat_pos[1]) * np.cos(alpha) - (pos_agent[0] - current_sat_pos[0]) * np.sin(alpha)
        )

    def temporal_jacobian_ast(self, k, pos_agent, current_ast_pos, ast):
        a = 2 * k / (self.params.K - 1)
        glob_vel = self.get_glob_vel_asteroid(ast)
        return a * (
            (pos_agent[0] - current_ast_pos[0]) * glob_vel[0] + (pos_agent[1] - current_ast_pos[1]) * glob_vel[1]
        )

    def jacobian_for_circle(self, radius: float, center: NDArray, pos: NDArray) -> tuple[NDArray, float]:
        safe_radius = self.params.mult_padding * (radius + self.params.add_padding)
        jacobian = np.array([-2 * (pos[0] - center[0]), -2 * (pos[1] - center[1])])
        residual = (
            -((pos[0] - center[0]) ** 2)
            - (pos[1] - center[1]) ** 2
            + safe_radius**2
            + 2 * (pos[0] - center[0]) * pos[0]
            + 2 * (pos[1] - center[1]) * pos[1]
        )
        # ravel to convert 2D array (2,1) to 1D array as expected
        return jacobian.ravel(), residual

    def _linear_constraint_from_points(self, p1, p2, side):
        x1, y1 = p1
        x2, y2 = p2

        # Compute coefficients
        a1 = y2 - y1
        a2 = -(x2 - x1)
        a3 = x2 * y1 - x1 * y2

        # Create A and b
        A = np.array([a1, a2])
        b = -a3

        if side == "above":
            return A, b
        elif side == "below":
            return -A, -b
        else:
            raise ValueError("Unidentified side of docker constraint !")

    def _compute_docker_constraints(self):
        B = self.landing_constraints_points["B"]
        A1 = self.landing_constraints_points["A1"]
        A2 = self.landing_constraints_points["A2"]
        C = self.landing_constraints_points["C"]

        A_arm2, b_arm2 = self._linear_constraint_from_points(B, A2, "below")
        A_base, b_base = self._linear_constraint_from_points(A1, A2, "above")
        A_arm1, b_arm1 = self._linear_constraint_from_points(C, A1, "above")

        return np.vstack((A_arm1, A_base, A_arm2)), np.array([b_arm1, b_base, b_arm2])

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """

        pass

    def _update_trust_region(self, rho: float, J_star: float):
        """
        Update trust region radius.
        """
        if rho < self.params.rho_0:
            self.problem_parameters["tr_radius"].value = max(
                self.params.min_tr_radius, self.problem_parameters["tr_radius"].value / self.params.alpha
            )
            # we don't update the "_bar" variables
            print("rho < rho_0")
        elif rho >= self.params.rho_0 and rho < self.params.rho_1:
            self.problem_parameters["tr_radius"].value = max(
                self.params.min_tr_radius, self.problem_parameters["tr_radius"].value / self.params.alpha
            )
            (
                self.J_bar,
                self.problem_parameters["X_bar"].value,
                self.problem_parameters["U_bar"].value,
                self.problem_parameters["p_bar"].value,
            ) = (
                J_star,
                self.variables["X"].value,
                self.variables["U"].value,
                self.variables["p"].value,
            )
            print("rho >= rho_0 and rho < rho_1")
        elif rho >= self.params.rho_1 and rho < self.params.rho_2:
            (
                self.J_bar,
                self.problem_parameters["X_bar"].value,
                self.problem_parameters["U_bar"].value,
                self.problem_parameters["p_bar"].value,
            ) = (
                J_star,
                self.variables["X"].value,
                self.variables["U"].value,
                self.variables["p"].value,
            )
            # we don't update the trust region radius
            print("rho >= rho_1 and rho < rho_2")
        elif rho >= self.params.rho_2:
            self.problem_parameters["tr_radius"].value = min(
                self.params.max_tr_radius, self.params.beta * self.problem_parameters["tr_radius"].value
            )
            (
                self.J_bar,
                self.problem_parameters["X_bar"].value,
                self.problem_parameters["U_bar"].value,
                self.problem_parameters["p_bar"].value,
            ) = (
                J_star,
                self.variables["X"].value,
                self.variables["U"].value,
                self.variables["p"].value,
            )
            print("rho >= rho_2")
        else:
            raise ValueError("Invalid rho value")

    def _extract_optimal_cmds_and_traj(self):
        """
        Extract optimal commands and trajectory from the solver.
        """

        print("Extracting optimal commands and trajectory...")

        # print(f"p : {self.variables['p'].value}")
        # print(f"X : {self.variables['X'].value}")
        # print(f"U : {self.variables['U'].value}")
        final_time = self.variables["p"][0].value
        print(f"Final time: {final_time}")

        ts: Sequence[Timestamp] = np.linspace(self.sim_time, final_time + self.sim_time, self.params.K).tolist()
        cmds_list = [
            SatelliteCommands(f, dd) for f, dd in zip(self.variables["U"].value[0, :], self.variables["U"].value[1, :])
        ]
        states = [SatelliteState(*v) for v in self.variables["X"].value.T]

        cmd_sequence = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)
        state_sequence = DgSampledSequence[SatelliteState](timestamps=ts, values=states)
        return cmd_sequence, state_sequence

    def lqr_controller(
        self, current_state: SatelliteState, current_input: SatelliteCommands, expected_state: SatelliteState
    ):
        """
        LQR controller to stabilize the system around the expected state.
        """
        X_curr = np.array(
            [
                current_state.x,
                current_state.y,
                current_state.psi,
                current_state.vx,
                current_state.vy,
                current_state.dpsi,
                # current_state.delta,
                # current_state.m,
            ]
        )
        X_exp = np.array(
            [
                expected_state.x,
                expected_state.y,
                expected_state.psi,
                expected_state.vx,
                expected_state.vy,
                expected_state.dpsi,
                # expected_state.delta,
                # expected_state.m,
            ]
        )

        U_curr = np.array([current_input.F_left, current_input.F_right])

        _, A_func, B_func, _ = self.satellite.get_dynamics()
        A = A_func(X_curr, U_curr, self.variables["p"].value)
        B = B_func(X_curr, U_curr, self.problem_parameters["p_bar"].value)
        Q = np.diag([10, 10, 10, 10, 10, 1])
        R = np.diag([0.1, 0.1])

        try:
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            # print(f"P found: {P}")
            # K = np.linalg.inv(R) @ B.T @ P
            K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
            # print(f"K: {K}")
        except:
            print("LQR failed")
            return [0, 0]
        print(f"Output: {-K @ (X_curr - X_exp)}")
        return -K @ (X_curr - X_exp)

    def plot_trajectory(self, i):
        plt.xlim(self.map_borders["xmin"], self.map_borders["xmax"])
        plt.ylim(self.map_borders["ymin"], self.map_borders["ymax"])
        plt.plot(self.variables["X"].value[0], self.variables["X"].value[1], label=f"traj_{i}")
        for j in range(self.params.K):
            plt.arrow(
                self.variables["X"].value[0, j],
                self.variables["X"].value[1, j],
                1.0 * np.cos(self.variables["X"].value[2, j]),
                1.0 * np.sin(self.variables["X"].value[2, j]),
                color="blue",
                head_width=0.25,
                head_length=0.4,
            )
            # if self.variables["U"].value[0, j] > 0:
            #     delta_angle = self.variables["X"].value[2, j] + np.pi + self.variables["X"].value[6, j]
            # else:
            #     delta_angle = self.variables["X"].value[2, j] + self.variables["X"].value[6, j]
            # thrust = np.abs(self.variables["U"].value[0, j]) / self.sp.F_limits[1]
            # plt.arrow(
            #     self.variables["X"].value[0, j],
            #     self.variables["X"].value[1, j],
            #     thrust * np.cos(delta_angle),
            #     thrust * np.sin(delta_angle),
            #     color="red",
            #     head_width=0.2,
            #     head_length=0.1,
            # )
        # Plot planets
        for planet in self.planets.values():
            circle = Circle((planet.center[0], planet.center[1]), planet.radius, color="brown", fill=True)
            plt.gca().add_patch(circle)

        # Plot satellites
        for sat in self.satellites:
            orbit_r = self.satellites[sat].orbit_r
            tau = self.satellites[sat].tau
            omega = self.satellites[sat].omega
            center = None
            for planet in self.planets:
                if sat.startswith(planet):
                    center = np.array(self.planets[planet].center)
                    break
            if center is not None:
                # Plot initial position
                x_s_0 = center[0] + orbit_r * np.cos(tau)
                y_s_0 = center[1] + orbit_r * np.sin(tau)
                circle = Circle((x_s_0, y_s_0), self.satellites[sat].radius, color="green", fill=True)
                plt.gca().add_patch(circle)

                # Plot satellite movement outline
                p_s = np.zeros((2, self.params.K))
                for k in range(self.params.K):
                    p_s[:, k], _, _ = self.satellite_pos_at_k(sat, k)
                plt.plot(p_s[0], p_s[1], "b--")  # Movement outline

        # Plot asteroids
        for ast in self.asteroids:
            p_a_0 = self.asteroids[ast].start
            glob_vel = self.get_glob_vel_asteroid(ast)
            # Plot initial position
            circle = Circle((p_a_0[0], p_a_0[1]), self.asteroids[ast].radius, color="gray", fill=True)
            plt.gca().add_patch(circle)

            # Plot asteroid movement outline
            p_a = np.zeros((2, self.params.K))
            for k in range(self.params.K):
                p_a[:, k], _ = self.asteroid_pos_at_k(ast, k)
            plt.plot(p_a[0], p_a[1], "k--")  # Movement outline

        plt.legend()
        plt.savefig(f"traj.png")
        plt.close()

    def plot_heatmap(self):
        v_abs = np.abs(self.variables["v"].value)
        plt.imshow(v_abs, cmap="hot", interpolation="nearest", aspect="auto")
        plt.colorbar(label="Absolute value of v")
        plt.title("Heatmap of v")
        plt.xlabel("Time step")
        plt.ylabel("State dimension")
        plt.savefig("heatmap_v.png")
        plt.close()

    def plot_linearized_obstacles(self, k):
        """
        Plot all the linearized constraints of the obstacles at time k.
        """
        plt.gca().set_aspect("equal", adjustable="box")
        X = self.problem_parameters["X_bar"].value
        plt.xlim(self.map_borders["xmin"], self.map_borders["xmax"])
        plt.ylim(self.map_borders["ymin"], self.map_borders["ymax"])
        plt.scatter(X[0, k], X[1, k], label=f"constraints_{k}")

        for i, planet in enumerate(self.planets):
            jacobian = self.problem_parameters["C_planet_bar"].value[i * self.dim_pos : (i + 1) * self.dim_pos, k]
            r = self.problem_parameters["r_prime_planet_bar"].value[i, k]
            # Plot the planet
            circle = Circle(
                (self.planets[planet].center[0], self.planets[planet].center[1]),
                self.planets[planet].radius,
                color="brown",
                fill=True,
            )
            plt.gca().add_patch(circle)
            # Plot the linearized constraint
            x = np.linspace(self.map_borders["xmin"], self.map_borders["xmax"], 100)
            y = (-jacobian[0] * x - r) / jacobian[1]
            plt.plot(x, y, "r--")

        for i, sat in enumerate(self.satellites):
            jacobian = self.problem_parameters["C_sat_bar"].value[i * self.dim_pos : (i + 1) * self.dim_pos, k]
            r = self.problem_parameters["r_prime_sat_bar"].value[i, k]
            # Plot the satellite
            orbit_r = self.satellites[sat].orbit_r
            tau = self.satellites[sat].tau
            omega = self.satellites[sat].omega
            center = None
            for planet in self.planets:
                if sat.startswith(planet):
                    center = np.array(self.planets[planet].center)
                    break
            if center is not None:
                x_s = center[0] + orbit_r * np.cos(tau + omega * self.variables["p"].value[0] * k / (self.params.K - 1))
                y_s = center[1] + orbit_r * np.sin(tau + omega * self.variables["p"].value[0] * k / (self.params.K - 1))
                circle = Circle((x_s, y_s), self.satellites[sat].radius, color="green", fill=True)
                plt.gca().add_patch(circle)
                # Plot the linearized constraint
                x = np.linspace(self.map_borders["xmin"], self.map_borders["xmax"], 100)
                y = (-jacobian[0] * x - r) / jacobian[1]
                plt.plot(x, y, "r--")

        for i, ast in enumerate(self.asteroids):
            jacobian = self.problem_parameters["C_ast_bar"].value[i * self.dim_pos : (i + 1) * self.dim_pos, k]
            r = self.problem_parameters["r_prime_ast_bar"].value[i, k]
            # Plot the asteroid
            p_a_0 = self.asteroids[ast].start
            glob_vel = self.get_glob_vel_asteroid(ast)
            t_k = self.variables["p"].value[0] * k / (self.params.K - 1)
            x_a = p_a_0[0] + glob_vel[0] * t_k
            y_a = p_a_0[1] + glob_vel[1] * t_k
            circle = Circle((x_a, y_a), self.asteroids[ast].radius, color="gray", fill=True)
            plt.gca().add_patch(circle)
            # Plot the linearized constraint
            x = np.linspace(self.map_borders["xmin"], self.map_borders["xmax"], 100)
            y = (-jacobian[0] * x - r) / jacobian[1]
            plt.plot(x, y, "r--")

        plt.legend()
        plt.title(f"Linearized constraints at time {k}")
        plt.savefig(f"linearized_constraints.png")
        plt.close()

    @staticmethod
    def _extract_seq_from_array() -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
    def _extract_seq_from_array() -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F = np.array([0, 1, 2, 3, 4])
        ddelta = np.array([0, 0, 0, 0, 0])
        cmds_list = [SatelliteCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)
        cmds_list = [SatelliteCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 8)
        states = [SatelliteState(*v) for v in npstates]
        mystates = DgSampledSequence[SatelliteState](timestamps=ts, values=states)
        return mycmds, mystates

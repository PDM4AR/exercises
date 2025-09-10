import numpy as np
import cvxpy as cvx
import sympy as spy
from dataclasses import dataclass, field

from numpy.typing import NDArray

from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)

from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams

from pdm4ar.exercises.ex11.spaceship import Spaceship
from pdm4ar.exercises.ex11.discretization import (
    DiscretizationMethod,
    FirstOrderHold,
    ZeroOrderHold,
)

import math
from dg_commons.sim.goals import PlanningGoal
from pdm4ar.exercises_def.ex11.goal import (
    SpaceshipTarget,
    SatelliteTarget,
    DockingTarget,
)


@dataclass(frozen=True)
class ToggleParams:
    """
    Tunable parameters
    """

    # Let's hope for the best

    weight_mf: float = 5
    tf_initial_guess: float = 25  # 12
    safety_buffer: float = 1.6


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 200  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 1000  # 5  # initial trust region radius
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


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: list[PlanetParams]
    satellites: list[SatelliteParams]
    spaceship: Spaceship
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

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

    Al: NDArray
    bl: NDArray

    dock: bool

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()

        # Tunable Parameters
        self.toggle_params = ToggleParams()

        # Spaceship Dynamics
        self.spaceship = Spaceship(self.sg, self.sp)

        self.n_x = self.spaceship.n_x
        self.n_u = self.spaceship.n_u
        self.n_p = self.spaceship.n_p

        self.dock = False

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SpaceshipState, goal: PlanningGoal, X_guess, U_guess, p_guess
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        # Initialization
        # Initial Guess
        print("u shape ", U_guess.shape)
        X_bar = X_guess
        U_bar = U_guess
        p_bar = p_guess
        self.problem_parameters["X_bar"].value = X_bar.copy()
        self.problem_parameters["U_bar"].value = U_bar.copy()
        self.problem_parameters["p_bar"].value = p_bar.copy()
        self.problem_parameters["init_state"].value = init_state.as_ndarray()
        self.goal = goal
        self.problem_parameters["tr_radius"].value = np.array([self.params.tr_radius])
        self.problem_parameters["pos_tol"].value = np.array([goal.pos_tol])
        self.problem_parameters["vel_tol"].value = np.array([goal.vel_tol])
        self.problem_parameters["dir_tol"].value = np.array([goal.dir_tol])
        self.dock = False
        if isinstance(goal, DockingTarget):
            self.dock = True
            a, b, c, a1, a2, p = goal.get_landing_constraint_points()
            A, B = self.get_landing_constraints(a, b, c, goal)
            self.Al = A
            self.bl = B

            self.problem_parameters["A_landing"].value = A
            self.problem_parameters["b_landing"].value = B
            print(f"p_value = {p}")
            self.problem_parameters["p_landing"].value = [p]
            self.problem_parameters["Ap_landing"].value = [1]
            self.problem_parameters["goal_dir"].value = [goal.target.as_ndarray()[2]]
        else:
            self.problem_parameters["A_landing"].value = np.zeros((2, 2))
            self.problem_parameters["b_landing"].value = np.zeros((2))
            self.problem_parameters["p_landing"].value = [0]
            self.problem_parameters["Ap_landing"].value = [0]
            self.problem_parameters["goal_dir"].value = [goal.target.as_ndarray()[2]]

        # ***** DEBUG *****
        # print("Start: ", self.problem_parameters["init_state"].value)
        # print("Goal: ", self.problem_parameters["goal_state"].value)
        # *****************

        it = 0
        converged = False
        while not converged and it < 100:
            self._convexification()

            # ***** DEBUG *****
            # print("tr_radius: ", self.problem_parameters["tr_radius"].value)
            # *****************

            try:
                error = self.problem.solve(
                    verbose=self.params.verbose_solver,
                    solver=self.params.solver,
                    max_iters=self.params.max_iterations,
                )
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            # Check convergence
            converged = self._check_convergence()

            # Update trust region
            X_bar, U_bar, p_bar, tr_radius = self._update_trust_region()

            # Update parameters
            self.problem_parameters["X_bar"].value = X_bar.copy()
            self.problem_parameters["U_bar"].value = U_bar.copy()
            self.problem_parameters["p_bar"].value = p_bar.copy()
            self.problem_parameters["tr_radius"].value = np.array([tr_radius])

            # if np.any(
            #    np.any(np.abs(self.variables["nu"].value) > 1e-5, axis=0), axis=0
            # ):
            #    print("nu != 0: ")

            # Print iteration
            it += 1
            # print("Iteration: ", it)

        # ***** DEBUG *****
        if np.any(np.any(np.abs(self.variables["nu"].value) > 1e-5, axis=0), axis=0):
            print("nu != 0: ", self.variables["nu"].value)
        if np.any(np.any(np.abs(self.variables["nu_s"].value) > 1e-5, axis=0), axis=0):
            print("nu_s != 0: ", self.variables["nu_s"].value)
        if np.any(np.any(np.abs(self.variables["nu_tc"].value) > 1e-5, axis=0), axis=0):
            print("nu_tc != 0: ", self.variables["nu_tc"].value)

        print("p: ", self.variables["p"].value)
        # *****************

        # Example data: sequence from array
        # mycmds, mystates = self._get_commands()
        mycmds, mystates = self._get_cmds_states()

        nu_nonzero = np.any(np.any(np.abs(self.variables["nu"].value) > 1e-5, axis=0), axis=0)
        nu_s_nonzero = np.any(np.any(np.abs(self.variables["nu_s"].value) > 1e-5, axis=0), axis=0)
        # nu_tc_nonzero = np.any(
        #     np.any(np.abs(self.variables["nu_tc"].value) > 1e-5, axis=0), axis=0
        # )

        goal_reached = (
            np.linalg.norm(self.variables["nu_tc"].value[:2], 2) <= self.problem_parameters["pos_tol"].value
            and np.linalg.norm(self.variables["nu_tc"].value[2:4], 2) <= self.problem_parameters["vel_tol"].value
            and np.abs(self.variables["nu_tc"].value[4]) <= self.problem_parameters["dir_tol"].value
        )

        # feasible_solution = not (nu_nonzero or nu_s_nonzero or nu_tc_nonzero)
        feasible_solution = goal_reached and not (nu_nonzero or nu_s_nonzero)

        return mycmds, mystates, feasible_solution

    def get_landing_constraints(self, A, B, C, goal):
        """Function to calculate conic constraints for
        landing in the docker + arms.
        Must define better the margin on the wall based on
        the spaceship dimension"""
        # Compute constraints for the three edges
        A1, b1 = self.edge_constraint(A, B)
        A2, b2 = self.edge_constraint(C, A)

        # Stack the results in matrix form
        A_matrix = np.vstack([A1, A2])
        b_vector = np.array([b1, b2])

        # check that the goal is inside
        center = np.array([goal.target.x, goal.target.y])
        check = np.all((A_matrix @ center) <= b_vector)
        print(f"center is inside {check}")

        return A_matrix, b_vector

    def get_circle_constraints_r(self, B, goal):
        # for now use the goal position
        center = np.array([goal.target.x, goal.target.y])
        r = np.linalg.norm((B - center))

        return r

    def edge_constraint(self, P1, P2):
        """Compute the constraint coefficients (a, b, c) for the line P1-P2.
        In the form ax + by < c"""
        # Line coefficients
        print(f"calculating contstraints for {P1} and {P2}")
        a = -(P1[1] - P2[1])  # y2 - y1
        b = P1[0] - P2[0]  # -(x2 - x1)
        print(f"a = {a}, b = {b}, x = {P1[0]}, y = {P1[1]}")
        c = a * P1[0] + b * P1[1]  # a*x1 + b*y1 (right-hand side)
        print(f"c = {c}")
        return np.array([a, b]), c

    def intial_guess(self, start, goal) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K
        print("n_u ", self.n_u)
        X = np.zeros((self.n_x, K))
        U = np.zeros((self.n_u, K))
        p = np.zeros((self.n_p))

        p[0] = self.toggle_params.tf_initial_guess
        start_state = start.as_ndarray()

        if isinstance(goal, SatelliteTarget):
            # Forcast goal position at time p
            goal_state = np.concatenate(
                [
                    goal.get_target_state_at(p[0]).as_ndarray(),
                    np.array([0, start_state[7]]),
                ],
                axis=0,
            )
        elif isinstance(goal, SpaceshipTarget):
            goal_state = np.concatenate([goal.target.as_ndarray(), np.array([0, start_state[7]])], axis=0)

        for k in range(K):
            X[:, k] = (1 - (k / (K - 1))) * start_state + (k / (K - 1)) * goal_state

        U[0, :] = (self.sp.thrust_limits[1]) / 2
        U[1, :] = (self.sp.ddelta_limits[0] + self.sp.ddelta_limits[1]) / 2

        U[:, 0] = 0
        U[:, -1] = 0

        return X, U, p

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.n_x, self.params.K)),
            "U": cvx.Variable((self.n_u, self.params.K)),
            "p": cvx.Variable(self.n_p),
            "nu": cvx.Variable((self.n_x, self.params.K)),
            "nu_s": cvx.Variable((len(self.planets) + len(self.satellites), self.params.K)),
            "nu_tc": cvx.Variable(5),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        K = self.params.K
        problem_parameters = {
            "init_state": cvx.Parameter(self.n_x),
            "goal_state": cvx.Parameter(6),
            "A_bar": [cvx.Parameter((self.n_x, self.n_x)) for k in range(K - 1)],
            "B_minus_bar": [cvx.Parameter((self.n_x, self.n_u)) for k in range(K - 1)],
            "B_plus_bar": [cvx.Parameter((self.n_x, self.n_u)) for k in range(K - 1)],
            "F_bar": [cvx.Parameter((self.n_x, self.n_p)) for k in range(K - 1)],
            "r_bar": [cvx.Parameter(self.n_x) for k in range(K - 1)],
            "X_bar": cvx.Parameter((self.n_x, K)),
            "U_bar": cvx.Parameter((self.n_u, K)),
            "p_bar": cvx.Parameter(self.n_p),
            "tr_radius": cvx.Parameter(1),
            "C_goal": cvx.Parameter((5, 5)),
            "F_goal": cvx.Parameter((5, 1)),
            "r_goal": cvx.Parameter(5),
            "A_landing": cvx.Parameter((2, 2)),
            "b_landing": cvx.Parameter(2),
            "p_landing": cvx.Parameter(1),
            "Ap_landing": cvx.Parameter(1),
            "goal_dir": cvx.Parameter(1),
            "pos_tol": cvx.Parameter(1),
            "vel_tol": cvx.Parameter(1),
            "dir_tol": cvx.Parameter(1),
        }

        # Planets
        planet_names = list(self.planets.keys())
        planets_l = len(planet_names)
        for i in range(planets_l):
            problem_parameters[f"C_p_{planet_names[i]}"] = cvx.Parameter((2, K))
            problem_parameters[f"r_p_{planet_names[i]}"] = cvx.Parameter(K)

        # Satellites
        satellite_names = list(self.satellites.keys())
        satellites_l = len(satellite_names)
        for i in range(satellites_l):
            problem_parameters[f"C_s_{satellite_names[i]}"] = cvx.Parameter((2, K))
            problem_parameters[f"F_s_{satellite_names[i]}"] = cvx.Parameter(K)
            problem_parameters[f"r_s_{satellite_names[i]}"] = cvx.Parameter(K)

        return problem_parameters

    def _get_constraints(self) -> list[cvx.constraints]:
        constraints = []
        K = self.params.K
        print(f"Dimension of K {K}")

        # X(0) = initial_state
        constraints.append(self.variables["X"][:, 0] == self.problem_parameters["init_state"])

        # Goal
        constraints.append(
            self.problem_parameters["C_goal"] @ self.variables["X"][:5, K - 1]
            + self.problem_parameters["F_goal"] @ self.variables["p"]
            + self.problem_parameters["r_goal"]
            == self.variables["nu_tc"]
        )

        # p >= 0
        constraints.append(self.variables["p"] >= 0)

        # -phi_max <= phi(t) <= phi_max
        for k in range(K):
            constraints.append(self.variables["X"][6, k] >= self.sp.delta_limits[0])
            constraints.append(self.variables["X"][6, k] <= self.sp.delta_limits[1])

        # m(t) >= m_v
        for k in range(self.params.K):
            constraints.append(self.variables["X"][7, k] >= self.sp.m_v)

        # Space boundaries
        for k in range(self.params.K):
            constraints += [
                self.variables["X"][0, k] >= -9.6,
                self.variables["X"][0, k] <= 23.6,
                self.variables["X"][1, k] >= -9.6,
                self.variables["X"][1, k] <= 9.6,
            ]

        # U(0) = U(1) = 0
        constraints.append(self.variables["U"][:, 0] == np.zeros(2))
        constraints.append(self.variables["U"][:, K - 1] == np.zeros(2))

        # 0 <= F(t) <= F_max
        for k in range(K):
            constraints.append(self.variables["U"][0, k] >= self.sp.thrust_limits[0])
            constraints.append(self.variables["U"][0, k] <= self.sp.thrust_limits[1])

        # -dphi_max <= dphi(t) <= dphi_max
        for k in range(K):
            constraints.append(self.variables["U"][1, k] >= self.sp.ddelta_limits[0])
            constraints.append(self.variables["U"][1, k] <= self.sp.ddelta_limits[1])

        # Dynamics
        # FOH
        for k in range(0, K - 1):
            constraints.append(
                self.variables["X"][:, k + 1]
                == self.problem_parameters["A_bar"][k] @ self.variables["X"][:, k]
                + self.problem_parameters["B_minus_bar"][k] @ self.variables["U"][:, k]
                + self.problem_parameters["B_plus_bar"][k] @ self.variables["U"][:, k + 1]
                + self.problem_parameters["F_bar"][k] @ self.variables["p"]
                + self.problem_parameters["r_bar"][k]
                + self.variables["nu"][:, k]
            )

        # Trust region
        for k in range(K):
            delta_X_k = self.variables["X"][:, k] - self.problem_parameters["X_bar"][:, k]
            delta_U_k = self.variables["U"][:, k] - self.problem_parameters["U_bar"][:, k]
            delta_p = self.variables["p"] - self.problem_parameters["p_bar"]
            constraints += [
                cvx.norm(delta_X_k, 2) + cvx.norm(delta_U_k, 2) + cvx.norm(delta_p, 2)
                <= self.problem_parameters["tr_radius"]
            ]

        # Planets
        planet_names = list(self.planets.keys())
        planet_l = len(planet_names)

        for i in range(planet_l):
            for k in range(K):
                constraints += [
                    self.problem_parameters[f"C_p_{planet_names[i]}"][0, k] * self.variables["X"][0, k]
                    + self.problem_parameters[f"C_p_{planet_names[i]}"][1, k] * self.variables["X"][1, k]
                    + self.problem_parameters[f"r_p_{planet_names[i]}"][k]
                    <= self.variables["nu_s"][i, k]
                ]

        # Satellites
        satellite_names = list(self.satellites.keys())
        satellite_l = len(satellite_names)
        for i in range(satellite_l):
            constraints += [
                self.problem_parameters[f"C_s_{satellite_names[i]}"][0, k] * self.variables["X"][0, k]
                + self.problem_parameters[f"C_s_{satellite_names[i]}"][1, k] * self.variables["X"][1, k]
                + self.problem_parameters[f"F_s_{satellite_names[i]}"][k] * self.variables["p"]
                + self.problem_parameters[f"r_s_{satellite_names[i]}"][k]
                <= self.variables["nu_s"][planet_l + i, k]
            ]

        # goal landing constraints
        if K > 7:
            print("dok")
            for k in range(K - 7, K, 1):
                constraints += [
                    self.variables["X"][2, k] * self.problem_parameters["Ap_landing"]
                    <= self.problem_parameters["goal_dir"] + self.problem_parameters["p_landing"],
                    self.variables["X"][2, k] * self.problem_parameters["Ap_landing"]
                    >= self.problem_parameters["goal_dir"] - self.problem_parameters["p_landing"],
                ]
            for k in range(K - 7, K, 1):
                constraints += [
                    self.problem_parameters["A_landing"] @ self.variables["X"][0:2, k]
                    <= self.problem_parameters["b_landing"],
                ]
        # else:
        #     for k in range(0, K, 1):
        #         constraints += [
        #             self.problem_parameters["A_landing"] @ self.variables["X"][0:2, k]
        #             <= self.problem_parameters["b_landing"],
        #         ]
        #     for k in range(0, K, 1):
        #         constraints += [
        #             self.variables["X"][2, k]
        #             <= self.goal.target.as_ndarray()[2] + self.problem_parameters["p_landing"],
        #             self.variables["X"][2, k]
        #             >= self.goal.target.as_ndarray()[2] - self.problem_parameters["p_landing"],
        #         ]

        return constraints

    def _get_objective(self) -> cvx.Problem:
        """
        Define objective for SCvx.
        """
        # Example objective
        objective = (
            self.params.weight_p @ self.variables["p"]
            - self.toggle_params.weight_mf * self.variables["X"][7, self.params.K - 1]
            + self.params.lambda_nu * cvx.norm(self.variables["nu"], 1)
            + self.params.lambda_nu * cvx.norm(self.variables["nu_s"], 1)
            + self.params.lambda_nu * cvx.norm(self.variables["nu_tc"], 1)
        )

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        self.X_bar = self.problem_parameters["X_bar"].value.copy()
        self.U_bar = self.problem_parameters["U_bar"].value.copy()
        self.p_bar = self.problem_parameters["p_bar"].value.copy()
        (
            A_bar_flat,
            B_plus_bar_flat,
            B_minus_bar_flat,
            F_bar_flat,
            r_bar_flat,
        ) = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)

        K = self.params.K
        # Update parameters
        for k in range(K - 1):
            self.problem_parameters["A_bar"][k].value = np.reshape(
                A_bar_flat[:, k], (self.n_x, self.n_x), order="F"
            ).copy()
            self.problem_parameters["B_minus_bar"][k].value = np.reshape(
                B_minus_bar_flat[:, k], (self.n_x, self.n_u), order="F"
            ).copy()
            self.problem_parameters["B_plus_bar"][k].value = np.reshape(
                B_plus_bar_flat[:, k], (self.n_x, self.n_u), order="F"
            ).copy()
            self.problem_parameters["F_bar"][k].value = np.reshape(
                F_bar_flat[:, k], (self.n_x, self.n_p), order="F"
            ).copy()
            self.problem_parameters["r_bar"][k].value = r_bar_flat[:, k].copy()

        # Planets
        C_p_bar = np.zeros((2, K))
        r_p_bar = np.zeros(K)
        planets = list(self.planets.keys())
        n_planets = len(planets)
        for i in range(n_planets):
            xc = self.planets[planets[i]].center[0]
            yc = self.planets[planets[i]].center[1]
            r = self.planets[planets[i]].radius + self.toggle_params.safety_buffer
            for k in range(K):
                x_bar = self.X_bar[0, k]
                y_bar = self.X_bar[1, k]
                s_bar = r**2 - (x_bar - xc) ** 2 - (y_bar - yc) ** 2
                ds_dx = -2 * (x_bar - xc)
                ds_dy = -2 * (y_bar - yc)
                C_p_bar[0, k] = ds_dx
                C_p_bar[1, k] = ds_dy
                r_p_bar[k] = s_bar - ds_dx * x_bar - ds_dy * y_bar
            self.problem_parameters[f"C_p_{planets[i]}"].value = C_p_bar.copy()
            self.problem_parameters[f"r_p_{planets[i]}"].value = r_p_bar.copy()

        # Satellites
        C_s_bar = np.zeros((2, K))
        F_s_bar = np.zeros(K)
        r_s_bar = np.zeros(K)
        satellite_names = list(self.satellites.keys())
        satellite_l = len(satellite_names)
        for i in range(satellite_l):
            planet_name, _ = satellite_names[i].split("/")
            planet = self.planets[planet_name]
            xc_p = planet.center[0]
            yc_p = planet.center[1]
            satellite = self.satellites[satellite_names[i]]
            orbit_r = satellite.orbit_r
            omega = satellite.omega
            tau = satellite.tau
            radius = satellite.radius
            r = radius + self.toggle_params.safety_buffer
            for k in range(self.params.K):
                x_bar = self.X_bar[0, k]
                y_bar = self.X_bar[1, k]
                p_bar = self.p_bar[0]
                f = k / (self.params.K - 1)

                xc = xc_p + orbit_r * math.cos(omega * f * p_bar + tau)
                yc = yc_p + orbit_r * math.sin(omega * f * p_bar + tau)
                ds_dx = -2 * (x_bar - xc)
                ds_dy = -2 * (y_bar - yc)
                ds_dp = -2 * (x_bar - xc) * f * omega * orbit_r * math.sin(omega * f * p_bar + tau) + 2 * (
                    y_bar - yc
                ) * f * omega * orbit_r * math.cos(omega * f * p_bar + tau)
                s_bar = r**2 - (x_bar - xc) ** 2 - (y_bar - yc) ** 2
                C_s_bar[0, k] = ds_dx
                C_s_bar[1, k] = ds_dy
                F_s_bar[k] = ds_dp
                r_s_bar[k] = s_bar - ds_dx * x_bar - ds_dy * y_bar - ds_dp * p_bar
            self.problem_parameters[f"C_s_{satellite_names[i]}"].value = C_s_bar.copy()
            self.problem_parameters[f"F_s_{satellite_names[i]}"].value = F_s_bar.copy()
            self.problem_parameters[f"r_s_{satellite_names[i]}"].value = r_s_bar.copy()

        # Dynamic goal
        C_goal = np.zeros((5, 5))
        F_goal = np.zeros((5, 1))
        r_goal = np.zeros(5)

        if isinstance(self.goal, SatelliteTarget):
            planet_x = self.goal.planet_x
            planet_y = self.goal.planet_y
            omega = self.goal.omega
            tau = self.goal.tau
            orbit_r = self.goal.orbit_r
            radius = self.goal.radius
            offset_r = self.goal.offset_r

            p_bar = self.p_bar[0]
            # g(p_bar)
            cos_omega_t = (orbit_r + offset_r) * math.cos(omega * p_bar + tau)
            sin_omega_t = (orbit_r + offset_r) * math.sin(omega * p_bar + tau)
            x_goal_bar = cos_omega_t + planet_x
            y_goal_bar = sin_omega_t + planet_y
            v_goal_bar = omega * orbit_r
            psi_goal_bar = np.pi / 2 + omega * p_bar + tau
            vx_goal_bar = v_goal_bar * math.sin(psi_goal_bar)
            vy_goal_bar = v_goal_bar * math.cos(psi_goal_bar)
            goal_bar = np.array([x_goal_bar, y_goal_bar, psi_goal_bar, vx_goal_bar, vy_goal_bar])

            # goal constraint: g(p_bar) = x - x_goal
            C_goal = np.eye(5)
            # dg_dp
            F_goal[0, 0] = -omega * (orbit_r + offset_r) * math.sin(omega * p_bar + tau)
            F_goal[1, 0] = omega * (orbit_r + offset_r) * math.cos(omega * p_bar + tau)
            F_goal[2, 0] = omega
            F_goal[3, 0] = omega * v_goal_bar * math.cos(psi_goal_bar)
            F_goal[4, 0] = -omega * v_goal_bar * math.sin(psi_goal_bar)
            F_goal = -F_goal
            # r_goal
            r_goal = self.X_bar[:5, -1] - goal_bar - C_goal @ self.X_bar[:5, -1] - F_goal @ self.p_bar
        elif isinstance(self.goal, SpaceshipTarget):
            C_goal = np.eye(5)
            F_goal = np.zeros((5, 1))
            r_goal = (
                self.X_bar[:5, -1]
                - self.goal.target.as_ndarray()[:5]
                - C_goal @ self.X_bar[:5, -1]
                - F_goal @ self.p_bar
            )

        self.problem_parameters["C_goal"].value = C_goal.copy()
        self.problem_parameters["F_goal"].value = F_goal.copy()
        self.problem_parameters["r_goal"].value = r_goal.copy()

    def g(self, X, U, p):
        g_tot = np.zeros(5)

        if isinstance(self.goal, SatelliteTarget):
            planet_x = self.goal.planet_x
            planet_y = self.goal.planet_y
            omega = self.goal.omega
            tau = self.goal.tau
            orbit_r = self.goal.orbit_r
            radius = self.goal.radius
            offset_r = self.goal.offset_r

            # g(p_bar)
            cos_omega_t = (orbit_r + offset_r) * math.cos(omega * p[0] + tau)
            sin_omega_t = (orbit_r + offset_r) * math.sin(omega * p[0] + tau)
            x_goal_bar = cos_omega_t + planet_x
            y_goal_bar = sin_omega_t + planet_y
            v_goal_bar = omega * orbit_r
            psi_goal_bar = np.pi / 2 + omega * p[0] + tau
            vx_goal_bar = v_goal_bar * math.sin(psi_goal_bar)
            vy_goal_bar = v_goal_bar * math.cos(psi_goal_bar)
            goal_bar = np.array([x_goal_bar, y_goal_bar, psi_goal_bar, vx_goal_bar, vy_goal_bar])

            g_tot = X[:5, -1] - goal_bar
        elif isinstance(self.goal, SpaceshipTarget):
            g_tot = X[:5, -1] - self.goal.target.as_ndarray()[:5]

        return g_tot

    def s(self, X, U, p):
        # Planets
        planet_names = list(self.planets.keys())
        planet_l = len(planet_names)
        s_planets = np.zeros((planet_l, self.params.K))
        for i in range(planet_l):
            planet = self.planets[planet_names[i]]
            r = planet.radius + self.toggle_params.safety_buffer
            xc = planet.center[0]
            yc = planet.center[1]
            for k in range(self.params.K):
                s_planets[i, k] = r**2 - (X[0, k] - xc) ** 2 - (X[1, k] - yc) ** 2

        # Satellites
        satellite_names = list(self.satellites.keys())
        satellite_l = len(satellite_names)
        s_satellites = np.zeros((satellite_l, self.params.K))
        for i in range(satellite_l):
            planet_name, _ = satellite_names[i].split("/")
            planet = self.planets[planet_name]
            xc_p = planet.center[0]
            yc_p = planet.center[1]
            satellite = self.satellites[satellite_names[i]]
            orbit_r = satellite.orbit_r
            omega = satellite.omega
            tau = satellite.tau
            radius = satellite.radius
            r = radius + self.toggle_params.safety_buffer
            for k in range(self.params.K):
                t = (k / (self.params.K - 1)) * p[0]
                xc = xc_p + orbit_r * math.cos(omega * t + tau)
                yc = yc_p + orbit_r * math.sin(omega * t + tau)
                s_satellites[i, k] = r**2 - (X[0, k] - xc) ** 2 - (X[1, k] - yc) ** 2

        s_tot = np.concatenate([s_planets, s_satellites], axis=0)

        return s_tot

    def linear_cost(self, X, U, p, nu, nu_s, nu_tc):
        cost = 0
        cost += self.params.weight_p[0] * p[0]
        cost -= self.toggle_params.weight_mf * X[7, -1]
        cost += self.params.lambda_nu * np.linalg.norm(nu, ord=1)
        cost += self.params.lambda_nu * np.linalg.norm(nu_s, ord=1)
        cost += self.params.lambda_nu * np.linalg.norm(nu_tc, ord=1)
        return cost

    def nonlinear_cost(self, X, U, p):
        delta = self.integrator.integrate_nonlinear_piecewise(X, U, p) - X
        s_val = self.s(X, U, p)
        s_plus = np.maximum(s_val, np.zeros_like(s_val))
        g_val = self.g(X, U, p)
        return self.linear_cost(X, U, p, delta, s_plus, g_val)

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        X_bar = self.problem_parameters["X_bar"].value
        U_bar = self.problem_parameters["U_bar"].value
        p_bar = self.problem_parameters["p_bar"].value
        X = self.variables["X"].value
        U = self.variables["U"].value
        p = self.variables["p"].value
        nu = self.variables["nu"].value
        nu_s = self.variables["nu_s"].value
        nu_tc = self.variables["nu_tc"].value

        J_bar = self.nonlinear_cost(X_bar, U_bar, p_bar)

        L_star = self.linear_cost(X, U, p, nu, nu_s, nu_tc)

        # ***** DEBUG *****
        # print(f"Convergence: {J_bar} - {L_star} < {self.params.stop_crit * J_bar}")
        # print(f"Convergence: {J_bar - L_star} < {self.params.stop_crit * J_bar}")
        # if J_bar - L_star < 0:
        #    print("Negative convergence cost")
        # *****************

        return J_bar - L_star < self.params.stop_crit * J_bar

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        K = self.params.K
        X_bar = self.problem_parameters["X_bar"].value.copy()
        U_bar = self.problem_parameters["U_bar"].value.copy()
        p_bar = self.problem_parameters["p_bar"].value.copy()
        X = self.variables["X"].value.copy()
        U = self.variables["U"].value.copy()
        p = self.variables["p"].value.copy()
        nu = self.variables["nu"].value.copy()
        nu_s = self.variables["nu_s"].value.copy()
        nu_tc = self.variables["nu_tc"].value.copy()

        tr_radius = self.problem_parameters["tr_radius"].value[0]

        J_bar = self.nonlinear_cost(X_bar, U_bar, p_bar)
        J_star = self.nonlinear_cost(X, U, p)
        L_star = self.linear_cost(X, U, p, nu, nu_s, nu_tc)

        rho = (J_bar - J_star) / (J_bar - L_star)

        if rho < self.params.rho_0:
            # Case 0
            X_bar = X_bar.copy()
            U_bar = U_bar.copy()
            p_bar = p_bar.copy()
            tr_radius = max(self.params.min_tr_radius, tr_radius / self.params.alpha)
        elif rho < self.params.rho_1:
            # Case 1
            X_bar = X.copy()
            U_bar = U.copy()
            p_bar = p.copy()
            tr_radius = max(self.params.min_tr_radius, tr_radius / self.params.alpha)
        elif rho < self.params.rho_2:
            # Case 2
            X_bar = X.copy()
            U_bar = U.copy()
            p_bar = p.copy()
            tr_radius = tr_radius
        else:
            # Case 3
            X_bar = X.copy()
            U_bar = U.copy()
            p_bar = p.copy()
            tr_radius = min(self.params.max_tr_radius, tr_radius * self.params.beta)

        return X_bar, U_bar, p_bar, tr_radius

    def _get_cmds_states(
        self,
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Returns valid sequence of commands and states form computed solutions
        """
        # ts = np.linspace(0, self.variables["p"].value[0], self.params.K)

        tf = self.variables["p"].value[0]
        K = self.params.K
        timestep = tf / (K - 1)
        ts = tuple([i * timestep for i in range(K)])

        F = self.variables["U"].value[0, :]
        dphi = self.variables["U"].value[1, :]

        cmds_list = [SpaceshipCommands(f, dp) for f, dp in zip(F, dphi)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        npstates = self.variables["X"].value.T

        """
        # ***** DEBUG *****
        print("p: ", tf)
        print("F_l: ", F_l)
        print("F_r: ", F_r)
        print("dphi: ", dphi)
        print("nu: ", self.variables["nu"].value.T)
        print("X: ", npstates)
        # *****************
        """

        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)
        return mycmds, mystates

    def _get_commands(
        self,
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Get commands from plan
        """
        K = self.params.K
        X = self.variables["X"].value
        U = self.variables["U"].value
        p = self.variables["p"].value[0]
        ts = tuple(np.linspace(0, p, K).tolist())
        F = self.variables["U"].value[0, :]
        dphi = self.variables["U"].value[1, :]

        cmds_list = [SpaceshipCommands(f, dp) for f, dp in zip(F, dphi)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        npstates = [X[:, k].tolist() for k in range(K)]
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)
        return mycmds, mystates

    @staticmethod
    def _extract_seq_from_array() -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F_l = np.array([0, 1, 2, 3, 4])
        F_r = np.array([0, 1, 2, 3, 4])
        dphi = np.array([0, 0, 0, 0, 0])
        cmds_list = [SpaceshipCommands(l, r, dp) for l, r, dp in zip(F_l, F_r, dphi)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 8)
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)
        return mycmds, mystates

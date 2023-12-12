import numpy as np
import cvxpy as cvx
import sympy as spy
from dataclasses import dataclass, field

from numpy.typing import NDArray

from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.rocket import RocketCommands, RocketState
from dg_commons.sim.models.rocket_structures import RocketGeometry, RocketParameters

from pdm4ar.exercises_def.ex09.utils_params import PlanetParams, SatelliteParams

from pdm4ar.exercises.ex09.rocket import Rocket
from pdm4ar.exercises.ex09.discretization import DiscretizationMethod, FirstOrderHold, ZeroOrderHold

@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """
    # Cvxpy solver parameters
    solver: str = 'MOSEK'                                           # specify solver to use
    verbose_solver: bool = False                                    # if True, the optimization steps are shown
    max_iterations: int = 100                                       # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5                                          # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10*np.array([[1.0]]).reshape((1, -1))) # weight for final time
    
    tr_radius: float = 5                                            # initial trust region radius
    min_tr_radius: float = 1e-4                                     # min trust region radius
    max_tr_radius: float = 100                                      # max trust region radius
    rho_0: float = 0.0                                              # trust region 0
    rho_1: float = 0.25                                             # trust region 1
    rho_2: float = 0.9                                              # trust region 2
    alpha: float = 2.0                                              # div factor trust region update
    beta: float = 3.2                                               # mult factor trust region update

    # Discretization constants
    K: int = 50                                                     # number of discretization steps 
    N_sub: int = 5                                                  # used inside ode solver inside discretization
    stop_crit: float = 1e-5                                         # Stopping criteria constant
class RocketPlanner:
    """
    Feel free to change anything in this class.
    """
    
    planets: list[PlanetParams]
    satellites: list[SatelliteParams]
    rocket: Rocket
    sg: RocketGeometry
    sp: RocketParameters
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

    def __init__(self, planets: dict[PlayerName, PlanetParams], satellites: dict[PlayerName, SatelliteParams], sg: RocketGeometry, sp: RocketParameters):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()

        # Rocket Dynamics
        self.rocket = Rocket(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.rocket, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.rocket, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # Initial Guess
        self.X_bar, self.U_bar, self.p_bar = self.intial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(self, init_state: RocketState, goal_state: DynObstacleState) -> tuple[DgSampledSequence[RocketCommands], DgSampledSequence[RocketState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state

        #
        # TODO: Implement SCvx algorithm or comparable
        #

        self._convexification()
        try:
            error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
        except cvx.SolverError:
            print(f"SolverError: {self.params.solver} failed to solve the problem.")

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array()

        return mycmds, mystates
    
    def intial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K
        
        X = np.zeros((self.rocket.n_x, K))
        U = np.zeros((self.rocket.n_u, K))
        p = np.zeros((self.rocket.n_p))

        return X, U, p

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
            'X': cvx.Variable((self.rocket.n_x, self.params.K)),
            'U': cvx.Variable((self.rocket.n_u, self.params.K)),
            'p': cvx.Variable(self.rocket.n_p)
        }
        
        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            'init_state': cvx.Parameter(self.rocket.n_x)
            # ...
        }
        return problem_parameters

    def _get_constraints(self) -> list[cvx.constraints]:
        """
        Define constraints for SCvx.
        """
        constraints = [
            self.variables['X'][:, 0] == self.problem_parameters['init_state'],
            # ...
        ]
        return constraints

    def _get_objective(self) -> cvx.Problem:
        """
        Define objective for SCvx.
        """
        # Example objective
        objective =  self.params.weight_p@self.variables['p']
        
        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)

        self.problem_parameters['init_state'].value = self.X_bar[:, 0]
        # ...

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        pass

    def _update_trust_region(self) -> float:
        """
        Update trust region radius.
        """
        pass

    @staticmethod
    def _extract_seq_from_array() -> (
        tuple[DgSampledSequence[RocketCommands], DgSampledSequence[RocketState]]
    ):
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F_l = np.array([0, 1, 2, 3, 4])
        F_r = np.array([0, 1, 2, 3, 4])
        dphi = np.array([0, 0, 0, 0, 0])
        cmds_list = [RocketCommands(l, r, dp) for l, r, dp in zip(F_l, F_r, dphi)]
        mycmds = DgSampledSequence[RocketCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 8)
        states = [RocketState(*v) for v in npstates]
        mystates = DgSampledSequence[RocketState](timestamps=ts, values=states)
        return mycmds, mystates
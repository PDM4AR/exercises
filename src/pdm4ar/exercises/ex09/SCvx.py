import numpy as np
import cvxpy as cvx
import time

from numpy.typing import NDArray

from agent import RocketAgent
from utils.discretization import Discretization_Method, FirstOrderHold

# used to format printed lines
def format_line(name, value, unit=''):
    """
    Formats a line e.g.
    {Name:}           {value}{unit}
    """
    name += ':'
    if isinstance(value, (float, np.ndarray)):
        value = f'{value:{0}.{4}}'

    return f'{name.ljust(40)}{value}{unit}'

class Solver_Parameters:
    
    # Cvxpy solver parameters
    solver: str #specify solver used
    verbose_solver: bool #if True, the optimization steps are shown
    max_iterations: int #max iterations to find the optimal of the convex optimization

    # Weight constants
    lambda_nu: float # slack variable weight
    weight_p : NDArray  #t_f weight

    # Trust region radius + update constants
    tr_radius: float #initial trust region radius
    min_tr_radius: float #min trust region radius
    max_tr_radius: float #max trust region radius
    rho_0: float
    rho_1: float
    rho_2: float
    alpha: float # div factor trust region update
    beta: float # mult factor trust region update

    # Discretization constants
    K: int #number of discretization steps 
    N_sub: int #used inside ode solver inside discretization

    # Stopping criteria constant
    stop_crit: float

    def __init__(self):
        
        self.solver = ['ECOS', 'MOSEK'][1]
        self.verbose_solver = False
        self.max_iterations = 100
        self.lambda_nu = 1e5 # 1e6
        self.weight_p = 10*np.array([[1.0]]).reshape((1, -1))
        self.tr_radius = 5
        self.min_tr_radius = 1e-4
        self.max_tr_radius = 100
        self.rho_0 = 0.0
        self.rho_1 = 0.25
        self.rho_2 = 0.9
        self.alpha = 2.0
        self.beta = 3.2
        self.K = 50
        self.N_sub = 5
        self.stop_crit = 1e-5

class SCProblem:

    agent: RocketAgent
    parameters: Solver_Parameters
    integrator: Discretization_Method
    variables: dict
    parameters: dict
    problem: cvx.Problem
    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(self, agent: RocketAgent, map: Map, initial_guess: tuple[NDArray, NDArray, NDArray] = None):
        
        # # Agent
        # self.agent = agent
        # self.agent.normalize()

        # # Map
        # self.map = map
        # self.map.normalize(self.agent.meter_scale)

        # Solver parameters
        self.parameters = Solver_Parameters()

        # Discretization method
        self.integrator = FirstOrderHold(agent, self.parameters.K, self.parameters.N_sub)
        
        # Variables
        self.variables = self._get_variables()

        # Problem parameters
        self.problem_parameters = self._get_problem_parameters()

        # Initial guess
        if initial_guess is None:
            self.X_bar, self.U_bar, self.p_bar = self.agent.initial_guess(self.integrator.K, self.variables['X'].shape, self.variables['U'].shape, self.variables['p'].shape)
        else:
            self.X_bar, self.U_bar, self.p_bar = initial_guess

        self._set_goal(self.agent.tf_guess)

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx optimitation problem
        self.problem = cvx.Problem(objective, constraints)

    def _get_variables(self) -> dict:

        variables = {
            'X': cvx.Variable((self.agent.n_x, self.integrator.K)),
            'U': cvx.Variable((self.agent.n_u, self.integrator.K)),
            'p': cvx.Variable(self.agent.n_p, nonneg=True),
            "nu": cvx.Variable((self.agent.n_x, self.integrator.K-1)),
        }
        
        if self.map.n_nonconvex_constraints > 0:
            variables["nu_s"] = []
            for _ in range(self.map.n_nonconvex_constraints):
                variables["nu_s"].append(cvx.Variable((self.integrator.K, 1), nonneg=True))

        return variables
    
    def _get_problem_parameters(self) -> dict:

        problem_parameters = {
            "x1": cvx.Parameter(self.agent.n_x),
            "A_bar": cvx.Parameter((self.agent.n_x * self.agent.n_x, self.integrator.K-1)),
            "B_plus_bar": cvx.Parameter((self.agent.n_x * self.agent.n_u, self.integrator.K-1)),
            "B_minus_bar": cvx.Parameter((self.agent.n_x * self.agent.n_u, self.integrator.K-1)),
            "F_bar": cvx.Parameter((self.agent.n_x * self.agent.n_p, self.integrator.K-1)),
            "r_bar": cvx.Parameter((self.agent.n_x, self.integrator.K-1)),
            "X_last": cvx.Parameter((self.agent.n_x, self.integrator.K)),
            "U_last": cvx.Parameter((self.agent.n_u, self.integrator.K)),
            "p_last": cvx.Parameter(self.agent.n_p),
            "tr_radius": cvx.Parameter(nonneg=True)
        }

        return problem_parameters
    
    def _get_constraints(self) -> list[cvx.Constraint]:

        # Convex constraints
        constraints = self.agent.get_convex_constraints(self.map, self.variables['X'], self.variables['U'], self.variables['p'], self.problem_parameters["x1"])
        
        # Dynamics X+ = A*x + B_minus*u + B_plus*u+ + F*p + r + nu
        constraints += [
            self.variables['X'][:, k + 1] ==
            cvx.reshape(self.problem_parameters["A_bar"][:, k], (self.agent.n_x, self.agent.n_x)) @ self.variables['X'][:, k]
            + cvx.reshape(self.problem_parameters["B_minus_bar"][:, k], (self.agent.n_x, self.agent.n_u)) @ self.variables['U'][:, k]
            + cvx.reshape(self.problem_parameters["B_plus_bar"][:, k], (self.agent.n_x, self.agent.n_u)) @ self.variables['U'][:, k + 1]
            + cvx.reshape(self.problem_parameters["F_bar"][:, k], (self.agent.n_x, self.agent.n_p)) @ self.variables['p']
            + self.problem_parameters["r_bar"][:, k]
            + self.variables["nu"][:, k]
            for k in range(self.integrator.K-1)
        ]

        # NonConvex linearized constraints
        if self.map.n_nonconvex_constraints > 0:
            constraints += self.map.get_linearized_constraints(self.variables["X"], self.problem_parameters["X_last"], self.p_bar, self.variables["nu_s"], self.agent.radius)

        # Trust region |dx|+|du|+|dp| <= trust_radius
        dx = self.variables['X'] - self.problem_parameters["X_last"]
        du = self.variables['U'] - self.problem_parameters["U_last"]
        dp = self.variables['p'] - self.problem_parameters["p_last"]
        constraints += [cvx.norm(dx, 1) + cvx.norm(du, 1) + cvx.norm(dp, 1) <= self.problem_parameters["tr_radius"]]

        return constraints
    
    def _get_objective(self) -> cvx.Problem:

        # minimize final time
        objective = self.parameters.weight_p@self.variables['p']
        
        # minimize dynamics violations
        objective += self.parameters.lambda_nu * cvx.norm(self.variables["nu"], 1)
        
        # minimize non convex constraints violations
        slack = 0
        for i in range(self.map.n_nonconvex_constraints):
            slack += cvx.sum(self.variables["nu_s"][i])
        objective += self.parameters.lambda_nu * slack

        return cvx.Minimize(objective)
    
    def successive_convexification(self) -> tuple[NDArray, NDArray, NDArray, bool, bool]:
        """
        Solves the Scv problem 
        
        :return all_X_bar: history planned states
        :return all_U_bar: history planned actions
        :return all_p_bar: history planned parameters (final time)
        :return error: True if a solver error occured
        :return converged: True if converged without problems
        """
        
        # Set the final condition
        self.problem_parameters["x1"].value = self.agent.x1

        # Set initial trust radius
        self.problem_parameters["tr_radius"].value = self.parameters.tr_radius

        # Initialization of history matrices
        all_X_bar = [self.X_bar.copy()]
        all_U_bar = [self.U_bar.copy()]
        all_p_bar = [self.p_bar.copy()]

        # Initialize past non linear errors
        self.J_past = None
        print(cvx.installed_solvers())
        converged = False
        solver_error = False
        for it in range(self.parameters.max_iterations):
            
            # used only if we are in the satellite scenarios
            if it != 0 and self.map.n_satellites > 0:

                # Update goal if not fixed
                self._set_goal(self.p_bar[0])

                # Constraints
                constraints = self._get_constraints()

                # Objective
                objective = self._get_objective()

                # Cvx optimitation problem
                self.problem = cvx.Problem(objective, constraints)

            t0_it = time.time()

            print('-' * 50)
            print('-' * 18 + f' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)
            print('-' * 50)

            t0_tm = time.time()
            
            # Assign values to cvx.Parameters (Linearization and discretization of the system)
            self._convexification()
            
            print(format_line('Time for linearization and discretization', time.time() - t0_tm, 's'))
            
            accept_solution = False 
            while True:
                # Solve convex optimization problem
                try:
                    # error = self.problem.solve(verbose=self.parameters.verbose_solver, solver=self.parameters.solver, max_iters=200)
                    error = self.problem.solve(verbose=self.parameters.verbose_solver, solver=self.parameters.solver)
                except cvx.SolverError as error:
                    print(f"Solver Error: {error}")    
                    solver_error = True
                    converged = True
                    break

                print(format_line('Solver Status', self.problem.status))

                # Convex problem unfeasible => increase trust radius
                if self.problem.status == "infeasible" or self.problem.status == "infeasible_inaccurate":
                    self.problem_parameters["tr_radius"].value *= self.parameters.beta
                elif self.problem.status == "unbounded": # Convex problem unbounded => error (the trust region should remove this case)
                    solver_error = True
                    converged = True
                    break
                else: # Convex problem feasible
                    accept_solution, converged = self._done()

                print('-' * 50)

                if accept_solution:
                    print('Done. Solution accepted.')
                    break
                elif self.problem_parameters['tr_radius'].value > self.parameters.max_tr_radius:
                    print('Trust region exceeded maximum. \n')
                    converged = True
                    break
                elif self.problem_parameters['tr_radius'].value < self.parameters.min_tr_radius:
                    print('Trust region exceeded minimum. \n')
                    converged = True
                    break
            print('')
            print(format_line('Time for iteration', time.time() - t0_it, 's'))
            print('')

            if not solver_error:
                all_X_bar.append(self.X_bar.copy())
                all_U_bar.append(self.U_bar.copy())
                all_p_bar.append(self.p_bar.copy())

            if converged:
                print(f'Converged after {it + 1} iterations.')
                break

        t0_tm = time.time()

        print(format_line('Time for nonlinear integration', time.time() - t0_tm, 's'))

        all_X_bar = np.stack(all_X_bar)
        all_U_bar = np.stack(all_U_bar)
        all_p_bar = np.array(all_p_bar)

        if not converged:
            print('Maximum number of iterations reached without convergence.')

        return all_X_bar, all_U_bar, all_p_bar, converged, solver_error

    def set_initial_trajectory(self, X_nl: NDArray, N: int = 1):
        """
        Used in the closed loop setting to update the initial guess. 
        The new initial guess will be a shifted (by N units) version of
        X_nl and U_bar with the last N elements equal to the goal state

        :param X_nl: non linear evolution of the past planned trajectory
        :param N: shifting amounts (linked to the frequency update of the system)
        """

        X = X_nl.copy()
        U = self.U_bar.copy()

        for _ in range(N):
            X = np.roll(X.copy(), -2, axis=1)
            U = np.roll(U.copy(), -2, axis=1)

        for i in range(N):
            X[:, -i-1] = X_nl[:, -1]
            U[:, -i-1] = self.U_bar[:, -1]

        self.X_bar = X.copy()
        self.U_bar = U.copy()

    def _set_goal(self, t_f: float):
        if self.agent.goal != "fixed":
            satellite = self.map.satellites[int(self.agent.goal)]
            satellite.set_centers(self.parameters.K, t_f)
            final_center = satellite.centers[:, -1]
            final_velocity = satellite.v
            planet_center = satellite.mother_planet.center

            connecting_vector = final_center - planet_center
            connecting_vector /= np.linalg.norm(connecting_vector, 2)
            goal = final_center + connecting_vector * 1.1*(satellite.radius + self.agent.radius)

            v_direction = np.array([-connecting_vector[1], connecting_vector[0]])
            v = np.dot(final_velocity, v_direction) * v_direction

            theta = np.arctan2(connecting_vector[1], connecting_vector[0])+np.pi/2

            x1 = self.agent.x1
            print("Old goal: ", x1)

            x1[0:2] = goal
            x1[2] = theta
            x1[4:6] = v
            # x1[4:6] = 0.
            self.problem_parameters["x1"].value = x1.copy()
            
            print("New goal: ", x1)

    def _convexification(self):

        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)

        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar
        self.problem_parameters["X_last"].value = self.X_bar
        self.problem_parameters["U_last"].value = self.U_bar
        self.problem_parameters["p_last"].value = self.p_bar
        
    def _done(self) -> tuple[bool, bool]:

        # Current solution of the convex problem
        X_star = self.variables['X'].value
        U_star = self.variables['U'].value
        p_star = self.variables['p'].value

        # Non linear evolution of the current solution
        X_nl = self.integrator.integrate_nonlinear_piecewise(X_star, U_star, p_star)

        # Error in the dynamics non linearity evaluated through non linear evolution
        dynamics_nonlinear_cost = np.linalg.norm(X_star - X_nl, 1)
        # Error in the obstacles constraints non linearity evaluated through non linear stage cost
        constraints_nonlinear_cost = self.map.get_nonlinear_cost(X_star, U_star, self.agent.radius)
        # Total non linear errors
        J = dynamics_nonlinear_cost + constraints_nonlinear_cost

        print(format_line('Nonlinear Dynamics Cost', dynamics_nonlinear_cost))
        print(format_line('Nonlinear Constraint Cost', constraints_nonlinear_cost))

        # To compute the changes we need a past J
        if self.J_past is None:
            self.X_bar = X_star
            self.U_bar = U_star
            self.p_bar = p_star
            self.J_past = J
            return True, False

        # Error in the dynamics non linearity evaluated through nu
        dynamics_linear_cost = np.linalg.norm(self.variables["nu"].value, 1)
        # Error in the obstacles constraints non linearity evaluated through nu_s
        if self.map.n_nonconvex_constraints > 0:
            constraints_linear_cost = self.map.get_linear_cost(self.variables["nu_s"])
        else:
            constraints_linear_cost = 0
        # Total non linear errors evaluated through slack variables
        L = dynamics_linear_cost + constraints_linear_cost

        actual_improvement = self.J_past - J
        predicted_improvement = self.J_past - L

        print('')
        print(format_line('Dynamic Slack Variable Cost', dynamics_linear_cost))
        print(format_line('Constraint Slack Variable Cost', constraints_linear_cost))
        print('')
        print(format_line('Actual improvement', actual_improvement))
        print(format_line('Predicted improvement', predicted_improvement))
        print('')
        print(format_line('Final time', self.p_bar[0]))
        print('')

        # # Exit criteria: small predicted change between iterations
        # p_change = np.linalg.norm(self.p_bar - p_star, 1)
        # x_change = np.linalg.norm(self.X_bar - X_star, 1)
        
        # if p_change + x_change < self.parameters.stop_crit:
        #     print('Predicted trajectory change very small, accepting solution + converged.')
        #     return True, True

        # Convergence criteria == small predicted improvement
        # if abs(predicted_improvement) < self.parameters.stop_crit and actual_improvement > 0:
        if abs(predicted_improvement) < self.parameters.stop_crit:
            print('Predicted improvement very small, accepting solution + converged.')
            return True, True
        
        accept_solution = self._update_trust_region(actual_improvement/predicted_improvement)

        if accept_solution: # accept solution == new reference for the next iteration
            print('Solution accepted.')

            self.X_bar = X_star
            self.U_bar = U_star
            self.p_bar = p_star
            self.J_past = J

        return accept_solution, False
    
    def _update_trust_region(self, rho: float) -> bool:
        
        # reject solution (actual improvement is negative => trust radius too big)
        if rho < self.parameters.rho_0:
            self.problem_parameters["tr_radius"].value /= self.parameters.alpha
            # self.problem_parameters["tr_radius"].value = max(self.problem_parameters["tr_radius"].value, self.parameters.min_tr_radius)
            print(f'Trust region too large. Solving again with radius={self.problem_parameters["tr_radius"].value}')
            return False 
        
        # accept solution
        if rho < self.parameters.rho_1: # decrease trust radius
            self.problem_parameters["tr_radius"].value /= self.parameters.alpha
            # self.problem_parameters["tr_radius"].value = max(self.problem_parameters["tr_radius"].value, self.parameters.min_tr_radius)
            print(f'Decreasing radius. Solving again with radius={self.problem_parameters["tr_radius"].value}')
        elif rho >= self.parameters.rho_2: # increase trust radius
            self.problem_parameters["tr_radius"].value *= self.parameters.beta
            # self.problem_parameters["tr_radius"].value = min(self.problem_parameters["tr_radius"].value, self.parameters.max_tr_radius)
            print(f'Increasing radius. Solving again with radius={self.problem_parameters["tr_radius"].value}')
        else:
            print(f'Radius unchanged. Solving again with radius={self.problem_parameters["tr_radius"].value}')

        return True

    def print_available_parameters(self):
        print("Parameter names:")
        for key in self.problem_parameters:
            print(f"\t {key}:{self.problem_parameters[key].value}")
        print("\n")

    def print_available_variables(self):
        print("Variable names:")
        for key in self.variables:
            if key != "nu_s":
                print(f"\t {key}:{self.variables[key].value}")
            else:
                print(f"\t {key}:")
                for nu in self.variables[key]:
                    print(f"{nu.value}")
        print("\n")
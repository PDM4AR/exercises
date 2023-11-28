# Spacecraft Obstacle Avoidance Challenge

In this exercise, we are going to land a space rocket on a satellite.
To this end, your task is to integrate a planner into an agent that is working in closed-loop with a simulation environment.
The task is divided in 3 cases:
## todo, high level description of the task


## WIP
- need to defined a performance weighted "cost functions"

## Specifics of the Problem
Your planner is specifically designed to cope with the unfortunate vent of loosing the main thruster.
The goal is to still navigate the rocket past obstacles while reaching a predefined target location, using only the two functional lateral thrusters. 
The rocket's dynamics are represented by the following equations:

1. **Position Dynamics:**
    - $\frac{dx}{dt} = v_x$
    - $\frac{dy}{dt} = v_y$

2. **Orientation Dynamics:**
    - $\frac{d\theta}{dt} = \omega$

3. **Fuel Dynamics:**
    - $\frac{dm}{dt} = -k_l(F_l + F_r)$

4. **Velocity Dynamics:**
    - $\frac{dv_x}{dt} = \frac{1}{m}(\sin(\phi+\theta)F_l + \sin(\phi-\theta)F_r)$
    - $\frac{dv_y}{dt} = \frac{1}{m}(-\cos(\phi+\theta)F_l + \cos(\phi-\theta)F_r)$

5. **Angular Velocity Dynamics:**
    - $\frac{d\omega}{dt} = \frac{l_2}{I}cos(\phi)(F_r - F_l)$
    - $\frac{d\phi}{dt} = v_\phi$

If the rocket's state is represented by $X = [x, y, \theta, m, v_x, v_y, \omega, \phi]'$, and the control inputs are $U = [F_l, F_r, v_\phi]$, we obtain the following dynamics equations:

6. **Dynamics:**
    - $\frac{dX(t)}{dt} = f(X(t), U(t))$

![Example Image](https://github.com/ywerneraris/pdm4ar_scvx/blob/main/images/Rocket.png)
## Hard Constraint

There are various constraints that needs to be satisfied:

- The initial and final inputs needs to be zero: $U(t_0) = U(t_f) = 0$
- The rocket needs to arrive close to the goal: $X(t_f) + \delta X \in X_1$ with $\delta X \in [-\delta, \delta]$
- The rocket needs to dodge every obstacle in its path: $(x, y) \bigoplus \mathcal{X}_{Rocket}(\theta) \notin Obstacle \quad \forall Obstacle \in Obstacles$
- The rocket's mass should be greater than or equal to the mass of the rocket without fuel: $m \geq m_{rocket}$
- Control inputs, $F_l$ and $F_r$, are limited: $F_l, F_r \in [0, F_{\text{max}}]$.
- The thrust angle is limited and coupled between the two lateral thusters: $\phi_l=\phi_r=\phi \in [-\phi_{\text{max}}, \phi_{\text{max}}]$.
- You have a maximum time to reach the goal position: $t_f \leq t_f^{max}$
- The speed of change of $\phi$ is limited: $v_\phi \in [-v^{max}_ϕ ,v^{max}_ϕ ]$

## Scenarios

This challenge can be approached in three different scenarios:

1. **Scenario 1: Static planets and fixed goal**
   - In this scenario, the rocket needs to avoid planets while reaching a fixed final goal, $X_1$.

   ![Example Image](https://github.com/ywerneraris/pdm4ar_scvx/blob/main/images/planets_scenario.png)

2. **Scenario 2: Static Planets with Satellites and Fixed Goal**
   - The rocket must navigate around a planet with multiple moving satellites to reach a fixed final goal, $X_1$.

   ![Example Image](https://github.com/ywerneraris/pdm4ar_scvx/blob/main/images/planets_satellites_scenario.png)

3. **Scenario 3: Planets with Satellites and time varying goal**
   - Similar to Scenario 2, but the final goal is linked with one of the satellites.

   ![Example Image](https://github.com/ywerneraris/pdm4ar_scvx/blob/main/images/planets_satellites_moving_goal_scenario.png)

## Evaluation Metrics

The quality of the rocket's trajectory is evaluated based on several key factors:

0. **Mission achieved** Satisfaction of Mission Goal while satisfying hard constraints

1. **Code Execution Speed:** The efficiency and speed of the control system's execution.

2. **Trajectory Final Time:** The time taken to reach the final goal while avoiding obstacles.

3. **Safety:** Ensuring that the rocket maintains a safe minimum distance from planets and avoids any radioactive areas beyond the map boundaries (penalty based on a potential function high close to the obstacles and outside of the map boundaries)

4. **Satellites Observability:** The ability of the rocket to observe the positions of moving satellites.

## Data Structures
The various data structures needed for the development of the exercise can be inspected in ...

### Mission
Structure storing the individual features of a mission.

    class Mission:

        x0: NDArray
        x1: NDArray
        goal_satellite: Optional[Satellite]
        mission_type: str
        t_f_max: float
        delta: float

        def __init__(self, x0: NDArray, x1: NDArray, goal_satellite: Optional[Satellite], mission_type: str, t_f_max: float, delta: float):
            pass

### Obstacle
Parent class of Planet and Satellite

    class Obstacle:

        radius: float

        def __init__(self, r: float):
            self.radius = r

        def normalize(self):
            pass

        def plot(self, ax: Any, meter_scale: float = 1, k: int = 0):
            pass

### Planet
Structure storing the individual features of a planet.

    class Planet(Obstacle): 

        center: NDArray
        M: float

        def __init__(self, x: float, y: float, r: float, M: float):
            pass
        
        def normalize(self):
            pass
        
        def plot(self, ax: Any, meter_scale: float = 1, k: int = 0):
            pass

### Satellite
Structure storing the individual features of a satellite.

    class Satellite(Obstacle):

        mother_planet: Planet
        v: float
        omega: float
        tau: float
        centers: NDArray

        def __init__(self, tau: float, r: float, mother_planet: Planet, orbit_radius: float):
            pass

        def set_centers(self, K: int, t_f: float):
            """
                Sets the centers of the satellite at each time step
                Recallable after a change of t_f

                x(t) = rcos(wt+tau)+xc
                y(t) = rsin(wt+tau)+yc
            """
            pass
        
        def normalize(self):
            pass
        
        def plot(self, ax: Any, meter_scale: float = 1, k: int = 0):
            pass

### Map
Structure storing the individual features of the map.

    class Map:

        planets: list[Planet]
        satellites: list[Satellite]

        lowerbound: float
        upperbound: float

        def __init__(self, planets: list[Planet] = [], satellites: list[Satellite] = []):
            pass

        def normalize(self, meter_scale):
            pass

### Rocket parameters
Structure storing the individual parameters of the rocket used for testing.

    class Rocket_parameters:

        m: float
        m_fuel: float

        # geometric parameters
        l1: float
        l2: float
        l: float
        b: float
        l3: float
        l_F: float

        # dynamic parameters
        I: float # moment of inertia
        F_L_max: float # max lateral thrust
        phi_max: float # max nozzle angle
        k_lateral_thrust: float # fuel lateral thrust usage coefficient

## Student Task
Your task is to implement the method **solve** of the class

    class Problem:

        mission: Mission
        map: Map

        def __init__(self, mission: Mission, map: Map):
            self.mission = mission
            self.map = map

        def solve(self) -> tuple[NDArray, NDArray, int]:
            """
            Solves the problem 

            :return: Times steps
            :return: Planned actions
            :return: 0 for ZOH, 1 for FOH (discretization used)
            """
            pass
As input, you receive a Mission and a Map, and you need to return a NDArray of planned action and a number representing the discretization used in your planner. The different constraints and the specific cost that your plan must optimize are checked a-posterior once we obtain a matrix of planned actions. You are free to create your own agent class, but keep in mind that the dimensions and constraints parameter defined inside **Rocket_parameters** are used in the a-posterior test of your planned actions.

## Hints
We developed the exercises after reading the following paper ... on SCvx, the planning method used in 2021 by spaceX to land their rocket on a moving platform in the middle of the ocean. We recommend to use such a method to solve the problem but you are free to come up with your own solution. We made available some basic structure to implement the SCvx pypeline 

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
            pass

    class SCProblem:

        agent: Rocket
        map: Map
        parameters: Solver_Parameters
        integrator: Discretization_Method
        variables: dict
        parameters: dict
        problem: cvx.Problem
        X_bar: NDArray
        U_bar: NDArray
        p_bar: NDArray

        def __init__(self, agent: Rocket, map: Map):
            pass

        def _get_variables(self) -> dict:
            pass
        
        def _get_problem_parameters(self) -> dict:
            pass
        
        def _get_constraints(self) -> list[cvx.Constraint]:
            pass
        
        def _get_objective(self) -> cvx.Problem:
            pass
        
        def successive_convexification(self) -> tuple[NDArray, NDArray, NDArray, bool, bool]:
            pass

and a discretization class 

    class Discretization_Method:

        K: int
        N_sub: int 
        range_t: tuple

        n_x: int
        n_u: int
        n_p: int

        f: Any
        A: Any
        B: Any
        F: Any

        def __init__(self, agent: Agent, K: int, N_sub: int):
            pass

    class ZeroOrderHold(Discretization_Method):

        A_bar: NDArray
        B_bar: NDArray
        F_bar: NDArray
        r_bar: NDArray

        x_ind: slice
        A_bar_ind: slice
        B_bar_ind: slice
        F_bar_ind: slice
        r_bar_ind: slice

        P0: NDArray

        def __init__(self, agent: Agent, K: int, N_sub: int):
            pass

        def calculate_discretization(self, X: NDArray, U: NDArray, p: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
            """
            Calculate discretization for given states, inputs and parameter matrices.

            :param X: Matrix of states at all time steps
            :param U: Matrix of inputs at all time steps
            :param p: Vector of parameters
            :return: The discretization matrices
            """
            pass

        def _ode_dPdt(self, P: NDArray, t: float, u: NDArray, p: NDArray) -> NDArray:
            pass
        
    class FirstOrderHold(Discretization_Method):

        A_bar: NDArray
        B_plus_bar: NDArray
        B_minus_bar: NDArray
        F_bar: NDArray
        r_bar: NDArray

        x_ind: slice
        A_bar_ind: slice
        B_plus_bar_end: slice
        B_minus_bar_end: slice
        F_bar_ind: slice
        r_bar_ind: slice

        P0: NDArray
        
        def __init__(self, agent: Agent, K: int, N_sub: int):
            pass

        def calculate_discretization(self, X: NDArray, U: NDArray, p: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
            """
            Calculate discretization for given states, inputs and parameter matrices.

            :param X: Matrix of states at all time steps
            :param U: Matrix of inputs at all time steps
            :param p: Vector of parameters
            :return: The discretization matrices
            """
            pass
        
        def _ode_dPdt(self, P: NDArray, t: float, u_t0: NDArray, u_t1: NDArray, p: NDArray) -> NDArray:
            pass

that implements the method explained inside the paper ....
if you want to work with the SCvx solution, we advice to define an agent with the following structure

    class Agent:

        def __init__(self):
            pass

        def normalize(self):
            pass

        def x_denormalize(self, x: NDArray) -> NDArray:
            pass

        def u_denormalize(self, u: NDArray) -> NDArray:
            pass

        def get_dynamics(self) -> tuple[Any, Any, Any, Any]:
            """
            Compute non linear and linear dynamics
            :return f_func: history planned parameters (final time)
            :return A_func: history planned parameters (final time)
            :return B_func: history planned parameters (final time)
            :return F_func: history planned parameters (final time)
            """
            f = self._compute_dynamics()

            A = sp.simplify(f.jacobian(self.x))
            B = sp.simplify(f.jacobian(self.u))
            F = sp.simplify(f.jacobian(self.p))

            f_func = sp.lambdify((self.x, self.u, self.p), f, 'numpy')
            A_func = sp.lambdify((self.x, self.u, self.p), A, 'numpy')
            B_func = sp.lambdify((self.x, self.u, self.p), B, 'numpy')
            F_func = sp.lambdify((self.x, self.u, self.p), F, 'numpy')

            return f_func, A_func, B_func, F_func

        def _compute_dynamics(self) -> sp.Function:
            pass

        def initial_guess(self, K: int, n_x: tuple[int, int], n_u: tuple[int, int], n_p: tuple[int, int]) -> tuple[NDArray, NDArray,NDArray]:
            pass

        def get_convex_constraints(self, map: Map, X: cvx.Variable, U: cvx.Variable, p: cvx.Variable, x1: cvx.Parameter) -> list[cvx.Constraint]:
            pass

        def plot_position(self, ax: Any, X: NDArray, U: NDArray = None):
            pass

        def plot_action(self, ax: Any, U: NDArray, t_f: float):
            pass

to work with the **Discretization_Method** and the **SCProblem** class easierly.
As you can observe, many data structures have a method called **normalize**. This method can be used in the optimization to make the solver converge to better solutions.

## Available Optimization Tools
If your solution needs to solve an optimization problem, we have added powerful libraries in the container to solve optimization problems. For instance, scipy.optimize, PuLP, cvxpy, and Google OR-Tools. We tested cvxpy with "ECOS" and "MOSEK" as solvers for our SCvx pipeline. If you want to use other optimizers or you are not using SCvx to solve the problem, please consider that we have not tested it.

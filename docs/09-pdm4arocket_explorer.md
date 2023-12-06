# PDM4ARocket Explorer

This exercise tackles a complex problem in aerospace engineering - guiding a spacecraft through a challenging space environment. The goal is to navigate the spacecraft past obstacles to reach a predefined target location using only the two functional lateral thrusters. 

## Task
To this end, your task is to write the planner for an agent that is simulated in closed-loop receiving observations 
(on its state and other obstacles' state) and it is expected to return control commands.

![sim2agent](https://user-images.githubusercontent.com/18750753/144580159-d4d29506-03b2-49b9-b4b8-3cde701cc7d4.png)

You can use any planning algorithm you want. 
However, we have tested and recommend to solve the problem by generating an initial trajectory with one of the sequential convexification approaches.
To this end, we provide a skeleton to complete that you are free to disregard if you were to choose other methods. 
The only important limitation is to leave the `Agent` interface unchanged (or it won't work with the simulator).

<!-- ## Student Task
Your task is to implement a planner (suggestion: **SCvx algorithm**) to solve the constrained problem and provide **RocketCommands** at each time step (0.1 seconds) to the simulator. -->

## Scenarios

This challenge has to be approached in three different scenarios:

1. **Scenario 1: Dodging Planets with a Fixed Goal**
   - In this scenario, the spacecraft needs to avoid planets while reaching a fixed final goal, $X_1$.

   ![Example Image](TODO)

2. **Scenario 2: Dodging a Planet and Its Satellites with a Fixed Goal**
   - The spacecraft must navigate around a planet with multiple moving satellites to reach a fixed final goal, $X_1$.

   ![Example Image](TODO)

3. **Scenario 3: Dodging a Planet and Its Satellites with a Dynamic Goal**
   - Similar to Scenario 2, but the final goal is linked with one of the satellites.

   ![Example Image](TODO)

## Rocket dynamics
Your planner is specifically designed to cope with the unfortunate event of losing the main thruster.
The goal is to still navigate the rocket past obstacles while reaching a predefined target location, using only the two functional lateral thrusters. 
The rocket's dynamics are represented by the following equations:

1. **Position Dynamics:**
    - $\frac{dx}{dt} = v_x$
    - $\frac{dy}{dt} = v_y$

2. **Orientation Dynamics:**
    - $\frac{d\psi}{dt} = \dot{\psi}$

3. **Fuel Dynamics:**
    - $\frac{dm}{dt} = -k_l(F_l + F_r)$

4. **Velocity Dynamics:**
    - $\frac{dv_x}{dt} = \frac{1}{m}(sin(\phi+\psi)F_l + sin(\phi-\psi)F_r)$
    - $\frac{dv_y}{dt} = \frac{1}{m}(-cos(\phi+\psi)F_l + cos(\phi-\psi)F_r)$

5. **Angular Velocity Dynamics:**
    - $\frac{d\dot{\psi}}{dt} = \frac{l_m}{I}cos(\phi)(F_r - F_l)$
    - $\frac{d\phi}{dt} = \dot{\phi}$

If the spacecraft's state is represented by $X = [x, y, \psi, v_x, v_y, \dot{\psi}, \phi, m]'$, and the control inputs are $U = [F_l, F_r, \dot{\phi}]$, we obtain the following dynamics equations:

6. **Dynamics:**
    - $\frac{dX(t)}{dt} = f(X(t), U(t))$

![Rocket Dynamics](https://github.com/PDM4AR/exercises/assets/91316303/e0e3d3ff-5fac-49c3-b312-f7043f711d5a)

## Constraints

There are various constraints that need to be satisfied:

- The initial and final inputs needs to be zero: $U(t_0) = U(t_f) = 0$
- The spacecraft needs to arrive close to the goal.
    - $\left\lVert \begin{bmatrix} x(t) \\ y(t) \end{bmatrix} - \begin{bmatrix} x_{\text{1}} \\ y_{\text{1}} \end{bmatrix} \right\rVert _{2} < \text{pos\_tol}$
- The spacecraft needs to point in a specified direction.
    - $|\psi(t) - \psi_{\text{1}}| < \text{dir\_tol}$
- The spacecraft needs to arrive with a specified velocity.
    - $\left\lVert \begin{bmatrix} v_x(t) \\ v_y(t) \end{bmatrix} - \begin{bmatrix} v_{x_{\text{1}}} \\ v_{y_{\text{1}}} \end{bmatrix} \right\rVert _{2} < \text{vel\_tol}$
- The spacecraft needs to dodge every obstacle in its path: $(x, y) \bigoplus \mathcal{X}_{Rocket}(\psi) \notin Obstacle \quad \forall Obstacle \in Obstacles$
- The spacecraft's mass should be greater than or equal to the mass of the spacecraft without fuel: $m(t) \geq m_{spacecraft}(t) \quad \forall t$
- Control inputs, $F_l$ and $F_r$, are limited: $F_l, F_r \in [0, F_{\text{max}}]$.
- The thrust angle is limited and coupled between the two lateral thusters: $\phi_l=\phi_r=\phi \in [-\phi_{\text{max}}, \phi_{\text{max}}]$.
- You have a maximum time to reach the goal position: $t_f \leq t_f^{max}$
- The speed of change of $\phi$ is limited: $v_\phi \in [-v^{max}_ϕ ,v^{max}_ϕ ]$

## Evaluation Metrics

The quality of the spacecraft's trajectory is evaluated based on several key factors:

0. **Mission achieved** Satisfaction of Mission Goal while satisfying hard constraints

1. **Code Execution Speed:** The efficiency and speed of the control system's execution.

2. **Trajectory Final Time:** The time taken to reach the final goal while avoiding obstacles.

3. **Average Actuation Effort:** The average amount of fuel used to reach the final goal.

4. **Mass Consumption:** The amount of fuel used to reach the final goal.

<!-- 5. **Safety:** Ensuring that the spacecraft maintains a safe minimum distance from planets and avoids any radioactive areas beyond the map boundaries (penalty based on a potential function high close to the obstacles and outside of the map boundaries)

6. **Satellites Observability:** The ability of the spacecraft to observe the positions of moving satellites. -->

You can actually check yourself the function computing the final score in the file `src/pdm4ar/exercises_def/ex09/perf_metrics.py` 

## Data  Structures

The various data structures needed for the development of the exercise can be inspected in the following files:

- RocketState & RocketCommands: `dg_commons/sim/models/rocket.py`
- RocketGeometry & RocketParameters: `dg_commons/sim/models/rocket_structures.py`
- SatelliteParams & PlanetParams: `src/pdm4ar/exercises_def/ex09/utils_params.py`

## Code Structure
The various data structures needed for the development of the exercise can be inspected in the following files:

- **agent.py**: Interface with the simulator.
- **planner.py**: SCvx skeleton.
- **rocket.py**: Helper file for transfer of dynamics between the planner and discretization.
- **discretization.py**: ZeroOrderHold and FirstOrderHold Implementation  for convexification.

## Hints
We developed the exercises based on the following paper ([Convex Optimisation for Trajectory Generation](https://arxiv.org/pdf/2106.09125.pdf)) on SCvx, the planning method used in 2021 by spaceX to land their rocket on a moving platform in the middle of the ocean. We recommend to use such a method to solve the problem but you are free to come up with your own solution. We made available some basic skeleton structure to implement the SCvx pipeline in the **planner.py**. The **discretization.py** file provides an implementation of the ZeroOrderHold and FirstOrderHold that is used in the convexification step of the SCvx pipeline to linearize and discretize around a reference trajectory ([Discretization Performance and Accuracy Analysis for the Powered Descent Guidance Problem](https://www.researchgate.net/publication/330200259_Discretization_Performance_and_Accuracy_Analysis_for_the_Rocket_Powered_Descent_Guidance_Problem) and [A Real-Time Algorithm  for Non-Convex Powered Descent Guidance](https://depts.washington.edu/uwrainlab/wordpress/wp-content/uploads/2020/01/AIAA_SciTech_2020.pdf)).

<!-- In the paper "A Real-Time Algorithm for Non-Convex Powered Descent Guidance" (https://depts.washington.edu/uwrainlab/wordpress/wp-content/uploads/2020/01/AIAA_SciTech_2020.pdf), you will find the use of \textit{Scaling Matrices} to scale states, inputs and parameters to produce numerically well-conditioned optimization problems. Our solution implementation only made use of scaling the parameters, not touching on states and inputs, and converged reliably. We recommend to use the same approach and  only introducing the normalization of states and inputs if you are facing numerical issues. -->

As a general and final advice try to understand the method **before** starting to code.

## Available Optimization Tools
If your solution needs to solve an optimization problem, we have added powerful libraries in the container to solve optimization problems. For instance, scipy.optimize, PuLP, cvxpy and cvxopt. We tested cvxpy with "ECOS" and "MOSEK" as solvers for our SCvx pipeline. If you want to use other optimizers or you are not using SCvx to solve the problem, please consider that we have not tested it.


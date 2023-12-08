# PDM4ARocket Explorer

This exercise tackles a complex problem in space exploration - navigating and landing a spacecraft through challenging space environments. 
The goal is to reach a predefined target location using only the two functional lateral thrusters of the spaceship. 

## Task
Your task is to write the planning stack for a simulated "spaceship agent".
To this end, the agent is coupled with a simulator in closed-loop.
At each simulation step, the agent receives observations (about its state and other obstacles' state) and it is expected to return control commands.

![sim2agent](https://user-images.githubusercontent.com/18750753/144580159-d4d29506-03b2-49b9-b4b8-3cde701cc7d4.png)

You have the freedom to implement any planning algorithm you deem appropriate. 
However, based on experience, we suggest to solve the problem adopting a sequential convexification approach.
In case you choose to use this method, you are provided with a template of the algorithm. 
In case you choose a different approach, make sure to preserve the `Agent` interface (or it won't work with the simulator).

## Scenarios

Your agent will be tested in three different scenarios:

1. **Scenario 1: Dodging Planets with a Static Goal**
   - In this scenario, the spacecraft needs to avoid planets while reaching a fixed final goal, $X_1$.

   ![Planets Image](https://github.com/PDM4AR/exercises/assets/91316303/b8afdb04-2f5a-4236-bde7-09b26dcdfa4e)

2. **Scenario 2: Dodging a Planet and Its Satellites with a Static Goal**
   - The spacecraft must navigate around a planet with multiple moving satellites to reach a fixed final goal, $X_1$.

   ![Satellites Image](https://github.com/PDM4AR/exercises/assets/91316303/395d0a10-98ee-4a56-9cd3-7ce004c91bb5)

3. **Scenario 3: Dodging a Planet and Its Satellites with a Time Varying Goal**
   - Similar to Scenario 2, but the final goal is linked with one of the satellites.

   ![Mov Satellites Image](https://github.com/PDM4AR/exercises/assets/91316303/e775ad18-aa5e-4a83-bf16-b93aeff6c6b0)

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

The rocket you have the control over has two side thrusters where you are able to control the amount of thrust to produce $F_l$ and $F_r$ and the coupled angle of the thrusters $\phi$. The thrusters are mounted centrally on the rocket with an offset of $l_m$ to the CoG of the rocket. The velocity $v_x$ and $v_y$ are the velocities in the x and y direction of the world frame respectively. The angle $\psi$ is the angle of the rocket with respect to the x-axis. The angle $\phi$ is the angle of the thrusters with respect to the rocket. The length of the rocket is $l$.

![Rocket Dynamics](https://github.com/PDM4AR/exercises/assets/91316303/6557d710-f4e1-4f95-95b1-a9a11216eb32)

## Constraints

There are several constraints that need to be satisfied:

- The initial and final inputs needs to be zero: $U(t_0) = U(t_f) = 0$
- The spacecraft needs to arrive close to the goal
    - $\left\lVert \begin{bmatrix} x(t) \\ y(t) \end{bmatrix} - \begin{bmatrix} x_{\text{1}} \\ y_{\text{1}} \end{bmatrix} \right\rVert _{2} < \text{pos\_tol}$
- with a specified orientation.
    - $|\psi(t) - \psi_{\text{1}}| < \text{dir\_tol}$
- The spacecraft needs to arrive with a specified velocity.
    - $\left\lVert \begin{bmatrix} v_x(t) \\ v_y(t) \end{bmatrix} - \begin{bmatrix} v_{x_{\text{1}}} \\ v_{y_{\text{1}}} \end{bmatrix} \right\rVert _{2} < \text{vel\_tol}$
- The spacecraft needs to dodge every obstacle in its path: $(x, y) \bigoplus \mathcal{X}_{Rocket}(\psi) \notin Obstacle \quad \forall Obstacle \in Obstacles$
- The spacecraft's mass should be greater than or equal to the mass of the spacecraft without fuel: $m(t) \geq m_{spacecraft}(t) \quad \forall t$
- Control inputs, $F_l$ and $F_r$, are limited: $F_l, F_r \in [0, F_{\text{max}}]$.
- The thrust angle is limited and coupled between the two lateral thrusters: $\phi_l=\phi_r=\phi \in [-\phi_{\text{max}}, \phi_{\text{max}}]$.
- You have a maximum time to reach the goal position: $t_f \leq t_f^{max}$
- The rate of change of $\phi$ is limited: $v_\phi \in [-v^{max}_ϕ ,v^{max}_ϕ ]$

## Evaluation Metrics

The quality of the spacecraft's trajectory is evaluated based on several key factors:

0. **Mission Accomplishment** You safely reach the goal region.

1. **Planning Efficiency:** We consider the average time spent in the "get_commands" method as a proxy for efficiency and quality of the planner.

2. **Time Taken To Reach the Goal:** The time taken to reach the goal.

3. **Actuation Effort:** The amount (integral) of fuel used to reach the final goal .

4. **Mass Consumption:** The amount of fuel used to reach the final goal.

You can verify more precisely the function computing the final score in  `src/pdm4ar/exercises_def/ex09/perf_metrics.py` 

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


# PDM4ARocket Explorer

This exercise tackles a complex problem in space exploration - navigating and vertically landing a spacecraft through challenging space environments.
The goal is to reach a predefined target location using only the thrusters of the spaceship, which force and direction can be controlled.

## Task

Your task is to write the planning stack for a simulated "spaceship agent".
To this end, the agent is coupled with a simulator in closed-loop.
At each simulation step, the agent receives observations (about its state and other obstacles' state) and it is expected
to return control commands.

<!-- TODO change the image -->
![sim2agent](https://user-images.githubusercontent.com/18750753/144580159-d4d29506-03b2-49b9-b4b8-3cde701cc7d4.png)

You have the freedom to implement any planning algorithm you deem appropriate.
However, based on experience, we suggest to solve the problem adopting a sequential convexification approach.
In case you choose to use this method, you are provided with a template of the algorithm.
In case you choose a different approach, make sure to preserve the `Agent` interface (or it won't work with the
simulator).

## Scenarios

Your agent will be tested in three different scenarios:

1. **Scenario 1: Dodging Planets with Landing Goal**
    - In this scenario, the spacecraft needs to avoid planets while trying to land in a fixed final goal, $X_1$. 

   ![Planets Image](https://github.com/PDM4AR/exercises/img/example1.png)


2. **Scenario 2: Dodging a Planet and Its Satellites with a Static Goal**
    - The spacecraft must navigate around a planet with multiple moving satellites to reach a certain fixed position, $X_1$.

   ![Satellites Image](https://github.com/PDM4AR/exercises/img/example2.png)

3. **Scenario 3: Dodging a Planet and Its Satellites with a Landing Goal**
    - Similar to Scenario 2, but the final goal must be reached by landing.

   ![Mov Satellites Image](https://github.com/PDM4AR/exercises/img/example3.png)

## Rocket dynamics

Your planner is specifically designed to navigate the rocket past obstacles while reaching a predefined target location. On this location the rocket should be able to land vertically, preparing therefore for a future departure. The system you are going to work with is a semplification of the real scenarios that companies such as SpaceX and BlueOrigin are facing.
The rocket's dynamics are represented by the following equations:

1. **Position Dynamics:**
    - $\frac{dx}{dt} = v_x cos(\psi) - v_y sin(\psi)$
    - $\frac{dy}{dt} = v_x sin (\psi) + v_y cos(\psi)$

2. **Orientation Dynamics:**
    - $\frac{d\psi}{dt} = \dot{\psi}$

3. **Fuel Dynamics:**
    - $\frac{dm}{dt} = -C_T * F_{thrust}$

4. **Velocity Dynamics:**
    - $\frac{dv_x}{dt} = \frac{1}{m} cos(\delta)F_{thrust} + \dot{\psi} v_y$
    - $\frac{dv_y}{dt} = \frac{1}{m}sin(\delta)F_{thrust} - \dot{\psi} v_x$

5. **Angular Velocity Dynamics:**
    - $\frac{d\dot{\psi}}{dt} = \- frac{l_r}{I}sin(\delta)F_{thrust}$
    - $\frac{d\psi}{dt} = v_{\delta}$

If the spacecraft's state is represented by $X = [x, y, \psi, v_x, v_y, \dot{\psi}, \delta, m]'$, and the control inputs 
are $U = [F_{thrust}, \dot{\delta}]$, we obtain the following dynamics equations:

6. **Dynamics:**
    - $\frac{dX(t)}{dt} = f(X(t), U(t))$

The rocket you have the control over has one central thruster where you are able to control the amount of thrust to
produce $F_{thrust}$ and the angle of the thruster with respect to the rocket $\delta$. The thruster is mounted centrally on the rocket
with an offset of $l_r$ to the CoG of the rocket. The velocity $v_x$ and $v_y$ are the velocities in the x and y
direction of the rocket frame respectively. The angle $\psi$ is the angle of the rocket with respect to the x-axis. The length of the rocket is $l$.
377128230-92d9b410-3804-42fb-8a2c-952b3a3a20b6

![Rocket Dynamics](https://github.com/PDM4AR/exercises/img/spaceship.png)

## Constraints

There are several constraints that need to be satisfied, [$x_0, y_0$] is the starting location and [$x_1, y_1$] is the goal location:

- The initial and final inputs needs to be zero: $U(t_0) = U(t_f) = 0$
- The spacecraft needs to arrive close to the goal
    - $\left\lVert \begin{bmatrix} x(t_f) \\ y(t_f) \end{bmatrix} - \begin{bmatrix} x_{\text{1}} \\ y_{\text{1}}
      \end{bmatrix} \right\rVert _{2} < \text{pos\_tol}$
- with a specified orientation.
    - $\left\lVert \psi(t_f) - \psi_{\text{1}} \right\rVert _{1} < \text{dir\_tol}$
- The spacecraft needs to arrive with a specified velocity.
    - $\left\lVert \begin{bmatrix} v_x(t_f) \\ v_y(t_f) \end{bmatrix} - \begin{bmatrix} v_{x,1} \\ v_{y,1}
      \end{bmatrix} \right\rVert _{2} < \text{vel\_tol}$
- The spacecraft needs to dodge every obstacle in its path: $(x, y) \bigoplus \mathcal{X}_{Rocket}(\psi) \notin Obstacle
  \quad \forall Obstacle \in Obstacles$
- The spacecraft's mass should be greater than or equal to the mass of the spacecraft without fuel: $m(t) \geq m_
  {spacecraft} \quad \forall t$
- Control inputs, $F_{thrust}$ is limited: $F_{thrust} \in [-F_{\text{max}}, F_{\text{max}}]$.
- The thrust angle is limited: $\delta
  \in [-\delta_{\text{max}}, \delta_{\text{max}}]$.
- You have a maximum time to reach the goal position: $t_f \leq t_f^{max}$
- The rate of change of $\phi$ is limited: $v_\phi \in [-v^{max}_ϕ ,v^{max}_ϕ ]$

## Evaluation Metrics

The quality of the spacecraft's trajectory is evaluated based on several key factors:

0. **Mission Accomplishment** You safely reach the goal region.

1. **Planning Efficiency:** We consider the average time spent in the "get_commands" method as a proxy for efficiency
   and quality of the planner.

2. **Time Taken To Reach the Goal:** The time taken to reach the goal.

3. **Mass Consumption:** The amount of fuel used to reach the final goal.

You can verify more precisely the function computing the final score in  `src/pdm4ar/exercises_def/ex09/perf_metrics.py`

## Data  Structures

The various data structures needed for the development of the exercise can be inspected in the following files:

- SpaceshipState & SpaceshipCommands: `dg_commons/sim/models/spaceship.py`
- SpaceshipGeometry & SpaceshipParameters: `dg_commons/sim/models/spaceship_structure.py`
- SatelliteParams & PlanetParams: `src/pdm4ar/exercises_def/ex11/utils_params.py`

## Code Structure

The various data structures needed for the development of the exercise can be inspected in the following files:

- **agent.py**: Interface with the simulator.
- **planner.py**: SCvx skeleton.
- **rocket.py**: Helper file for transfer of dynamics between the planner and discretization.
- **discretization.py**: ZeroOrderHold and FirstOrderHold Implementation for convexification.

## Hints

We developed the exercises based on the following
paper ([Convex Optimisation for Trajectory Generation](https://arxiv.org/pdf/2106.09125.pdf)) on SCvx, the planning
method used in 2021 by spaceX to land their rocket on a moving platform in the middle of the ocean. We recommend to use
such a method to solve the problem but you are free to come up with your own solution. We made available some basic
skeleton structure to implement the SCvx pipeline in the **planner.py**. The **discretization.py** file provides an
implementation of the ZeroOrderHold and FirstOrderHold that is used in the convexification step of the SCvx pipeline to
linearize and discretize around a reference
trajectory ([Discretization Performance and Accuracy Analysis for the Powered Descent Guidance Problem](https://www.researchgate.net/publication/330200259_Discretization_Performance_and_Accuracy_Analysis_for_the_Rocket_Powered_Descent_Guidance_Problem)
and [A Real-Time Algorithm  for Non-Convex Powered Descent Guidance](https://depts.washington.edu/uwrainlab/wordpress/wp-content/uploads/2020/01/AIAA_SciTech_2020.pdf)).

<!-- In the paper "A Real-Time Algorithm for Non-Convex Powered Descent Guidance" (https://depts.washington.edu/uwrainlab/wordpress/wp-content/uploads/2020/01/AIAA_SciTech_2020.pdf), you will find the use of \textit{Scaling Matrices} to scale states, inputs and parameters to produce numerically well-conditioned optimization problems. Our solution implementation only made use of scaling the parameters, not touching on states and inputs, and converged reliably. We recommend to use the same approach and  only introducing the normalization of states and inputs if you are facing numerical issues. -->

In addition, the docker goal class has a method to return notable points. Try to think how you can use them to create a valid constraints. (We suggest to activate the landing constraints only on the final [5-7] steps).

As a general and final advice try to understand the method **before** starting to code.

## Available Optimization Tools

If your solution needs to solve an optimization problem, we have added powerful libraries in the container to solve
optimization problems. For instance, scipy.optimize, PuLP, cvxpy and cvxopt. We tested cvxpy with "ECOS" and "MOSEK" as
solvers for our SCvx pipeline. If you want to use other optimizers or you are not using SCvx to solve the problem,
**please consider that we have not tested it**.


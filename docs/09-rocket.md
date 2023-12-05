# Spacecraft Obstacle Avoidance Challenge

In this exercise, we are going to guide a space rocket through planets and satellites, eventually landing on one.

## Task
To this end, your task is to write the planner for an agent that is simulated in closed-loop receiving observations 
(on its state and other obstacles' state) and it is expected to return control commands.

![sim2agent](https://user-images.githubusercontent.com/18750753/144580159-d4d29506-03b2-49b9-b4b8-3cde701cc7d4.png)

You can use any planning algorithm you want. 
However, we have tested and recommend to solve the problem by generating an initial trajectory with one of the sequential convexification approaches.
To this end, we provide a skeleton to complete that you are free to disregard if you were to choose other methods. 
The only important limitation is to leave the `Agent` interface unchanged (or it won't work with the simulator).



The task is divided in 3 cases:
## todo, high level description of the task



### The 3 scenarios

This challenge can be approached in three different scenarios:

1. **Scenario 1: Static planets and fixed goal**
   - In this scenario, the rocket needs to avoid the static planets while reaching a fixed final goal region.

   ![Example Image](https://github.com/ywerneraris/pdm4ar_scvx/blob/main/images/planets_scenario.png)

2. **Scenario 2: Static Planets with Satellites and Fixed Goal**
   - The rocket must navigate around a planet with multiple moving satellites to reach a fixed final goal region.

   ![Example Image](https://github.com/ywerneraris/pdm4ar_scvx/blob/main/images/planets_satellites_scenario.png)

3. **Scenario 3: Planets with Satellites and time varying goal**
   - Similar to Scenario 2, but the final goal region is attached to one of the moving satellites.

   ![Example Image](https://github.com/ywerneraris/pdm4ar_scvx/blob/main/images/planets_satellites_moving_goal_scenario.png)






## Rocket dynamics
Your planner is specifically designed to cope with the unfortunate event of losing the main thruster.
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
## Constraints

There are various constraints that needs to be satisfied:

- The initial and final inputs needs to be zero: $U(t_0) = U(t_f) = 0$ #AZ why?

- The rocket needs to arrive in the goal region that is defined with all the state to the goal: $X(t_f) + \delta X \in X_1$ with $\delta X \in [-\delta, \delta]$
- The rocket needs to dodge every obstacle in its path: $(x, y) \bigoplus \mathcal{X}_{Rocket}(\theta) \notin Obstacle \quad \forall Obstacle \in Obstacles$
- The rocket's mass should be greater than or equal to the mass of the rocket without fuel: $m \geq m_{rocket}$
- Control inputs, $F_l$ and $F_r$, are limited: $F_l, F_r \in [0, F_{\text{max}}]$.
- The thrust angle is limited and coupled between the two lateral thusters: $\phi_l=\phi_r=\phi \in [-\phi_{\text{max}}, \phi_{\text{max}}]$.
- You have a maximum time to reach the goal position: $t_f \leq t_f^{max}$
- The speed of change of $\phi$ is limited: $v_\phi \in [-v^{max}_ϕ ,v^{max}_ϕ ]$



## Evaluation Metrics

The quality of the rocket's trajectory is evaluated based on several key factors:

0. **Mission achieved** Satisfaction of Mission Goal while satisfying hard constraints

1. **Code Execution Speed:** The efficiency and speed of the control system's execution.

2. **Trajectory Final Time:** The time taken to reach the final goal while avoiding obstacles.

3. **Safety:** Ensuring that the rocket maintains a safe minimum distance from planets and avoids any radioactive areas beyond the map boundaries (penalty based on a potential function high close to the obstacles and outside of the map boundaries)

4. **Satellites Observability:** The ability of the rocket to observe the positions of moving satellites.

You can actually check yourself the function computing the final score in the file `src/pdm4ar/exercises_def/ex09/perf_metrics.py` 

### Data Structures
The various data structures needed for the development of the exercise can be inspected in ...



#### Obstacle information (Planets, satellites, boundaries)
TODO

#### Rocket parameters
TODO

## Hints
We tested the exercises after reading the following [paper](https://arxiv.org/abs/2106.09125), which is also the core planning method used in 2021 by SpaceX to land their rocket on a moving platform in the middle of the ocean.
Other resources: 
- [Discretization techniques](http://depts.washington.edu/uwrainlab/wordpress/wp-content/uploads/2020/01/AIAA_SciTech_2020.pdf) Section C

Consider carefully reviewing the following files:
- `xxx.py` provides xxx
- `xxx.py` provides xxx
- ....

As a general and final advice try to understand the method **before** starting to code.

## Available Optimization Tools
If your solution needs to solve an optimization problem, we have added powerful libraries in the container to solve optimization problems. 
For instance, scipy.optimize, PuLP, cvxpy, and Google OR-Tools. We tested cvxpy with "ECOS" and "MOSEK" as solvers for our SCvx pipeline. 
If you want to use other optimizers or you are not using SCvx to solve the problem, please consider that we have not tested it.


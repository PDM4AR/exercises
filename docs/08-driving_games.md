# Driving Games [Final '22]

This exercise is the final graded exercise issued for the Fall semester of 2022.

## Problem description
Your task is to implement a planning (and control) stack for a car-like vehicle.
The planning stack needs to safely bring the vehicle inside the goal set (a polygon). 
Unfortunately, just before a truck has lost some of its heavy load on the road, and additional debris are present on the road.  

![image](https://user-images.githubusercontent.com/18750753/207476501-2330675d-d18e-4897-a29f-4ad8ac30d4f0.png)

To test your agent, you are provided with a simulator able to perform closed loop simulations.
The simulator at each time step provides observations to the agent, and it expects commands in return.
See below an image of the environment. In purple the obstacles, in yellow the goal set, in red the vehicle:

![sim2agent](https://user-images.githubusercontent.com/18750753/144580159-d4d29506-03b2-49b9-b4b8-3cde701cc7d4.png)

#### Simulation loop
Take a moment to check out the agent interface in the `exercises/ex08/agent.py` file.

At the beginning of the simulation, the method `on_init` is called.
This provides valuable "static" information to the agent (e.g., agent's name, static obstacles, map,...).

The simulation then enters a loop, where at each time step the simulator calls the method `get_commands`.
The method `get_commands` receives the latest "sensed" observations by the agent and is expected to return control commands.

The observations are computed assuming a 2D Lidar sensor with 360 fields of view. 
This means that they contain information only about the non-occluded players, see the image below for clarification.
Another vehicle in the gray area would not be visible to the agent:

![image](https://user-images.githubusercontent.com/18750753/207558372-afd91da4-4e0d-47a0-ae54-eb6dc7e013f4.png)

The *simulation terminates* upon one of the following cases:
- All the agents reach their goals (you manage to bring the vehicle CoG inside the goal area, other terms in the final state such as speed are irrelevant since the agent gets "deactivated" once it reaches the goal);
- An agent crashes into an obstacle;
- The maximum simulation time is reached.

### Vehicle model
The vehicle is a car-like robot modeled via a kinematic bicycle model (see [here](https://github.com/idsc-frazzoli/dg-commons/blob/master/src/dg_commons/sim/models/vehicle.py#L197) the dynamics equation).

The specific `VehicleState`, `VehicleCommands`, `VehicleGeometry`, and `VehicleParameters` of the vehicle are implemented according to the [dg-commons](https://github.com/idsc-frazzoli/dg-commons) library.
We suggest to get familiar with the required basic structures by navigating the code. 

Note that the simulator will enforce the following constraints:
- **Actuation limits**: The acceleration limits can be found in the `VehicleParameters` object.
- **State constraints**: The speed and steering limits of the vehicle can be found in the `VehicleParameters` object.

If the actuation limits are violated, the simulator will clip the actuation to the limits.
If state constraints are violated, the simulator will set the commands to zero (unless they help to return within the physical constraints).

### Test cases and performance criteria
Your task is to implement the agent in the `exercises/ex08/agent.py` file.

Your solution will then be embodied in one or multiple agents at once. 
Each will receive their own goals, observations, and parameters.

Once you run a simulation a report containing the performance metrics and a visualisation of the episode (make sure to click on the _data_nodes_ tab) is generated.
Performance criteria involve:
- **Safety**: The vehicle should not crash into any obstacle;
- **Completeness**: The vehicle should reach the goal set, if the simulation stops earlier, smaller beeline distance left to the goal is rewarded;
- **Efficiency**: The vehicle should reach the goal set in the shortest time possible driving the shortest possible path;
- **Compliance**: The vehicle should drive as much as possible aligned with traffic lanes, have a look at the [CommonRoad API](https://commonroad-io.readthedocs.io/en/latest/api/scenario/#module-commonroad.scenario.lanelet) definition of lanelet and the related network;
- **Smoothness**: The vehicle should drive smoothly, i.e., without sudden accelerations or steering angles;
- **Computation**: The vehicle should on average take as little as possible to compute new commands.

You can find a precise definition of the performance criteria in `exercises_def/ex08/perf_metrics.py`.
In particular the `reduce_to_score` method defines how the performance metrics are reduced to a single score.

The **test cases** on the server differ to the ones provided only by the config file. This implies, for instance, that the map topology is fixed. 

## Run the exercise
Update your repository running `make update` (refer to [Hello World](01-helloworld.md) for more instructions).

Make sure to **rebuild the container** running the VS Code command (click Ctrl+Shift+P) `Remote-Containers: Rebuild Container` or `Remote-Containers: Rebuild and Reopen in Container`, and then reinstall the *pdm4ar* module running `pip3 install -e [path/to/exercises_repo]` in the VS Code terminal.

Run the exercise with:
```shell
python3 [path/to/]src/pdm4ar/main.py --exercise 08
```
or:
```shell
python3 [path/to/]src/pdm4ar/main.py -e 08
```


### Suggestions

**Planning vs Control rate**
The simulator performs steps at 10 _Hz_. 
While the agent is expected to provide commands every 0.1 _s_, the (re-)planning rate can probably be lower.
Consider decoupling the planning and control rate for better performance overall.

**Early development**: 
Note that running your agent in closed loop might be not the best way to build early prototypes.
Adding custom visualisation might be tedious and creating the animation might take some seconds every time.
We suggest developing and test first your agent's planning on a specific snapshot of the environment.

An example of how to visualise the initial configuration is provided in `exercise_def/ex08/sim_context.py`.

**Test on different instances**:
To avoid hard-coded solutions we will test your submissions on different instances of the environment.
You can make sure that your solution can deal with different instances of the world by changing the parameters in the config files or by creating new ones in the folder `exercises_def/ex08/*`.
If you create new configs, make sure to add them to the `exercises_def/ex08/ex08.py` file in the `get_exercise8` method.

**Test faster**:
To speed up the testing you can reduce the animation resolution by modifying the `dt` and the `dpi` parameters in `exercise_def/ex08/ex08.py`.

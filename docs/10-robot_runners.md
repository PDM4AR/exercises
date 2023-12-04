# Robot Runners [Final '23]

This exercise is the final graded exercise issued for the Fall semester of 2023.

## Problem description
Your task is to implement a planning (and control) stack for a fleet of differential drive robots operating in warehouse.
The planning stack needs to safely bring the robot inside its goal region (modeled as a polygon).  

To test your agent, you are provided with a simulator able to perform closed loop simulations.
The simulator at each time step provides observations to the agent, and it expects commands in return.
See below an image of the environment:

![image](https://github.com/PDM4AR/exercises/assets/18750753/b4445a9d-98b9-4b56-96d2-281de2a97947)

#### Simulation loop
Take a moment to check out the agent interface in the `exercises/ex10/agent.py` file.

At the beginning of the simulation, the method `on_episode_init` is called.
This provides valuable "static" information to the agent (e.g., agent's name, static obstacles, map,...).

The simulation then enters a loop, where at each timestep the simulator calls the method `get_commands` of each agent.
The method `get_commands` receives the latest "sensed" observations by the agent and is expected to return control commands.

![sim2agent](https://user-images.githubusercontent.com/18750753/144580159-d4d29506-03b2-49b9-b4b8-3cde701cc7d4.png)

The observations are computed assuming a 2D Lidar sensor with 360 fields of view. 
This means that they contain information only about the non-occluded players, see the image below for clarification.
Another robot in the gray area would not be visible to the agent:

![image](https://user-images.githubusercontent.com/18750753/207558372-afd91da4-4e0d-47a0-ae54-eb6dc7e013f4.png)

The *simulation terminates* upon one of the following cases:
- All the agents reach their goals: that is, you manage to bring the robot CoG inside the goal area. 
Other terms in the final state such as orientation are irrelevant;
- An agent crashes into an obstacle;
- The maximum simulation time is reached.

### Differential drive model
The robot is a differential drive robot modeled according to the one seen in `Lecture 4: Steering`.

The specific `DiffDriveState`, `DiffDriveCommands`, `DiffDriveGeometry`, and `DiffDriveParameters` of the robot are implemented according to the [dg-commons](https://github.com/idsc-frazzoli/dg-commons) library.
We suggest to get familiar with the required basic structures by navigating the code (usually "Ctrl+click" helps). 

Note that the simulator will enforce the following constraints:
- **Actuation limits**: The min/max rotational velocity of teh wheels can be found in the `DiffDriveParameters` object.

If the actuation limits are violated, the simulator will simply clip the value to the limits.

### Test cases and performance criteria
Your task is to implement the agent in the `exercises/ex10/agent.py` file.

Your solution will then be embodied in one or multiple agents at once. 
Each will receive their own goals, observations, and parameters.
The exercise is designed to have each agent with its own (same) policy acting upon observations and without communication.
Unfortunately, we cannot force this by design, but **we will perform some random checks and take out points for solutions that circumvent this**.

Once you run a simulation a report containing the performance metrics and a visualisation of the episode (make sure to click on the _data_nodes_ tab) is generated.
Performance criteria involve:
- **Safety**: The robot should not crash into any obstacle;
- **Completeness**: The robot should reach the goal set, if the simulation stops earlier, smaller beeline distance left to the goal is rewarded;
- **Efficiency**: The robot should reach the goal set in the shortest time possible driving the shortest possible path;
- **Computation**: The robot should on average take as little as possible to compute new commands.

You can find a precise definition of the performance criteria in `exercises_def/ex10/perf_metrics.py`.
In particular the `reduce_to_score` method defines how the performance metrics are reduced to a single scalar value.

The **test cases** on the server differ to the ones provided only by the config file. This implies, for instance, that the map topology is fixed. 

## Run the exercise
Update your repository running `make update` (refer to [Hello World](01-helloworld.md) for more instructions).

Make sure to **rebuild the container** running the VS Code command (click Ctrl+Shift+P) `Remote-Containers: Rebuild Container` or `Remote-Containers: Rebuild and Reopen in Container`, and then reinstall the *pdm4ar* module running `pip3 install -e [path/to/exercises_repo]` in the VS Code terminal.

Run the exercise with:
```shell
python3 [path/to/]src/pdm4ar/main.py --exercise 10
```
or:
```shell
python3 [path/to/]src/pdm4ar/main.py -e 10
```


### Suggestions

**Planning vs Control rate**
The simulator invokes the `get_commands` method at 10 _Hz_. 
While the agent is expected to provide new commands every 0.1 _s_, the (re-)planning rate can probably be lower.
Consider decoupling the planning and control rate for better performance overall.

**Decentralized solutions and game theory**:
Note that every robot is running _"the same"_ agent that you designed. 
Try to leverage this to your advantage. 
To coordinate and plan in the proximity of other robots you know that they would react exactly as you would. 

The exercise is designed to have each agent with its own (same) policy acting upon observations and without communication.
Unfortunately, we cannot impose the solution to be fully decentralized, but **we will perform some random checks and take out points for solutions that circumvent this**.


**Early development**: 
Note that running your agent in closed loop might be not the best way to build early prototypes.
Adding custom visualisation might be tedious and creating the animation might take some seconds every time.
We suggest developing and test first your agent's planning on a specific snapshot of the environment.

An example of how to visualise the initial configuration is provided in `exercise_def/ex10/utlis_config.py`.

**Test on different instances**:
To avoid hard-coded solutions we will test your submissions on different instances of the environment.
You can make sure that your solution can deal with different instances of the world by changing the parameters in the config files or by creating new ones in the folder `exercises_def/ex10/*`.
If you create new configs, make sure to add them to the `exercises_def/ex10/ex10.py` file in the `get_exercise10` method.

**Test faster**:
To speed up the testing you can reduce the animation resolution by modifying the `dt` and the `dpi` parameters in `exercise_def/ex10/ex10.py`.

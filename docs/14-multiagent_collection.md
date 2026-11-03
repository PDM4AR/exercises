# Multi-agent Goal Collection [Final '25]

## Problem description
Your task is to implement a planning and control system for a fleet of differential drive robots. The robots navigate the environment, collect goals, and deliver them to any collection point. Pickups and deliveries are triggered when the robot's reference point `(x, y)` lies strictly inside the corresponding polygon; overlap of the robot's footprint alone is not sufficient. Each robot has a scenario-defined carrying capacity and automatically unloads its complete inventory at a collection point.

To test your agent, you are provided with a simulation environment. The environment provides observations to each agent at each time step and expects control commands in return. Additionally, an one-time global planning phase occurs before the simulation starts, allowing you to coordinate all agents to maximize overall efficiency.

See below an image of the environment:

![image](img/multiagent_collection_example.png)


## Task Overview

You can find all `TODOs` for this exercise in `src/pdm4ar/exercises/ex14/agent.py`. Feel free to add addition files under `src/pdm4ar/exercises/ex14/` if needed. We will replace the entire `ex14` folder with your submission when evaluating on the server.

### 1. One-time Global Planning

To develop your own global planner, implement the `Pdm4arGlobalPlanner.send_plan(...)` method. The method receives an `Ex14InitSimGlobalObservations` object and generates a serialized global plan in `str` type, which will be broadcast to all agents. You can encode any information you want into this global plan.

Every agent exposes an `Pdm4arAgent.on_receive_global_plan(...)` callback which is called when the global plan is received. You can use this callback to parse the received global plan and initialize your agent's internal states.

To help to define and serialize/deserialize structured message (i.e., the global plan), we provide an example using `Pydantic` in the template code. You can choose to use this method or any other methods you prefer (e.g., JSON, pure string, etc.).

#### Note:
`Pdm4arAgent.on_receive_global_plan(...)` is called **only once** at the beginning of the simulation, after `Pdm4arAgent.on_episode_init(...)` is called for each agent but before the first call to `Pdm4arAgent.get_commands(...)`.

#### Available Data Structures:

`src/pdm4ar/exercises_def/ex14/simulator.py::Ex14InitSimGlobalObservations` extends the base `dg_commons` observation for this exercise and contains:
- Initial positions of all robots
- Locations and values of all goals (as local `ValuedSharedPolygonGoal` objects; each goal also exposes its optional `assigned_robot`)
- Locations of all collection points (as `shared_goals.py::CollectionPoint` objects)
- The carrying capacity of each robot in `agent_capacities`
- All map information (boundaries and static obstacles)

The global planner also receives each robot's model geometry and parameters through `players_obs`. Include capacities, goal values, ownership, and collection-point information in the broadcast if the agents need them: these are not all present in the per-step sensor observations.

`src/pdm4ar/exercises/ex14/agent.py::GlobalPlanMessage` is an example `Pydantic` model for defining a structured global plan message. You can modify it or create your own structure as needed.

### 2. Agent Control
After the global planning phase, the simulation starts. Each robot is controlled by an instance of a `Pdm4arAgent` which inherits from the base class `Agent`. You must implement your agent by:

1. **Overriding `on_episode_init(init_obs: InitSimObservations)`**: This method is called once at the beginning of the simulation for each agent. The `InitSimObservations` object contains:
   - `my_name`: The agent's unique player name
   - `seed`: Random seed for reproducibility
   - `dg_scenario`: The scenario information (boundaries, static obstacles)
   - `goal`: The agent's individual goal (NOT used in this exercise)
   - `model_geometry`: The robot's geometry (`DiffDriveGeometry`)
   - `model_params`: The robot's dynamic parameters (`DiffDriveParameters`)

2. **Overriding `on_receive_global_plan(serialized_msg: str)`**: This method receives the string returned by the global planner's `send_plan(...)` method. You can deserialize it here and store the information for use during execution.

3. **Overriding `get_commands(sim_obs: Ex14SimObservations) -> DiffDriveCommands`**: This method is called every `dt_commands` seconds (0.1s by default) and must return control commands. `Ex14SimObservations` extends the base `dg_commons` observation with:
   - `players`: A mapping of player names to `Ex14PlayerObservations` objects, which contain:
     - `state`: The player's current state (e.g., `DiffDriveState`)
     - `occupancy`: The player's footprint polygon
     - `collected_goal_ids`: The IDs of the goals currently carried by this player
   - `time`: The current simulation time
   - `available_goals`: A mapping of goal IDs to `SharedGoalObservation` objects (only includes goals not yet collected), each containing:
     - `occupancy`: The goal's polygon footprint

:warning: Your own player observation is always included. Other players and available goals are filtered by sensor range and line of sight (see [Observations and Sensors](#obs-sensor)).

#### Simulation loop
At the beginning of the simulation, the method `on_episode_init` is called for each agent.

After initialization and the one-time global-plan broadcast, the simulation advances in 0.01 s steps. It calls `get_commands` every 0.1 s for each robot that has not been disabled, holding the latest command between calls.
The method `get_commands` receives the latest "sensed" observations and is expected to return control commands.

![sim2agent](https://user-images.githubusercontent.com/18750753/144580159-d4d29506-03b2-49b9-b4b8-3cde701cc7d4.png)

#### Observations and Sensors
{:#obs-sensor}
The observations are computed assuming a 2D Lidar sensor with 360-degree field of view and a detection range of **5 meters**. This means they contain information only about non-occluded players and goals. The sensor is implemented using the `FovObsFilter` class, which filters the full observations based on line-of-sight.

Another robot or goal in the gray (occluded) area would not be visible to the agent:

![image](https://user-images.githubusercontent.com/18750753/207558372-afd91da4-4e0d-47a0-ae54-eb6dc7e013f4.png)

In addition to the agent's own observation, `sim_obs.players` includes other robots whose footprints intersect its visible sensor area. Similarly, `sim_obs.available_goals` only includes goals that are:
- Not yet collected by any agent
- Within sensor range and line-of-sight

**Important**: All static obstacles are known beforehand (available in `init_obs.dg_scenario.static_obstacles`), but other robots are dynamic obstacles that must be detected via the sensor.

### Rules and Mechanics

1. **No communication after global planning**: After the initial global planning phase, each robot operates independently based on its local observations and the pre-computed global plan. **No further explicit communication between robots is allowed (no shared global state, shared memory, inter-agent messaging, file sharing, etc.).**

2. **Automatic goal collection**: When a robot with spare capacity enters a goal's polygon, it automatically picks up that goal. This is handled by the `SharedPolygonGoalsManager` in the simulator.

3. **Automatic goal delivery**: When a robot with a non-empty inventory enters a collection point's polygon, it automatically drops off every carried goal. This is also handled by the `SharedPolygonGoalsManager`.

4. **Varying capacity**: Each robot's capacity is defined by its scenario `capacity` field (default: 1) and is available to the global planner through `agent_capacities`. Carried goals remain in that robot's inventory until delivery.

5. **Goal ownership**: A goal may have an `assigned_robot`. Any robot can collect and deliver it, but it counts only when the assigned robot delivers it. If another robot delivers it, the goal earns neither the delivery reward nor its revenue, and the task counts as not accomplished, so the early completion bonus is lost. Goals without an assignment count for any robot.

6. **Goal values**: Every goal has an integer `value` between 10 and 100 dollars. The revenue earned for a goal decays linearly with the time at which it is **delivered** (not picked up):
   ```python
   revenue_i = value_i * max(0.2, 1.0 - t_i / max_sim_time)
   ```
   where `t_i` is the delivery time and `max_sim_time` is 60 s. A late delivery still earns at least 20% of the goal's value, while a goal that is never delivered earns nothing. The order in which the goals are delivered therefore matters: delivering valuable goals early 

7. **Disable after collision**: A robot that collides with a static obstacle or another robot is disabled: subsequent wheel commands cannot move it, and it remains an obstacle for other robots. The simulator stops requesting commands from disabled robots.

8. **Decentralized execution**: Each agent instance runs the same policy you design, acting only on local observations without direct communication.

### Termination Conditions
The *simulation terminates* upon one of the following cases:
- **Physical completion**: All goals have been collected and delivered to collection points
- **Timeout**: Simulation time exceeds `max_sim_time` (60 s)
- **All agents collided**: All robots have collided and are disabled

Physical completion does not check goal ownership. A goal delivered by the wrong robot can therefore end the simulation while still receiving no revenue or reward.

### Differential drive model
The robot is a differential drive robot modeled according to the one seen in `Lecture 4: Steering`.

The specific `DiffDriveState`, `DiffDriveCommands`, `DiffDriveGeometry`, and `DiffDriveParameters` of the robot are implemented according to the [dg-commons](https://github.com/idsc-frazzoli/dg-commons) library.
We suggest getting familiar with the required basic structures by navigating the code (usually "Ctrl+click" helps).

The robot's state (`DiffDriveState`) includes:
- `x`, `y`: Position coordinates
- `psi`: Heading angle

The control commands (`DiffDriveCommands`) specify:
- `omega_l`: Left wheel angular velocity in rad/s. Positive values indicate forward rotation.
- `omega_r`: Right wheel angular velocity in rad/s. Positive values indicate forward rotation.

Note that the simulator will enforce the following constraints:
- **Actuation limits**: The min/max rotational velocity of the wheels can be found in the `DiffDriveParameters` object.

If the actuation limits are violated, the simulator will simply clip the value to the limits. However, the actuation effort penalty will still be computed based on the original command values you provided. See [Performance Criteria and Scoring](#performance-criteria-and-scoring) below for details.

The scenario builder uses wheel limits of ±10 rad/s, giving a maximum straight-line speed of 2 m/s. Commands specify wheel velocities directly.

## Performance Criteria and Scoring
{:#performance-criteria-and-scoring}

Your solution will be evaluated based on multiple criteria. The performance metrics are defined in `exercises_def/ex14/perf_metrics.py`.

### Individual Player Metrics (`PlayerMetrics`)
For each player, the following metrics are tracked:
- **`collided`**: Whether the player crashed into an obstacle or another robot
- **`num_goal_delivered`**: Number of goals successfully delivered by this player (goals assigned to another robot are not counted)
- **`revenue`**: Time-dependent revenue of the goals counted in `num_goal_delivered`
- **`travelled_distance`**: Total distance traveled by the player
- **`actuation_effort`**: Integral of absolute wheel velocities
- **`avg_computation_time`**: Average time taken by the `get_commands` method

### Overall Metrics (`AllPlayerMetrics`)
The overall performance combines all players' metrics:
- **`num_collided_players`**: Total number of players that crashed
- **`num_goals_delivered`**: Total number of deliveries by all players, excluding goals delivered by the wrong robot
- **`total_revenue`**: Sum of the time-dependent revenues of those deliveries (see [Goal values](#rules-and-mechanics))
- **`total_travelled_distance`**: Sum of distances traveled by all players
- **`max_sim_time`**: Maximum simulation time allowed
- **`task_accomplishment_time`**: Total time taken to deliver all goals (time when last goal was delivered). It equals `max_sim_time` if any goal is not delivered
- **`total_actuation_effort`**: Sum of actuation efforts of all players
- **`avg_computation_time`**: Arithmetic mean of the per-player average `get_commands` wall-clock times
- **`global_planning_time`**: Total time spent on global planning (execution time of `send_plan` method)

### Score Function
The final score is computed by the `reduce_to_score()` method in `AllPlayerMetrics`:

```python
score = num_goals_delivered * 100
score += total_revenue * 1.0
score -= num_collided_players * 500
score += (max_sim_time - task_accomplishment_time) * 10
score -= total_travelled_distance * 0.5
score -= total_actuation_effort * 0.01
score -= max(0.0, avg_computation_time - 0.1) * 100
score -= max(0.0, global_planning_time - 60) * 10
```

**Key takeaways**:
- **Objective**: Maximize the complete score. Each correct delivery earns 100 points plus revenue; completing every goal also unlocks the time bonus. Revenue alone is not the objective.
- **Collision penalty**: 500 point penalty per collided robot. Avoid collisions!
- **Plan the order**: Earlier delivery of valuable goals increases revenue, so plan accordingly
- **Respect assignments**: A goal delivered by a robot it is not assigned to earns nothing.
- **Early completion bonus**: Deliver every goal with the correct robot to earn 10 points per second saved. If even one goal is missing completion time is 60 s and the bonus is zero
- **Efficiency matters**:
  - Minimize travel distance (0.5 point penalty per unit distance - highest weight among efficiency metrics)
  - Minimize actuation effort (0.01 point penalty per unit effort)
- **Computation time budget**: The penalty is `100 * max(0, avg_computation_time - 0.1)`. It applies to the mean defined above, not separately to every slow call. 
- **Global planning budget**: Keep global planning under 60 seconds (10 point penalty per second over budget). But sometimes spending more time here can be worth it if it leads to better overall performance.
- **Higher scores are better**

### Reference Scores
TO BE GENERATED

## Test cases
The test cases on the server differ from the ones provided locally only by configuration parameters (number of robots, number of goals, map layout, etc.). Your solution should be general enough to handle different scenarios. But don't be worried, the topology of the environment will remain similar to the ones provided in the local tests.

### Generate new test cases
You can create new test cases by adding new configuration files in the `exercises_def/ex14/` folder and adding them to the list of available configurations in the `exercises_def/ex14/ex14.py` file.

:sparkles: We provide a script to generate random configurations for this exercise at `exercises_def/ex14/random_config.py`. Feel free to modify and use it to create new test cases for your own testing.

## Run the exercise

Run the exercise with:
```shell
python path/to/src/pdm4ar/main.py --exercise 14
```
or:
```shell
python path/to/src/pdm4ar/main.py -e 14
```

## View the report
Once you run a simulation, a report containing the performance metrics and a visualization of the episode is generated. Click the `data nodes` button in the top bar of the report to show the generated visualizations.

![image](img/data_nodes_report.png)

## Suggestions and Tips

**Planning vs Control rate**:
The simulator invokes the `get_commands` method at 10 Hz (every 0.1s). While the agent is expected to provide new commands every 0.1s, the (re-)planning rate can probably be lower. Consider decoupling the planning and control rate for better performance overall. For example, you might recompute paths every 0.5s but update control commands every 0.1s.

**Decentralized solutions and game theory**:
Note that every robot is running _"the same"_ agent that you designed. Try to leverage this to your advantage. To coordinate and plan in the proximity of other robots, you know that they would react exactly as you would. This can be used for implicit coordination without communication.

**Early development**:
Note that running your agent in closed loop might not be the best way to build early prototypes. Adding custom visualization might be tedious and creating the animation might take some seconds every time. We suggest developing and testing first your agent's planning on a specific snapshot of the environment.

**Test on different instances**:
To avoid hard-coded solutions, we will test your submissions on different instances of the environment. You can make sure that your solution can deal with different instances by changing the parameters in the config files or by creating new ones in the folder `exercises_def/ex14/*`. If you create new configs, make sure to add them to the `exercises_def/ex14/ex14.py` file in the appropriate method.

**Test faster**:
The standard student runner renders an animation after each simulation. Its rendering parameters are in `exercises_def/ex14/ex14.py::_ex14_vis`. Animation rendering is separate from scoring and is not part of `global_planning_time` or the per-command computation metric.

**Use the global planner wisely**:
The one-time global planning phase is powerful. Consider:
- Task allocation: Which robot should collect which goals? and in which order?
- Path planning: Pre-compute collision-free paths or waypoints
- Coordination: Design strategies to avoid deadlocks and conflicts

**Handle dynamic obstacles**:
While static obstacles are known in advance, other robots are dynamic obstacles. Your control policy in `get_commands` should:
- React to observed robots in real-time
- Use the global plan as a high-level guide
- Implement local collision avoidance

**Monitor goal availability**:
The `sim_obs.available_goals` dictionary only includes uncollected goals currently visible to the sensor. A goal can disappear because it was collected, left sensor range, or became occluded. Keep the global plan's goal information and track observed pickup/inventory events. The agent's own `collected_goal_ids` tuple is always available in its player observation.

**Debug using observations**:
You can return extra information for visualization using the `on_get_extra()` method in your agent. This information can be visualized in the report.

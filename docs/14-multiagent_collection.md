# Multi-agent Goal Collection [Final '25]

## Problem description
Your task is to implement a planning and control stack for a fleet of differential drive robots operating in a warehouse environment to collect goals scattered around the area and deliver them to designated collection points. The robots automatically pick up goals when they move close enough and drop them off when they reach a collection area. Each robot can carry maximum **ONE** goal at a time.

To test your agent, you are provided with a simulation environment. The environment provides observations to each agent at each time step and expects control commands in return. Additionally, an one-time global planning phase occurs before the simulation starts, allowing you to coordinate all agents to maximize overall efficiency.

See below an image of the environment:

![image](https://github.com/PDM4AR/exercises/assets/18750753/b4445a9d-98b9-4b56-96d2-281de2a97947)

## Rules
1. Before the start of the simulation, there will be a one-time global planning phase where you can plan an overall strategy for all robots and the plan will be broadcasted to all robots.

2. After the global planning phase, each robot will operate individually based on its local observations and the global plan. **No further communication or coordination between robots is allowed after the initial global planning phase (No global variables, no shared memory, no inter-agent messaging, no file sharing, etc.).**

## Task Overview

You can find all `TODOs` for this exercise in `src/pdm4ar/exercises/ex14/agent.py`. Feel free to add addition files under `src/pdm4ar/exercises/ex14/` if needed. We will replace the entire `ex14` folder with your submission when evaluating on the server.

### 1. One-time Global Planning Phase

To develop your own global planner, implement the `Pdm4arGlobalPlanner.send_plan(...)` method. The method receives an `InitSimGlobalObservations` object and generates a serialized global plan in `str` type, which will be broadcasted to all agents. You can encode any information you want into this global plan.

Every agent exposes an `Pdm4arAgent.on_receive_global_plan(...)` callback which is called when the global plan is received. You can use this callback to parse the received global plan and initialize your agent's internal states.

To help to define and serialize/deserialize structured message (i.e., the global plan), we provide an example using `Pydantic` in the template code. You can choose to use this method or any other methods you prefer (e.g., JSON, pure string, etc.).

#### Note:
`Pdm4arAgent.on_receive_global_plan(...)` is called **only once** at the beginning of the simulation, after `Pdm4arAgent.on_episode_init(...)` is called for each agent but before the first call to `Pdm4arAgent.get_commands(...)`.

#### Available Data Structures:

`InitSimGlobalObservations` contains the following information:
- Initial positions of all robots
- Locations of all goals (as `SharedPolygonGoal` objects)
- Locations of all collection points (as `CollectionPoint` objects)
- All map information (boundaries and static obstacles)

For more details, refer to 
- `dg_commons/sim/simulator_structures.py::InitSimGlobalObservations`
- `dg_commons/sim/shared_goals.py::SharedPolygonGoal`
- `dg_commons/sim/shared_goals.py::CollectionPoint`

### 2. Per-step Agent Control
After the global planning phase, the simulation starts. Each robot is controlled by an instance of a class inheriting from the base class `Agent`. You must implement your agent by:

1. **Overriding `on_episode_init(init_obs: InitSimObservations)`**: This method is called once at the beginning of the simulation for each agent. The `InitSimObservations` object contains:
   - `my_name`: The agent's unique player name
   - `seed`: Random seed for reproducibility
   - `dg_scenario`: The scenario information (boundaries, static obstacles)
   - `goal`: The agent's individual goal (if any - not used in this exercise)
   - `model_geometry`: The robot's geometry (`DiffDriveGeometry`)
   - `model_params`: The robot's dynamic parameters (`DiffDriveParameters`)

2. **Overriding `on_receive_global_plan(serialized_msg: str)`**: This method receives the string returned by the global planner's `send_plan(...)` method. You can deserialize it here and store the information for use during execution.

3. **Overriding `get_commands(sim_obs: SimObservations) -> DiffDriveCommands`**: This method is called every `dt_commands` seconds (0.1s by default) and must return control commands. The `SimObservations` object contains:
   - `players`: A mapping of player names to `PlayerObservations` objects, which contain:
     - `state`: The player's current state (e.g., `DiffDriveState`)
     - `occupancy`: The player's footprint polygon
     - `collected_goal_id`: The ID of the goal currently being carried by this player (if any)
   - `time`: The current simulation time
   - `available_goals`: A mapping of goal IDs to `SharedGoalObservation` objects (only includes goals not yet collected), each containing:
     - `occupancy`: The goal's polygon footprint

   You can access your own state with: `my_current_state = sim_obs.players[self.name].state`

#### Simulation loop
At the beginning of the simulation, the method `on_episode_init` is called for each agent.

The simulation then enters a loop, where at each timestep the simulator calls the `get_commands` method of each agent.
The method `get_commands` receives the latest "sensed" observations and is expected to return control commands.

![sim2agent](https://user-images.githubusercontent.com/18750753/144580159-d4d29506-03b2-49b9-b4b8-3cde701cc7d4.png)

#### Observations and Sensors
The observations are computed assuming a 2D Lidar sensor with 360-degree field of view. This means they contain information only about non-occluded players and goals. The sensor is implemented using the `FovObsFilter` class, which filters the full observations based on line-of-sight.

Another robot or goal in the gray (occluded) area would not be visible to the agent:

![image](https://user-images.githubusercontent.com/18750753/207558372-afd91da4-4e0d-47a0-ae54-eb6dc7e013f4.png)

The available observations in `sim_obs.players` will only include robots within sensor range and line-of-sight. Similarly, `sim_obs.available_goals` will only include goals that are:
- Not yet collected by any agent
- Within sensor range and line-of-sight

**Important**: All static obstacles are known beforehand (available in `init_obs.dg_scenario.static_obstacles`), but other robots are dynamic obstacles that must be detected via the sensor.

### Rules and Mechanics

1. **No communication after global planning**: After the initial global planning phase, each robot operates independently based on its local observations and the pre-computed global plan. No further communication or coordination is allowed.

2. **Automatic goal collection**: When a robot that is not currently carrying a goal comes within range of a goal (i.e., the robot's position enters the goal's polygon), it automatically picks up that goal. This is handled by the `SharedPolygonGoalsManager` in the simulator.

3. **Automatic goal delivery**: When a robot carrying a goal enters a collection point's polygon, it automatically drops off the goal. This is also handled by the `SharedPolygonGoalsManager`.

4. **Single goal capacity**: Each robot can carry at most one goal at a time. To pick up another goal, it must first deliver its current goal to a collection point.

5. **Decentralized execution**: Each agent instance runs the same policy you design, acting only on local observations without direct communication.

### Termination Conditions
The *simulation terminates* upon one of the following cases:
- **Success**: All goals have been collected and delivered to collection points
- **Timeout**: The maximum simulation time (`max_sim_time`) is reached
- **Collision**: An agent crashes into a static obstacle or another robot
- **All agents disabled**: All agents have collided

Note: After the first collision, the simulation continues for `sim_time_after_collision` seconds (default 0s) to allow other agents to complete their tasks.

### Differential drive model
The robot is a differential drive robot modeled according to the one seen in `Lecture 4: Steering`.

The specific `DiffDriveState`, `DiffDriveCommands`, `DiffDriveGeometry`, and `DiffDriveParameters` of the robot are implemented according to the [dg-commons](https://github.com/idsc-frazzoli/dg-commons) library.
We suggest getting familiar with the required basic structures by navigating the code (usually "Ctrl+click" helps).

The robot's state (`DiffDriveState`) includes:
- `x`, `y`: Position coordinates
- `psi`: Heading angle
- `vl`, `vr`: Left and right wheel velocities

The control commands (`DiffDriveCommands`) specify:
- `omega_l`: Left wheel angular velocity
- `omega_r`: Right wheel angular velocity

Note that the simulator will enforce the following constraints:
- **Actuation limits**: The min/max rotational velocity of the wheels can be found in the `DiffDriveParameters` object.

If the actuation limits are violated, the simulator will simply clip the value to the limits.

## Performance Criteria and Scoring

Your solution will be evaluated based on multiple criteria. The performance metrics are defined in `exercises_def/ex14/perf_metrics.py`.

### Individual Player Metrics (`PlayerMetrics`)
For each player, the following metrics are tracked:
- **`collided`**: Whether the player crashed into an obstacle or another robot
- **`num_goal_delivered`**: Number of goals successfully delivered by this player
- **`travelled_distance`**: Total distance traveled by the player
- **`waiting_time`**: Total time elapsed before goals are delivered (sum of delivery times)
- **`actuation_effort`**: Integral of absolute wheel velocities (integral of |omega_l| + integral of |omega_r|)
- **`avg_computation_time`**: Average time taken by the `get_commands` method

### Overall Metrics (`AllPlayerMetrics`)
The overall performance combines all players' metrics:
- **`num_collided_players`**: Total number of players that crashed
- **`num_goals_delivered`**: Total number of goals delivered by all players
- **`total_travelled_distance`**: Sum of distances traveled by all players
- **`total_waiting_time`**: Sum of waiting times for all goals
- **`total_actuation_effort`**: Sum of actuation efforts of all players
- **`avg_computation_time`**: Average computation time across all players

### Score Function
The final score is computed by the `reduce_to_score()` method in `AllPlayerMetrics`:

```python
score = num_goals_delivered * 100
score -= num_collided_players * 500
score -= total_travelled_distance * 0.1
score -= total_waiting_time * 0.1
score -= total_actuation_effort * 0.1
score -= avg_computation_time * 100
```

**Key takeaways**:
- **Primary objective**: Maximize the number of goals delivered (100 points each)
- **Critical penalty**: Avoid collisions (500 point penalty per collision)
- **Efficiency matters**: Minimize travel distance, waiting time, actuation effort, and computation time
- **Higher scores are better**

Additionally, the global planning execution time is measured and stored in `sim_context.global_plan_execution_time` (though not currently included in the score function, it may be considered for evaluation).

## Test cases
The **test cases** on the server differ from the ones provided locally only by configuration parameters (number of robots, number of goals, map layout, etc.). Your solution should be general enough to handle different scenarios.

Once you run a simulation, a report containing the performance metrics and a visualization of the episode (make sure to click on the _data_nodes_ tab) is generated.

## Run the exercise
Update your repository running `make update` (refer to [Hello World](01-helloworld.md) for more instructions).

Make sure to **rebuild the container** running the VS Code command (click Ctrl+Shift+P) `Remote-Containers: Rebuild Container` or `Remote-Containers: Rebuild and Reopen in Container`, and then reinstall the *pdm4ar* module running `pip3 install -e [path/to/exercises_repo]` in the VS Code terminal.

Run the exercise with:
```shell
python3 [path/to/]src/pdm4ar/main.py --exercise 14
```
or:
```shell
python3 [path/to/]src/pdm4ar/main.py -e 14
```

## Implementation Guide

### Required Classes

You need to implement two classes in `exercises/ex14/agent.py`:

1. **`Pdm4arGlobalPlanner(GlobalPlanner)`**: Implements the one-time global planning
   - Override `send_plan(init_sim_obs: InitSimGlobalObservations) -> str`
   - This method has access to all robots' initial positions and all goals' locations
   - Return a serialized string containing your global plan

2. **`Pdm4arAgent(Agent)`**: Implements the per-robot control policy
   - Override `on_episode_init(init_sim_obs: InitSimObservations)` to initialize your agent
   - Override `on_receive_global_plan(serialized_msg: str)` to receive and parse the global plan
   - Override `get_commands(sim_obs: SimObservations) -> DiffDriveCommands` to compute control commands

### Example Implementation Structure

The template provides an example using Pydantic models for serialization:

```python
class GlobalPlanMessage(BaseModel):
    # Define your global plan structure here
    # Example fields:
    fake_id: int
    fake_name: str
    fake_np_data: NDArray

class Pdm4arGlobalPlanner(GlobalPlanner):
    def send_plan(self, init_sim_obs: InitSimGlobalObservations) -> str:
        # Access initial information
        # init_sim_obs.initial_states: dict of agent initial states
        # init_sim_obs.shared_goals: dict of all goals
        # init_sim_obs.collection_points: dict of collection points
        # init_sim_obs.dg_scenario: map and obstacles

        # TODO: Implement your global planning here

        # Serialize and return
        global_plan = GlobalPlanMessage(...)
        return global_plan.model_dump_json(round_trip=True)

class Pdm4arAgent(Agent):
    def on_receive_global_plan(self, serialized_msg: str):
        # Deserialize the global plan
        global_plan = GlobalPlanMessage.model_validate_json(serialized_msg)
        # Store relevant information for use in get_commands

    def get_commands(self, sim_obs: SimObservations) -> DiffDriveCommands:
        # Access observations
        my_state = sim_obs.players[self.name].state
        other_players = {name: obs for name, obs in sim_obs.players.items() if name != self.name}
        available_goals = sim_obs.available_goals  # Goals not yet collected

        # Check if carrying a goal
        my_obs = sim_obs.players[self.name]
        carrying_goal_id = my_obs.collected_goal_id

        # TODO: Implement your control logic here

        return DiffDriveCommands(omega_l=omega1, omega_r=omega2)
```

## Suggestions

**Planning vs Control rate**:
The simulator invokes the `get_commands` method at 10 Hz (every 0.1s). While the agent is expected to provide new commands every 0.1s, the (re-)planning rate can probably be lower. Consider decoupling the planning and control rate for better performance overall. For example, you might recompute paths every 0.5s but update control commands every 0.1s.

**Decentralized solutions and game theory**:
Note that every robot is running _"the same"_ agent that you designed. Try to leverage this to your advantage. To coordinate and plan in the proximity of other robots, you know that they would react exactly as you would. This can be used for implicit coordination without communication.

The exercise is designed to have each agent with its own (same) policy acting upon observations and without communication (after global planning). Unfortunately, we cannot impose the solution to be fully decentralized by design, but **we will perform random checks and take points off for solutions that circumvent this**.

**Early development**:
Note that running your agent in closed loop might not be the best way to build early prototypes. Adding custom visualization might be tedious and creating the animation might take some seconds every time. We suggest developing and testing first your agent's planning on a specific snapshot of the environment.

You can access the scenario and initial states in both the global planner and individual agents to create static visualizations for debugging.

**Test on different instances**:
To avoid hard-coded solutions, we will test your submissions on different instances of the environment. You can make sure that your solution can deal with different instances by changing the parameters in the config files or by creating new ones in the folder `exercises_def/ex14/*`. If you create new configs, make sure to add them to the `exercises_def/ex14/ex14.py` file in the appropriate method.

**Test faster**:
To speed up testing, you can reduce the animation resolution or disable visualization during development. Check the configuration parameters in `exercises_def/ex14/ex14.py`.

**Use the global planner wisely**:
The one-time global planning phase is powerful but also contributes to your score via computation time. Consider:
- Task allocation: Which robot should collect which goals?
- Path planning: Pre-compute collision-free paths or waypoints
- Coordination: Design strategies to avoid deadlocks and conflicts

**Handle dynamic obstacles**:
While static obstacles are known in advance, other robots are dynamic obstacles. Your control policy in `get_commands` should:
- React to observed robots in real-time
- Use the global plan as a high-level guide
- Implement local collision avoidance

**Monitor goal availability**:
The `sim_obs.available_goals` dictionary only includes goals that haven't been collected yet. If a goal you were planning to collect disappears from this dictionary, it means another robot collected it first. Your agent should be able to adapt and select a different goal.

**Debug using observations**:
You can return extra information for visualization using the `on_get_extra()` method in your agent. This information will be logged and can be visualized in the report (though the default visualization may not show it).

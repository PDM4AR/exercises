# Final '22

This exercise is the final graded exercise issued for the Fall semester of 2022.

## Problem description

To test your agent, you are provided with a simulator able to perform closed loop simulations.
The simulator at each time step provides observations to the agent, and it expects commands in return.

![sim2agent](https://user-images.githubusercontent.com/18750753/144580159-d4d29506-03b2-49b9-b4b8-3cde701cc7d4.png)

#### Simulation loop
Take a moment to check out the agent interface in the `exercises/ex08/agent.py` file.

At the beginning of the simulation, the method `on_init` is called.
This provides valuable "static" information to the agent (e.g., agent's name, static obstacles, map,...).

The simulation then enters a loop, where at each time step the simulator calls the method `get_commands`.
The method `get_commands` receives the latest "sensed" observations by the agent and is expected to return control commands.

The observations are computed assuming a 2D Lidar sensor with 360 fields of view. 
This means that they contain information only about the non-occluded players, see the image below for clarification:

The simulation terminates upon one of the following cases:
- The agent reaches the goal (you manage to bring the Spacecraft CoG inside the goal area)
- The agent crashes into an obstacle
- The maximum simulation time is reached

### Vehicle model
The vehicle model has a left and right thruster at the back that can be activated to push the spacecraft forward or backward.
Note that applying differential thrust will also cause the spacecraft to rotate.


State, commands, vehicle geometry, and parameters of the vehicle are implemented according to the [dg-commons](https://github.com/idsc-frazzoli/dg-commons) library.
We suggest to get familiar with the basic structures by navigating the code. 

Note that the simulator will enforce the following constraints:
- **Actuation limits**: The acceleration of each thruster is saturated in the interval [-10, 10] m/s^2
- **State constraints**: Linear velocities cannot exceed the interval [-50, 50] km/h
- **State constraints**: Angular velocities cannot exceed the interval [-2pi, 2pi] rad/s

If state constraints are violated, the simulator will set the commands to zero (unless they help to return within the physical constraints).


**Early development**: 
Note that running your agent paired in closed loop might be not the best way to build early prototypes.
Adding custom visualisation might be tedious and creating the animation might take a few seconds every time.
We suggest developing and test first your agent on a specific snapshot of the environment.
This, for example, allows you to add custom visualisations to debug your algorithm.
A simple example is provided in `test_agent.py`.

**Test on different instances**:
To avoid hard-coded solutions we will test your submissions on different instances of the environment.
You can make sure that your solution can deal with different instances of the world by changing the parameters that create the space/obstacles/goal region and different initial conditions in the file `exercises_def/final21/scenario.py`.

### Test cases and performance criteria
Your solution will be benchmarked against xxx scenarios. 
One containing only static obstacles (asteroids), one containing also dynamic obstacles (asteroids).
![image](https://user-images.githubusercontent.com/18750753/144765049-ffed6186-8269-4380-b382-a8e049ca7d39.png)
Once you run a simulation a report containing the visualisation of the episode and a few performance metrics is generated.
We will generate a ranking based on the performance criteria.



## Run the exercise
Update your repository running `make update` (refer to [Hello World](01-helloworld.md) for more instructions).

Make sure to rebuild the container running the VS Code command (click Ctrl+Shift+P) `Remote-Containers: Rebuild Container` or `Remote-Containers: Rebuild and Reopen in Container`, and then reinstall the *pdm4ar* module running `pip3 install -e [path/to/exercises_repo]` in the VS Code terminal.

Run the exercise with:
```shell
python3 [path/to/]src/pdm4ar/main.py --exercise 08
```
or:
```shell
python3 [path/to/]src/pdm4ar/main.py -e 08
```


## Hints
- one
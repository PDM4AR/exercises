# Exercise 15: Robust Satellite Docking

Plan and control a spaceship towards its docking target while avoiding planets,
moving asteroids, the map boundary, and the complete docking structure. The base
and both arms are physical obstacles. Collision checks must account for the
spaceship's entire footprint, including its orientation.

The three public scenarios include short external accelerations. Their effects
appear in the observed state; the agent is not given the sampled acceleration.
Private scenarios use different disturbance seeds and activation times. Design a closed-loop solution that
can track its trajectory and replan when needed. Any planning method is allowed.

## Student implementation

Implement the files in `src/pdm4ar/exercises/ex15/`:

- `agent.py`: initialize the agent and return commands at each observation.
- `planner.py`: construct feasible trajectories and implement your planning method.
- `spaceship.py`: implement the nominal dynamics and their derivatives.
- `discretization.py`: provided numerical integration and discretization helpers.

Keep the `SpaceshipAgent` interface and the existing method signatures.
The supplied planning and dynamics skeletons intentionally require implementation;
they do not contain the private reference controller.

Run the public exercise from the template repository:

```bash
python src/pdm4ar/main.py -e 15
```

The report and animation are written under `out/15/`. To choose scenarios or
create a local one, use `exercises_def/ex15/get_config.py` and its YAML files.

## Dynamics and disturbances

The state is `[x, y, psi, vx, vy, dpsi, delta, m]`. Position and heading are in
the world frame; linear velocities are in the spaceship body frame. Commands
are thrust and nozzle angle rate. The nominal model has the same dynamics as
the [satellite docking exercise](13-satellite_docking.md).

The simulator samples one linear acceleration vector in the world frame and
one angular acceleration from the configured seed. It applies them throughout
the configured activation window and rotates the linear acceleration into the
body frame at each integration step. The nominal planning model does not receive
this disturbance. The public bursts last 1.5 seconds and begin at 4, 6, and 8
seconds respectively. The three public configurations were verified with the
reference controller before inclusion in the exercise.

## Constraints and evaluation

Reach the goal with the configured position, orientation, and velocity tolerances.
Avoid all collisions and respect the map boundary, body velocity limits, nozzle
angle and rate limits, thrust limits, and dry mass. Starting trajectories use
zero initial input; when replanning, continue from the currently applied input.

The score uses `exercises_def/ex15/perf_metrics.py`: mission success and collisions
dominate, followed by computation time, terminal position error, distance
travelled, episode duration, and average absolute thrust. Scores are averaged
within each scenario and then across scenarios. Excessive rotations and detours
increase the time, distance, and actuation costs.

Reports include the animation and state/command plots. If the agent records
tracking and replanning events, these are plotted alongside the disturbance
window. Reporting does not provide disturbance parameters to the agent.

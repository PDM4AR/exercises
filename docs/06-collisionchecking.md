# Collision Checking and Sampling-based Planning :collision:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-helloworld.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Exercise Overview

In this exercise, you will build a collision detection system for a circular
differential-drive robot and then apply it to sampling-based motion planning.

### What You Will Implement

- **Geometric Collision Detection**: Implement the Separating Axis Theorem (SAT)
- **Discretization Methods**: Apply occupancy grids for collision checking in continuous spaces
- **Spatial Data Structures**: Use R-trees for efficient collision queries in environments with many obstacles  
- **Coordinate Frame Transformations**: Handle collision detection with sensor data in robot coordinate frames
- **Optimization-based Methods**: Implement Differentiable Collision Detection (DCDL) for continuous collision measures
- **Sampling-based Planning**: Apply the collision checker in PRM and RRT*

Unless otherwise specified, you are **NOT allowed** to use any geometry libraries like `shapely` for geometric operations and collision detection. :warning: <span style="color:red">We will check your implementation for this.</span> :warning:

## Part 1: Collision Checking Primitives with Separating Axis Theorem

Two convex shapes do not collide if there is an axis on which their
projections do not overlap. Tasks 1--3 build the projection and candidate-axis
operations used by the Separating Axis Theorem.

### Task 1: Project a Polygon onto a Segment

Implement `CollisionPrimitives_SeparateAxis.proj_polygon` in
`src/pdm4ar/exercises/ex06/collision_primitives.py`. The axis need not pass
through the origin. Return the segment bounded by the two extreme projected
points. The same function also accepts a `Circle` for Task 3.

### Task 2: Separating Axis Theorem for Two Polygons

Implement `overlap`, `get_axes`, and the polygon--polygon branch of
`separating_axis_thm`. Candidate axes may be chosen perpendicular to the
polygon edges. `overlap` and `get_axes` are supporting functions and are not
graded separately.

### Task 3: Separating Axis Theorem for a Polygon and a Circle

Implement `get_axes_cp` and the polygon--circle branch of
`separating_axis_thm`. In addition to the axes normal to the polygon edges,
include the axis through the circle centre and the closest polygon vertex.

## Part 2: Collision Check Module

In this part, you will implement a collision checking module for a circle-shaped differential drive robot navigating through obstacles.

**Context:** A circular robot moves along predefined paths in a 2D world with fixed obstacles (circles, triangles, polygons).

**Goal:** Implement different collision detection methods to check if robot paths are collision-free. Each method should use a unique approach to solve the collision-checking problem.

**Available Tools:** Use the primitives implemented in Part 1 and the provided
point/segment collision primitives in `CollisionPrimitives`. The collision
checker implemented here will also serve as the local planner in Part 3.

**Path Representation:** Robot path is represented using the following data structure (`src/pdm4ar/exercises_def/ex06/structures.py`):

```python
@dataclass(frozen=True)
class Path:
    waypoints: List[Point]
```

Unlike `Polygon` which connects first and last vertices, `Path` does not connect first and last waypoints.

You will implement functions in the `CollisionChecker` class (`src/pdm4ar/exercises/ex06/collision_checker.py`) using different collision detection strategies.

#### Task 4: Collision Checking Procedure for Circle-shaped Robot

In this task, you will implement a baseline collision checking method using the available collision check primitives. You are **not allowed** to use any geometry libraries like `shapely` in this task.

You will implement the `path_collision_check` function which returns the indices of `Segment`s from the given `Path` that collide with any of the given obstacles.

**Function signature:**
- **Input:** `Path` *t*, robot radius *r*, list of obstacles
- **Output:** List of indices representing the `Segment`s of the `Path` that collide with any obstacles

**Implementation approaches:** To account for the robot's radius, you can either:
1. **Inflate obstacles** by the robot's radius (creating a larger "danger zone") and convert robot-vs-obstacle collision checks into point-vs-inflated-obstacle checks
2. Or, **Inflate path segments** by the robot's radius and check the resulting swept volume against each obstacle

#### Task 5: Collision Checking via Occupancy Grid

The goal and all assumptions are the same as `Task 4`.

Implement collision checking using an occupancy grid approach. You may use `shapely` to create the occupancy grid, but you are not allowed to use it for checking collisions between the path segments and obstacles directly.

**Implementation approaches:**
1. Create an occupancy grid representing the environment
2. Mark obstacle cells as occupied
3. Check which path segments pass through occupied cells

You will implement the `path_collision_check_occupancy_grid` function which returns the `Segment` indices of colliding path segments.

**Note:** Due to the discrete nature of occupancy grids, perfect accuracy of 1.0 is not expected. You may tweak the grid resolution to balance accuracy and performance.

#### Task 6: Collision Checking using R-Trees

The goal and all assumptions are the same as `Task 4`.

In this task, you will use R-Trees to improve the query time performance of your collision checking module.

**Background:** R-Tree is an important spatial indexing data structure used in collision detection optimization. For environments with a large number of obstacles, it provides execution time improvements through its bounding box volume hierarchy structure, which allows for efficient spatial queries by quickly eliminating obstacles that cannot possibly collide with the query object.

**Implementation approach:**
1. Build an R-Tree containing all the given obstacles
2. For each path segment, query the R-Tree to get potential collision candidates
3. Perform detailed collision checking only on the candidate obstacles

You may use the functionalities of `shapely` here, including `STRtree` for R-Tree implementation. You are also free to implement your own R-Tree if you prefer.

You will implement the `path_collision_check_r_tree` function which returns the indices of `Segment`s from the given `Path` that collide with any obstacles.

#### Task 7: Collision Checking in Robot Frame

Raw sensor data are often provided in the sensor frame of the robot. 
In this task, you receive the current pose of the robot and the next pose of the robot in the world frame (planning is done with respect to the world frame), but the observed obstacles are given in the robot's sensor frame. 
At each step, the robot will observe obstacles in the 2D world.
The function needs to check if there is a collision during the robot's movement to its next pose. You may use the functionalities of `shapely` here.

<p align="center">
  <img alt="img-name" src="https://github-production-user-asset-6210df.s3.amazonaws.com/92320167/279223946-5dafecda-622e-4cae-8771-8a52ea5f807e.jpg">
  <br>
    <em>Sensor frame diagram</em>
</p>

In this task, you will implement the `collision_check_robot_frame` function which returns *True* if the robot will collide with any of the fixed obstacles during its movement to the next pose. 
This function takes the robot radius *r*, current pose `SE2Transform`, next pose `SE2Transform`, and a list of observed obstacles in the robot frame as arguments.

#### Task 8: Collision Checking with Optimization-based Collision Checking

The goal and all assumptions are the same as in Task 4. 

However, in this task, you are required to implement the [Differentiable Collision Detection (DCDL)](https://arxiv.org/abs/2207.00669) framework that formulates collision detection as a convex optimization problem. This method solves for the minimum uniform scaling applied to each primitive before they intersect, providing a uniform collision detection between a set of convex primitives.

> Compared to the previous methods, DCDL provides not only a binary collision check but also a derivative of a continuous measure of how close the primitives are to colliding, namely the scaling, with respect to the problem parameters, in this case the path segments' and obstacles' positions and orientations. This gradient information provides a direction of how to "pull" the path segments away from the obstacles, which can be beneficial for the downstream planning and control tasks. However, we won't use this information in this exercise. Interested readers can refer to the paper for more details.

In this task, you will implement the `path_collision_check_opt` function which returns the `Segment` indices of the given `Path` which collide with any of the given obstacles.

You can use the code structure of the `OptCollisionCheckingPrimitives` class in `src/pdm4ar/exercises/ex06/opt_collision_checking_primitives.py` and implement the corresponding methods to solve the optimization problem. Or you can implement your own `DCDL` framework from scratch.

We will only call the `path_collision_check_opt` function during the evaluation.

## Part 3: Sampling-based Planning Applications

Tasks 9 and 10 are implemented in
`src/pdm4ar/exercises/ex06/sampling_planners.py`. They reuse the collision
checker from Tasks 4--8. The robot is a disk with radius `robot_radius`, and
the configuration space is bounded by an `AABB`.

Direct calls to motion-planning libraries that already implement PRM or RRT*
are not allowed. General-purpose numerical operations and data structures may
be used.

### Task 9: Probabilistic Roadmap (PRM)

The evaluator supplies a deterministic list of configurations. The list is
generated using a fixed seed, so every student receives the same roadmap
input. Some configurations may collide with obstacles.

Implement `SamplingBasedPlanner.prm`:

1. Discard configurations that are outside the bounds or in collision.
2. Connect pairs at most `connection_radius` apart only when the complete edge
   is collision-free.
3. Weight each edge by its Euclidean length.
4. Return a shortest path for every supplied start--goal query using Dijkstra
   or A*.

```python
prm(
    samples: list[Point],
    queries: list[tuple[Point, Point]],
    bounds: AABB,
    robot_radius: float,
    obstacles: list[GeoPrimitive],
    connection_radius: float,
) -> list[Path]
```

Every waypoint in a returned path must come from `samples`. Return `Path([])`
for a query whose endpoint is invalid or whose roadmap contains no solution.

### Task 10: Rapidly-exploring Random Tree Star (RRT*)

Implement `SamplingBasedPlanner.rrt_star`. You may choose how to sample the
configuration space. A typical implementation:

1. Samples a configuration, optionally with a goal bias.
2. Steers from the nearest tree node by at most `step_size`.
3. Rejects colliding configurations and edges.
4. Chooses the lowest-cost parent among nearby nodes.
5. Rewires nearby nodes when the new node lowers their cost.
6. Reconstructs the best path to the goal through parent links.

```python
rrt_star(
    start: Point,
    goal: Point,
    bounds: AABB,
    robot_radius: float,
    obstacles: list[GeoPrimitive],
    max_iterations: int = 3000,
    step_size: float = 0.5,
    rewire_radius: float = 1.5,
    goal_bias: float = 0.1,
    seed: int = 0,
) -> Path
```

The same inputs and seed must produce the same result. Return `Path([])` when
the endpoints are invalid or no solution is found within the iteration
budget.

### Evaluation

For this exercise our performance metric is accuracy and execution time.

**Test Data Generation:**
- Tasks 1–8 use data from `src/pdm4ar/exercises_def/ex06/data.py`
- Tasks 9–10 use deterministic planning problems from
  `src/pdm4ar/exercises_def/ex06/sampling_data.py`
- Each task contains multiple test cases

**Accuracy Calculation:**
- **Tasks 1–3:** compare geometric results with the expected values
- **Tasks 4-8:** Lists of indices are converted into a boolean list which represents whether there is a collision on each line segment of the path
- **Tasks 4-8:** Accuracies are calculated by the average of the accuracy of test cases
- **Task 9:** checks query endpoints, sample membership, bounds,
  collision-free edges, and shortest-path cost
- **Task 10:** checks endpoints, bounds, collision-free edges, reproducibility,
  and path-cost suboptimality

**Execution Time:**
- Execution time of each task is calculated as an average of its test cases
- Below is the table summarizing the execution times of reference implementations on the evaluation dataset.

| Task **ID** | **Average Solving Time** |
|-------------|---------------------------|
| 04          | 0.0028s                      |
| 05          | 1.0879s                      |
| 06          | 0.0002s                      |
| 07          | 0.0030s                      |
| 08          | 0.1639s                      |
| 09          | hardware dependent        |
| 10          | hardware dependent        |


**Final Scoring:**
- Accuracies and execution times of each task are aggregated as a weighted average

| Task **ID** | **Number of Test Cases** | *Accuracy Weight* | *Solving Time Weight* |
|-------------|--------------------------|-------------------|-----------------------|
| 01          | 05                       | 05                | 05                    |
| 02          | 10                       | 20                | 20                    |
| 03          | 06                       | 20                | 20                    |
| 04          | 05                       | 20                | 20                    |
| 05          | 05                       | 20                | 20                    |
| 06          | 05                       | 30                | 30                    |
| 07          | 05                       | 20                | 20                    |
| 08          | 05                       | 30                | 30                    |
| 09          | 05                       | 20                | 20                    |
| 10          | 10                       | 20                | 20                    |

### Advice

Be cautious of clashing class names between our self-defined `GeoPrimitive` classes and the `shapely` classes. It is not recommended to run the following: `import triangle` or `from shapely import *` as these will result in errors due to identical class/module names. You may instead choose to use aliases for your imported modules (e.g. `import triangle as tr` or `from shapely.geometry import Point as shapelyPoint`) or to just import the methods that you need (e.g. `from triangle import triangulate`).

There are also times when you may be dealing with calculations involving lots of floating point numbers and you may wish to compare the result against a certain value. The `math.isclose` method might be helpful as a direct `==` comparison will likely return *False* more often than not.

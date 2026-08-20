# Collision Checking :collision:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-helloworld.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Exercise Overview

In this exercise, you will build a comprehensive collision detection system for a circular differential drive robot navigating through environments with various obstacles. This exercise progresses from baseline collision checking to advanced spatial data structures, optimization-based methods, and sampling-based planning.

### What You Will Implement

- **Geometric Collision Detection**: Implement collision checking for a circular robot moving along a path
- **Discretization Methods**: Apply occupancy grids for collision checking in continuous spaces
- **Spatial Data Structures**: Use R-trees for efficient collision queries in environments with many obstacles  
- **Coordinate Frame Transformations**: Handle collision detection with sensor data in robot coordinate frames
- **Optimization-based Methods**: Implement Differentiable Collision Detection (DCDL) for continuous collision measures

Unless otherwise specified, you are **NOT allowed** to use any geometry libraries like `shapely` for geometric operations and collision detection. :warning: <span style="color:red">We will check your implementation for this.</span> :warning:

## Part 1: Collision Check Module

In this part, you will implement a collision checking module for a circle-shaped differential drive robot navigating through obstacles.

**Context:** A circular robot moves along predefined paths in a 2D world with fixed obstacles (circles, triangles, polygons).

**Goal:** Implement different collision detection methods to check if robot paths are collision-free. Each method should use a unique approach to solve the collision-checking problem.

**Available Tools:** Collision check primitives between [circle, polygon, triangle] and [point, segment] are provided in the `CollisionPrimitives` class (`src/pdm4ar/exercises/ex06/collision_primitives.py`).

**Path Representation:** Robot path is represented using the following data structure (`src/pdm4ar/exercises_def/ex06/structures.py`):

```python
@dataclass(frozen=True)
class Path:
    waypoints: List[Point]
```

Unlike `Polygon` which connects first and last vertices, `Path` does not connect first and last waypoints.

You will implement functions in the `CollisionChecker` class (`src/pdm4ar/exercises/ex06/collision_checker.py`) using different collision detection strategies.

#### Task 1: Collision Checking Procedure for Circle-shaped Robot

In this task, you will implement a baseline collision checking method using the available collision check primitives. You are **not allowed** to use any geometry libraries like `shapely` in this task.

You will implement the `path_collision_check` function which returns the indices of `Segment`s from the given `Path` that collide with any of the given obstacles.

**Function signature:**
- **Input:** `Path` *t*, robot radius *r*, list of obstacles
- **Output:** List of indices representing the `Segment`s of the `Path` that collide with any obstacles

**Implementation approach:** To account for the robot's radius, inflate obstacles by the robot's radius (creating a larger "danger zone") and convert robot-vs-obstacle collision checks into point-vs-inflated-obstacle checks using the provided primitives.

#### Task 2: Collision Checking via Occupancy Grid

The goal and all assumptions are the same as `Task 1`.

Implement collision checking using an occupancy grid approach. You may use `shapely` to create the occupancy grid, but you are not allowed to use it for checking collisions between the path segments and obstacles directly.

**Implementation approaches:**
1. Create an occupancy grid representing the environment
2. Mark obstacle cells as occupied
3. Check which path segments pass through occupied cells

You will implement the `path_collision_check_occupancy_grid` function which returns the `Segment` indices of colliding path segments.

**Note:** Due to the discrete nature of occupancy grids, perfect accuracy of 1.0 is not expected. You may tweak the grid resolution to balance accuracy and performance.

#### Task 3: Collision Checking using R-Trees

The goal and all assumptions are the same as `Task 1`.

In this task, you will use R-Trees to improve the query time performance of your collision checking module.

**Background:** R-Tree is an important spatial indexing data structure used in collision detection optimization. For environments with a large number of obstacles, it provides execution time improvements through its bounding box volume hierarchy structure, which allows for efficient spatial queries by quickly eliminating obstacles that cannot possibly collide with the query object.

**Implementation approach:**
1. Build an R-Tree containing all the given obstacles
2. For each path segment, query the R-Tree to get potential collision candidates
3. Perform detailed collision checking only on the candidate obstacles

You may use the functionalities of `shapely` here, including `STRtree` for R-Tree implementation. You are also free to implement your own R-Tree if you prefer.

You will implement the `path_collision_check_r_tree` function which returns the indices of `Segment`s from the given `Path` that collide with any obstacles.

#### Task 4: Collision Checking in Robot Frame

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

#### Task 5: Collision Checking with Optimization-based Collision Checking

The goal and all assumptions are the same as in Task 1.

However, in this task, you are required to implement the [Differentiable Collision Detection (DCDL)](https://arxiv.org/abs/2207.00669) framework that formulates collision detection as a convex optimization problem. This method solves for the minimum uniform scaling applied to each primitive before they intersect, providing a uniform collision detection between a set of convex primitives.

> Compared to the previous methods, DCDL provides not only a binary collision check but also a derivative of a continuous measure of how close the primitives are to colliding, namely the scaling, with respect to the problem parameters, in this case the path segments' and obstacles' positions and orientations. This gradient information provides a direction of how to "pull" the path segments away from the obstacles, which can be beneficial for the downstream planning and control tasks. However, we won't use this information in this exercise. Interested readers can refer to the paper for more details.

In this task, you will implement the `path_collision_check_opt` function which returns the `Segment` indices of the given `Path` which collide with any of the given obstacles.

You can use the code structure of the `OptCollisionCheckingPrimitives` class in `src/pdm4ar/exercises/ex06/opt_collision_checking_primitives.py` and implement the corresponding methods to solve the optimization problem. Or you can implement your own `DCDL` framework from scratch.

We will only call the `path_collision_check_opt` function during the evaluation.

## Part 2: Sampling-based Planning Applications

Tasks 6 and 7 are implemented in `src/pdm4ar/exercises/ex06/sampling_planners.py`
and reuse the collision checker from Tasks 1-5 for both configurations and
complete edges. Planning problems may combine `Circle`, `Polygon`, and
`Triangle` obstacles. You must implement the planning logic yourself: direct
calls to libraries that already provide complete PRM or RRT* implementations
are not allowed.

### Task 6: Probabilistic Roadmap (PRM)

Implement `SamplingBasedPlanner.prm`. The evaluator supplies a deterministic
list of configurations generated with a fixed seed, together with one or more
start-goal queries. Some supplied configurations may be in collision and must
be discarded. Build the roadmap by connecting valid pairs within
`connection_radius`, but only when the complete edge is collision-free. Then
return a shortest path for every query. Every returned waypoint must come from
the supplied samples: do not generate or interpolate additional points. Return
`Path([])` when a query is invalid or no connection can be found.

#### Suggested PRM procedure

1. Remove samples that are outside `bounds` or in collision.
2. Create an undirected graph whose vertices are the remaining samples.
3. Connect two samples when their distance is at most `connection_radius` and
   `Path([a, b])` is collision-free. Use their Euclidean distance as edge cost.
4. For each query, check that start and goal are valid supplied samples, then
   use Dijkstra or A* to find the shortest path.
5. Return the original sample points in order, or `Path([])` if no route exists.

Build the roadmap only once and reuse it for every query. Always check the full
edge, not only its endpoints.

### Task 7: Rapidly-exploring Random Tree Star (RRT*)

Implement `SamplingBasedPlanner.rrt_star`. You may choose the sampling strategy,
but the same inputs and seed must always produce the same result. Grow a tree
from the start while rejecting configurations and edges in collision. For each
new node, select the valid nearby parent with the lowest total cost and rewire
nearby nodes whenever the new connection reduces their cost. These two steps
are required: a basic RRT implementation is not sufficient. Return `Path([])`
when the endpoints are invalid or no solution is found within the iteration
budget.

#### Suggested RRT* procedure

Keep each node's point, parent, and total cost from the start. Use a local random
generator initialized with `seed`.

For every iteration:

1. Sample the goal with probability `goal_bias`; otherwise sample inside
   `bounds`.
2. Find the nearest tree node and move toward the sample by at most `step_size`.
   Reject the new point if its configuration or connecting edge collides.
3. Among nodes within `rewire_radius`, choose the collision-free parent that
   gives the lowest total cost.
4. Rewire nearby nodes through the new node when this lowers their cost, and
   update the costs of their descendants.
5. Record collision-free connections to the goal and keep the cheapest one.

Finally, follow parent links from the goal back to the start and reverse the
result. Return `Path([])` if the endpoints are invalid or the goal was not
reached.

### Evaluation

For this exercise our performance metric is accuracy and execution time.

**Test Data Generation:**
- For each task, random inputs are generated with the algorithm provided in `src/pdm4ar/exercises_def/ex06/data.py`
- Tasks 6-7 use deterministic planning problems from `src/pdm4ar/exercises_def/ex06/sampling_data.py`
- Each task contains multiple test cases

**Accuracy Calculation:**
- **Tasks 1-5:** Lists of indices are converted into a boolean list which represents whether there is a collision on each line segment of the path; accuracies are averaged across test cases
- **Task 6:** Paths are checked for endpoints, sample membership, bounds, collision-free edges, and shortest-path cost
- **Task 7:** Paths are checked for endpoints, bounds, collision-free edges, reproducibility, and cost suboptimality

**Execution Time:**
- Execution time of each task is calculated as an average of its test cases
- Below is the table summarizing the execution times of reference implementations on the evaluation dataset.

| Task **ID** | **Average Solving Time** |
|-------------|---------------------------|
| 01          | 0.0028s                      |
| 02          | 1.0879s                      |
| 03          | 0.0002s                      |
| 04          | 0.0030s                      |
| 05          | 0.1639s                      |
| 06          | TODO                          |
| 07          | TODO                          |


**Final Scoring:**
- Accuracies and execution times of each task are aggregated as a weighted average

| Task **ID** | **Number of Test Cases** | *Accuracy Weight* | *Solving Time Weight* |
|-------------|--------------------------|-------------------|-----------------------|
| 01          | 05                       | 20                | 20                    |
| 02          | 05                       | 20                | 20                    |
| 03          | 05                       | 30                | 30                    |
| 04          | 05                       | 20                | 20                    |
| 05          | 05                       | 30                | 30                    |
| 06          | 05                       | 20                | 20                    |
| 07          | 10                       | 20                | 20                    |

### Advice

Be cautious of clashing class names between our self-defined `GeoPrimitive` classes and the `shapely` classes. It is not recommended to run the following: `import triangle` or `from shapely import *` as these will result in errors due to identical class/module names. You may instead choose to use aliases for your imported modules (e.g. `import triangle as tr` or `from shapely.geometry import Point as shapelyPoint`) or to just import the methods that you need (e.g. `from triangle import triangulate`).

There are also times when you may be dealing with calculations involving lots of floating point numbers and you may wish to compare the result against a certain value. The `math.isclose` method might be helpful as a direct `==` comparison will likely return *False* more often than not.

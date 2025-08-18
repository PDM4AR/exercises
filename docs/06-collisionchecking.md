# Collision Checking :collision:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Exercise Overview

In this exercise, you will build a comprehensive collision detection system for a circular differential drive robot navigating through environments with various obstacles. This exercise progresses from fundamental geometric algorithms to advanced spatial data structures and optimization-based methods. 

### What You Will Implement

- **Geometric Collision Detection**: Implement the Separating Axis Theorem (SAT) for polygon-polygon and polygon-circle collision detection
- **Discretization Methods**: Apply occupancy grids for collision checking in continuous spaces
- **Spatial Data Structures**: Use R-trees for efficient collision queries in environments with many obstacles  
- **Coordinate Frame Transformations**: Handle collision detection with sensor data in robot coordinate frames
- **Optimization-based Methods**: Implement Differentiable Collision Detection (DCDL) for continuous collision measures

Unless otherwise specified, you are **NOT allowed** to use any geometry libraries like `shapely` for geometric operations and collision detection. :warning: <span style="color:red">We will check your implementation for this.</span> :warning:

## Part 1: Collision Checking Primitives with Separating Axis Theorem

### Recap: Separating Axis Theorem (SAT)

**Key Concept:** Two convex shapes do not collide if there exists a separating axis where their projections do not overlap.

**How it works:**
1. **Find candidate axes** - typically perpendicular to edges of the shapes
2. **Project both shapes** onto each axis
3. **Check for overlap** - if any axis shows no overlap, shapes don't collide
4. **Collision occurs** only if projections overlap on ALL axes

#### Task 1: Project a Polygon onto a Segment

First, implement the `proj_polygon` function inside the `CollisionPrimitives_SeparateAxis` class in the `src/pdm4ar/exercises/ex06/collision_primitives.py` file.

It projects a polygon onto an axis and returns the projection segment.

You can use the `numpy` library for linear algebra operations, but you should not use any geometry libraries like `shapely` in this task. 

**Notes:**
- We use a line segment (bounded by two points) to represent a straight line/axis (extending infinitely in both directions) containing the segment
- The axis does not necessarily pass through the origin
- The projection is a segment bounded by two endpoints on the axis
- Projection accuracy is verified by segment length and endpoint precision
- Function signature also accepts `Circle` input for later use in `Task 3`

#### Task 2a: Determine if Two Segments Overlap or Not

Implement the `overlap` function inside the `CollisionPrimitives_SeparateAxis` class in the `src/pdm4ar/exercises/ex06/collision_primitives.py` file.

It checks if two line segments overlap (intersect), i.e., if they share any points.

**Note:** This function will be used in later tasks for SAT implementation, but is not directly tested by the checker. We encourage you to test it yourself.

#### Task 2b: Return a List of Candidate Separating Axes
Implement a function that gets candidate separating axes given two polygons.

You will implement the `get_axes` function inside the `CollisionPrimitives_SeparateAxis` class in the `src/pdm4ar/exercises/ex06/collision_primitives.py` file.

If two polygons do not intersect, there are potentially infinite separating axes that can be computed. As a hint, we recommend returning axes that are orthogonal to the edges of each polygon only. 

**Note:** The checker will not verify your implementation for this task, so we encourage that you do your own testing. 

#### Task 2c: Separating Axis Theorem for Two Polygons

In this task, we bring it all together and implement the SAT for two polygons.

We will be modifying the **first** case in the `separating_axis_thm` function.

Using the methods you have previously implemented: `get_axes`, `proj_polygon`, and `overlap`, determine if two polygons intersect with each other or not using the SAT.

The `separating_axis_thm` function takes two polygons as inputs: *p1* and *p2* and returns a tuple with a mandatory argument and an optional argument. 

The first argument is a `bool` that is *True* if the polygons collide, and *False* if they do not. 

The second argument is an optional `Segment` which you can use to visualize which segment you are projecting against in your implementation of the SAT. 

#### Task 3a: Return a List of Candidate Separating Axes for a Polygon and a Circle
We now move to computing separating axes for a polygon and a circle.

You will implement the function `get_axes_cp` that takes a `Circle` *circ* and a `Polygon` *poly* as inputs and returns a list of segments representing the candidate separating axes.

**Hint**: Notice that the circle is a polygon with an infinite number of edges. Fortunately we do not need to check all axes normal to the edges.
It's sufficient to check the axes normal to the polygon edges plus ONE axis formed by the circle center and the closest vertex of the polygon.

**Note**: The checker will not verify your implementation for this task, so we encourage that you do your own testing.

#### Task 3b: Separating Axis Theorem for a Polygon and a Circle

We will be modifying the **second** case in the `separating_axis_thm` function. 

The `separating_axis_thm` function takes a polygon and a circle as inputs: *p1* and *p2* and returns a tuple with a mandatory argument and an optional argument. 

The first argument is a `bool` that is *True* if the shapes collide, and *False* if they do not. 

The second argument is an optional `Segment` which you can use to visualize which segment you are projecting against in your implementation of the Separating Axis Theorem. 

## Part 2: Collision Check Module

In this part, you will implement a collision checking module for a circle-shaped differential drive robot navigating through obstacles.

**Context:** A circular robot moves along predefined paths in a 2D world with fixed obstacles (circles, triangles, polygons).

**Goal:** Implement different collision detection methods to check if robot paths are collision-free. Each method should use a unique approach to solve the collision-checking problem.

**Available Tools:** 
- All collision check primitives implemented in `Part 1`
- Collision check primitives between [circle, polygon, triangle] and [point, segment] provided in the `CollisionPrimitives` class (`src/pdm4ar/exercises/ex06/collision_primitives.py`).

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
2. Or, **Inflate path segments** by the robot's radius and reuse the polygon-polygon and polygon-circle collision check primitives implemented in Part 1

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

### Evaluation

For this exercise our performance metric is accuracy and execution time.

**Test Data Generation:**
- For each task, random inputs are generated with the algorithm provided in `src/pdm4ar/exercises_def/ex06/data.py`
- Each task contains multiple test cases

**Accuracy Calculation:**
- **Tasks 1-3:** Accuracies are calculated by the ratio of correct answers
- **Tasks 4-8:** Lists of indices are converted into a boolean list which represents whether there is a collision on each line segment of the path
- **Tasks 4-8:** Accuracies are calculated by the average of the accuracy of test cases

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


**Final Scoring:**
- Accuracies and execution times of each task are aggregated as a weighted average

| Task **ID** | **Number of Test Cases** | *Accuracy Weight* | *Solving Time Weight* |
|-------------|--------------------------|-------------------|-----------------------|
| 01          | 05                       | 05                | 0                     |
| 2a          | 00                       | 00                | 0                     |
| 2b          | 00                       | 00                | 0                     |
| 2c          | 10                       | 20                | 0                     |
| 3a          | 00                       | 00                | 0                     |
| 3b          | 06                       | 20                | 0                     |
| 04          | 05                       | 20                | 20                    |
| 05          | 05                       | 20                | 20                    |
| 06          | 05                       | 30                | 30                    |
| 07          | 05                       | 20                | 20                    |
| 08          | 05                       | 30                | 30                    |

### How to run

Make sure to update your repo before running the exercise. 
Please refer to [Hello World](01-helloworld.md) for instructions.

### Advice

Be cautious of clashing class names between our self-defined `GeoPrimitive` classes and the `shapely` classes. It is not recommended to run the following: `import triangle` or `from shapely import *` as these will result in errors due to identical class/module names. You may instead choose to use aliases for your imported modules (e.g. `import triangle as tr` or `from shapely.geometry import Point as shapelyPoint`) or to just import the methods that you need (e.g. `from triangle import triangulate`).

There are also times when you may be dealing with calculations involving lots of floating point numbers and you may wish to compare the result against a certain value. The `math.isclose` method might be helpful as a direct `==` comparison will likely return *False* more often than not.

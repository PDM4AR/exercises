# Collision Checking :collision:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Exercise Overview

In this exercise, you will implement a collision checking module for a circle-shaped differential drive robot. The module will:

- **Check collisions** between geometric primitives (circles, polygons, segments)
- **Evaluate path safety** by determining if robot paths are collision-free
- **Use multiple approaches** including SAT, occupancy grids, R-trees, and safety certificates

Unless otherwise specified, you are **NOT allowed** to use any geometry libraries like `shapely` for geometric operations and collision detection. :warning: We will check your implementation for this. :warning:

## Part 1: Collision Check Primitives

### Recap: Separating Axis Theorem (SAT)

**Key Concept:** Two convex shapes do not collide if there exists a separating axis where their projections do not overlap.

**How it works:**
1. **Find candidate axes** - typically perpendicular to edges of the shapes
2. **Project both shapes** onto each axis
3. **Check for overlap** - if any axis shows no overlap, shapes don't collide
4. **Collision occurs** only if projections overlap on ALL axes

#### Step 1: Project A Polygon onto a Segment

First, implement the `proj_polygon` function inside the `CollisionPrimitives_SeparateAxis` class in `src/pdm4ar/exercises/ex06/collision_primitives.py` file.

It projects a polygon onto an axis and returns the projection segment.

You can use the `numpy` library for linear algebra operations, but you should not use any geometry libraries like `shapely` in this step. 

**Notes:**
- We use a line segment (bounded by two points) to represent an straight-line/axis (extending infinitely in both directions) containing the segment
- The axis does not necessarily pass through the origin
- The projection is a segment bounded by two endpoints on the axis
- Projection accuracy is verified by segment length and endpoint precision
- Function signature also accepts `Circle` input for later use in Step 3

#### Step 2a: Determine if Two Segments Overlap or Not

Implement the `overlap` function inside the `CollisionPrimitives_SeparateAxis` class in `src/pdm4ar/exercises/ex06/collision_primitives.py` file.

It checks if two line segments overlap (intersect).

**Note:** This function will be used in later steps for SAT implementation, but is not directly tested by the checker

#### Step 2b: Return a List of Candidate Separating Axes
Implement a function that gets candidate separating axes given two polygons.

You will implement the `get_axes` function inside the `CollisionPrimitives_SeparateAxis` class in `src/pdm4ar/exercises/ex06/collision_primitives.py` file.

If two polygons do not intersect, there are potentially infinite separating axes that can be computed. As a hint, we recommend returning axes that are orthogonal to the edges of each polygon only. 

**Note:** The checker will not verify your implementation for this step, so we encourage that you do your own testing. 

#### Step 2c: Separating Axis Theorem for Two Polygons

In this step, we bring it all together and implement the Separating Axis Theorem for two polygons.

We will be modifying the **first** case in the `separating_axis_thm` function.

Using the methods you have previously implemented: `get_axes`, `proj_polygon`, and `overlap`, determine using the Separating Axis Theorem if two polygons intersect with each other or not. 

The `separating_axis_thm` function takes in two polygons as inputs: *p1* and *p2* and returns a tuple with a mandatory argument and an optional argument. 

The first argument is a `bool` that is *True* if the polygons collide, and *False* if they do not. 

The second argument is an optional `Segment` which you can use to visualize which segment you are projecting against in your implementation of the Separating Axis Theorem. 

#### Step 3a: Return a List of Candidate Separating Axes for a Polygon and a Circle
We now move to computing separating axes for a polygon and a circle. 


You will implement the function `get_axes_cp` that takes as inputs a `Circle` *circ* and a `Polygon` *poly* and returns a list of segments, which will represent the axes.  

**Hint**: Notice that the circle is a polygon with an infinite number of edges. Fortunately we do not need to check all axes normal to the edges.
It's sufficient to check the axes normal to the polygon edges plus ONE axis formed by the circle center and the closest vertex of the polygon.


**Note**: The checker will not verify your implementation for this step, so we encourage that you do your own testing.


#### Step 3b: Separating Axis Theorem for a Polygon and a Circle

We will be modifying the **second** case in the `separating_axis_thm` function. 

The `separating_axis_thm` function takes in a polygon and a circle as inputs: *p1* and *p2* and returns a tuple with a mandatory argument and an optional argument. 

The first argument is a `bool` that is *True* if the shapes collide, and *False* if they do not. 

The second argument is an optional `Segment` which you can use to visualize which segment you are projecting against in your implementation of the Separating Axis Theorem. 


## Part 2: Collision Check Module

In the second part of this exercise, we will explore an alternative method for detecting collisions using shape intersections and triangulation. Although triangulation is less commonly employed than the Separating Axis Theorem (SAT), it offers an intuitive approach for decomposing large polygons into manageable triangular shapes.

For this exercise, the `CollisionPrimitives` class located in `src/pdm4ar/exercises/ex06/collision_primitives.py` is provided to you. This class includes the following functions:

- `circle_point_collision`
- `triangle_point_collision`
- `polygon_point_collision`
- `circle_segment_collision`
- `sample_segment`
- `triangle_segment_collision`
- `polygon_segment_collision`
- `polygon_segment_collision_aabb`
- `_poly_to_aabb`

We recommend that you thoroughly review these functions, as they will be crucial for the subsequent steps of the exercise.

The context for this part of the exercise assumes a circle-shaped differential drive robot navigating a 2D world populated with fixed obstacles arranged along a predefined path. These obstacles can be circular, triangular, or polygonal in shape.

You will implement various methods to check for collisions along the possible path of our robot in the following steps. It is important to note that each method you implement should adopt a unique approach to solving the collision-checking problem. As such, the code for each collision-checking function should be distinct from one another.

By employing different strategies, you will gain a comprehensive understanding of the strengths and limitations of various collision detection methods, ultimately enhancing the robustness of the collision-checking module for path planning.
To represent the path of the robot, the following data structure (`src/pdm4ar/exercises_def/ex06/structures.py`) will be used:

```python
@dataclass(frozen=True)
class Path:
    waypoints: List[Point]
```

Please note that the definitions of `Path` and `Polygons` are similar. 
The `Polygon` class connects the first and last vertices by default. 
However, there isn't any connection between the first and last waypoints in `Path` objects. 
For the remaining part of the exercise, you are free to use or modify the `check_collision` function. You may also not use it.
This function takes two `GeoPrimitives` and checks the collision between them by using the primitives implemented in the first part of this exercise.

The task is to implement the functions in the `CollisionChecker` class in `src/pdm4ar/exercises/ex06/collision_checker.py`.

#### Step 4: Collision Checking Procedure for Circle-shaped Robot

In this step, you will implement a baseline version for collision checking by using the primitives implemented before. You should not use `shapely` here. (We will check!)
The aim of this part is to check if a candidate path for our circular robot is collision-free. 

You will implement the `path_collision_check` function which returns the `Segment` indices of the given `Path` which are in collision with the given obstacles. 
This function takes a `Path` *t*, the radius of the robot's occupancy *r*, and a list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which are in collision with any of the obstacles.

#### Step 5: Collision Checking via Occupancy Grid

The aim and all of the assumptions are the same as Step 4. 
However, in this step, you will use a different approach for collision checking. 
You are asked to implement collision checking via occupancy grids. 
You will initially create an occupancy grid of the given environment. 
Then using the occupancy grid, you will find the segments of the path in which our robot will collide. You may use the functionalities of `shapely` here. Note that you will have to convert the `GeoPrimitives` to `shapely` geometries in order to work with `shapely`.

In this step, you will implement the `path_collision_check_occupancy_grid` function which returns the `Segment` indices of the given `Path` which collide with any of the given obstacles. 
This function takes a `Path` *t*, the radius of the robot *r*, and a list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which collide with any of the obstacles. Note that due to the discrete nature of an occupancy grid, it is completely reasonable that the method might not result in perfect accuracy of 1.0.

#### Step 6: Collision Checking using R-Trees

The aim and all of the assumptions are the same as Step 4. 
Like previous steps, the aim is to find the segments of the path in which our circular robot will collide. 
However, in this step you will use R-Tree to increase the execution time performance of your collision check module. 
R-Tree is an important optimization approach that is used in collision checking. 
For environments with a high number of obstacles, it provides us an execution time decrease via its bounding box volume hierarchy structure. 
In this method, you will build an R-Tree of the given obstacles. You may use the functionalities of `shapely` here, including `STRTree`.
You are also free to implement your own R-Tree if you wish.

In this step, you will implement the `path_collision_check_r_tree` function which returns the `Segment` indices of the given `Path` which collide with any of the given obstacles. 
This function takes a `Path` *t*, the radius of the robot *r*, and a list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which collide with any of the obstacles.

#### Step 7: Collision Checking in Robot Frame

Raw sensor data are often given in the sensor frame of the robot. 
In this step, you receive the current pose of the robot and the next pose of the robot in the global frame (planning is done with respect to the global frame), but the observed obstacles are given in the robot's frame. 
At each step, the robot will observe the obstacles in our 2D world.
The function needs to check if there is a collision during the movement of the robot until its next pose. You may use the functionalities of `shapely` here.

<p align="center">
  <img alt="img-name" src="https://github-production-user-asset-6210df.s3.amazonaws.com/92320167/279223946-5dafecda-622e-4cae-8771-8a52ea5f807e.jpg">
  <br>
    <em>Sensor frame diagram</em>
</p>

In this step, you will implement the `collision_check_robot_frame` function which returns *True* if the robot will collide with any of the fixed obstacles during its movement until its next pose. 
This function takes the radius of the robot *r*, current pose `SE2transform`, next pose `SE2transform`, and list of observed obstacles in robot frame as arguments. 

#### Step 8: Collision Checking via Safety Certificates

The aim and all the assumptions are the same as in Step 4. 
Like the previous steps, the aim is to find the segments of the path in which our circular robot will collide. 
However, in this step you will use a different optimization method called Safety Certificates. 
For environments with a small number of obstacles but high number of points to be checked, it provides us an execution time decrease via the approach it uses for collision checking. 
To obtain detailed information about the algorithm, you can check the part that is related to the safety certificates from the [given paper](https://journals.sagepub.com/doi/full/10.1177/0278364915625345) (`Algorithm 1`). You may use the functionalities of `shapely` here (the distance between a point and an obstacle can be easily calculated in `shapely` via the `distance` function).

In this step, you will implement the `path_collision_check_safety_certificate` function which returns the `Segment` indices of the given `Path` which collide with any of the given obstacles. 
This function takes a `Path` *t*, the radius of the robot *r*, and a list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which collide with any of the obstacles.

### Evaluation

For this exercise our performance metric is accuracy and execution time.

**Test Data Generation:**
- For each step, random inputs are generated with the algorithm provided in `src/pdm4ar/exercises_def/ex06/data.py`
- Each step contains multiple test cases

**Accuracy Calculation:**
- **Steps 1-3:** Accuracies are calculated by the ratio of correct answers
- **Steps 4-8:** Lists of indices are converted into a boolean list which represents whether there is a collision on each line segment of the path
- **Steps 4-8:** Accuracies are calculated by the average of the accuracy of test cases

**Execution Time:**
- Execution time of each step is calculated as an average of its test cases

**Final Scoring:**
- Accuracies and execution times of each step are aggregated as a weighted average

| Step **ID** | **Number of Test Cases** | *Accuracy Weight* | *Solving Time Weight* |
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

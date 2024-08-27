# Collision Checking :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Exercise

Collision checking is a crucial capability of any planning system. 
In this exercise, you will implement a basic collision checking module for a circle-shaped differential drive robot. 
The goal is to obtain a module that can perform collision checks between basic geometric primitives. 
This module can then be used for a given robot to check whether its candidate (sub-)paths are collision-free.

### Collision check primitives

We start off by implementing some collision check primitives for basic geometric shapes.  To do so, we will use the Separating Axis Theorem for 2d Primitives. The Separating Axis Theorem allows for collision checking between any convex n-polygon. It is also widely used as a tool in path planning and navigation for robotics, as well as game programming. 

As a reminder, the Separating Axis Theorem states: If two sets are closed and at least one of them is compact, then there is a hyperplane between them, and even two parallel hyperplanes separated by a gap. An axis that is orthogonal to a separating hyperplane is deemed a Separating Axis, because the orthogonal projections of the convex bodies onto the axis are disjoint.

For the 2D Case, the hyperplanes are no more than line segments. 

These will come in handy later on. 
The first step is to implement the functions inside the `CollisionPrimitives_SeparateAxis` class in `src/pdm4ar/exercises/ex06/collision_primitives.py` file.

In this section, we suggest the use of linear algebra modules like `numpy`. 
However, you are not allowed to use modules that implement collision checking directly such as `shapely`. We will be checking solutions for correct implementation without usage of `shapely`.

#### Step 1: Project A Polygon onto an Segment

Implement a function that Projects a Polygon onto a Segment (the segment will later represent the Axis when implementing the theorem). Accuracy of the projection is checked by length of the section of the segment onto which the polygon is projected on, as well as having the endpoints of the projected segment be within some epsilon. 

To represent a Polygon and Segment, the following data structures (`src/pdm4ar/exercises_def/ex06/structures.py`) will be used:

```python
@dataclass(frozen=True)
class Segment(GeoPrimitive):
    p1: Point
    p2: Point

@dataclass(frozen=True)
class Polygon(GeoPrimitive):
    vertices: list[Point]

```

In this part, you will implement the   `proj_polygon` function of the `CollisionPrimitives_SeparateAxis` class. 

As arguments, the function takes in a Polygon and a Segment, and returns a Segment type. You are to project the Polygon onto the Segment, and return a shorter Segment that represents the resulting projection. 

Note: For this step, you will only be checked on projecting a N-sided polygon. However note that the function also accepts a Circle as an input argument. You may need to modify your implementation of the `proj_polygon` function to also accept circles, when you get to Task 3. 


#### Step 2a: Determine if Two Segments overlap or not
In this step, you will implement a function that determines wether two Segments overlap or not. 

The segment datatype on (`src/pdm4ar/exercises_def/ex06/structures.py`) will be used:

```python
@dataclass(frozen=True)
class Segment(GeoPrimitive):
    p1: Point
    p2: Point
```

You will implement the `overlap` function, taht takes in two `Segment` *s1* and *s2* as arguments and return *True* if they overlap (intersect) or *False* if they do not. 

The checker will not verify your implementation for this step, so we encourage that you do your own testing. 


#### Step 2b: Return a list of Candidate Separating Axes
In this step, you will implement a function that gets candidate separating axes if given two Polygons. 

The polygon datatype on (`src/pdm4ar/exercises_def/ex06/structures.py`) will be used:

```python
@dataclass(frozen=True)
class Polygon(GeoPrimitive):
    vertices: list[Point]
```
Note that if two polygons do not intersect, there are potentially infinite separating axes that can be computed. As a hint, return one axis per EdgePoly1-EdgePoly2 pairing. We also recommed returning axes that are orthogonal to the edges of each polygon only. 

You will implement the `get_axes` function, which returns takes in two `Polygon`: *p1* and *p2*.

You will output a list of Segments, each representing a Separating Axis. We recommend constructing each segment as the same length and long enough to cover both polygons. (A common heuristic is to make each segment of length 10)

The checker will not verify your implementation for this step, so we encourage that you do your own testing. 

#### Step 2c: Separating Axis Theorem for Two Polygons

In this step, we bring it all together and implement the Separating Axis Theorem for two polygons. We will be working 

We will be modifying the FIRST case in the `separating_axis_thm` function. 

Using the methods you have previously implemented: `get_axes`, `proj_polygon`, and `overlap`, determine using the Separating Axis Theorem if two polygons intersect with each other or not. 

The `separating_axis_thm` function takes in two Polygons as inputs: *p1* and *p2* and returns a tuple with a mandatory argument and an optional argument. 

The first argument is a `bool` that is *True* if the Polygons collide, and *False* if they do not. 

The second argument is an optional `Segment` which you can use to verify which Segment you are projecting against in your implementation of the Separating Axis Theorem. 

Note: In the instructor solution, most of the code is written in the previously implemented methods, and we only use the `separating_axis_thm` function to put all the pieces together. 


Test cases are provided in the online checker for this exercise. 


#### Step 3a: Return a list of Candiadate Separating Axes for a Polygon and a Circle
We now move to computing separating axes for a Polygon and a Circle. 

We will use the Circle GeoPrimitive in (`src/pdm4ar/exercises_def/ex06/structures.py`):

```python
@dataclass(frozen=True)
class Circle(GeoPrimitive):
    center: Point
    radius: float
```

You will implement the function `get_axes_cp` that takes as inputs a `Circle` *circ* and a `Polygon` *poly* and returns a list of Segments, which will represent the Axes.  

Hint: Notice that the circle is a polygon with infinite number of edges. Fortunately we do not need to check all axes normal to the edges.
It's sufficient to check the axes normal to the polygon edges plus ONE axis formed by the circle center and the closest vertice of the polygon.


The checker will not verify your implementation for this step, so we encourage that you do your own testing. 


#### Step 3b: Separating Axis Theorem for a Polygon and a Circle

We will be modifying the SECOND case in the `separating_axis_thm` function. 

The `separating_axis_thm` function takes in a Polygon and a Circle as inputs: *p1* and *p2* and returns a tuple with a mandatory argument and an optional argument. 

The first argument is a `bool` that is *True* if the shapes collide, and *False* if they do not. 

The second argument is an optional `Segment` which you can use to verify which Segment you are projecting against in your implementation of the Separating Axis Theorem. 

Note: In the instructor solution, most of the code is written in the previously implemented methods, and we only use the `separating_axis_thm` function to put all the pieces together. 


Test cases are provided in the online checker for this exercise. 





### Collision check module

In the second part of this exercise, we leverage another method of computing collisions: through intersecting shapes and triangulation. Triangulation is less widely used than Separating Axis Theorem, although it is an intuitive way to break down large polygons into more manageable triangular shapes. For this exercise, the `CollisionPrimitives` class within `src/pdm4ar/exercises/ex06/collision_primitives.py` is given to you. Notably, the following functions:
`circle_point_collision`,
`triangle_point_collision`,
`polygon_point_collision`,
`circle_segment_collision`,
`sample_segment`,
`triangle_segment_collision`,
`polygon_segment_collision`,
`polygon_segment_collision_aabb`,
`_poly_to_aabb`
We encourage you to read through these given functions, as you will be calling them in the next part of the exercise. 
 
Please note that, for the remaining part of this exercise, we will use a circle-shaped differential drive robot. 
For this part of the exercise, you can assume that our robot will move inside a 2D world that contains fixed obstacles on a pre-defined path. 
The obstacles inside the world must be circular, triangular or polygon-shaped. 
For each of the following steps, you will implement different methods to check collisions for the possible path of our robot. Do note that this second part of the exercise is designed for you to implement different approaches to solve the collision-checking problem. Therefore, the code for each of the collision-checking functions should be distinct from one another.

To represent the path of robot, the following data structure (`src/pdm4ar/exercises_def/ex06/structures.py`) will be used:

```python
@dataclass(frozen=True)
class Path:
    waypoints: List[Point]
```

Please note that the definition of `Path` and `Polygons` are similar. 
`Polygon` class connects the first and last vertices by default. 
However, there isn't any connection between first and last waypoints in `Path` objects. 
For the remaining part of the exercise, you are free to use or modify `check_collision` function. You may also not use it.
This functions takes two `GeoPrimitives` and check the collision between them by using the primitives implemented in the first part of this exercise.

The task is to implement the functions in the `CollisionChecker` class in `src/pdm4ar/exercises/ex06/collision_checker.py`.

#### Step 4: Collision Checking Procedure for Circle-shaped Robot

In this step, you will implement a baseline version for collision checking by using the primitives implemented before. You should not use `shapely` here. (We will check!)
The aim of this part is to check if a candidate path for our circular robot is collision-free. 

You will implement `path_collision_check` function which returns the `Segment` indices of the given `Path` which are in collision with the given obstacles. 
This function takes a `Path` *t*, the radius of the robot's occupancy *r*, and a list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which are in collision with any of the obstacles.

#### Step 5: Collision Checking via Occupancy Grid

The aim and all the assumptions are same as `Step 8`. 
However, in this step, you will use different approach for collision checking. 
You are asked to implement collision checking via occupancy grids. 
You will initially create an occupancy grid of the given environment. 
Then using the occupancy grid, you will find the segments of the path in which our robot will collide. You may use the functionalities of `shapely` here. Note that you will have to convert the `GeoPrimitives` to `shapely` geometries in order to work with `shapely`.

In this step, you will implement `path_collision_check_occupancy_grid` function which returns the `Segment` indices of the given `Path` which collides with any of the given obstacles. 
This function takes `Path` *t*, radius of the robot *r*, and list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which collides with any of the obstacles. Note that due to the discrete nature of an occupancy grid, it is completely reasonable that the method might not result in a perfect accuracy of 1.0.

#### Step 6: Collision Checking using R-Trees

The aim and all of the assumptions are same as `Step 8`. 
Like previous steps, the aim is to find the segments of the path in which our circular robot will collide. 
However, in this step you will use R-Tree to increase the execution time performance of your collision check module. 
R-Tree is an important optimization approach that is used in collision checking. 
For environments with high number of obstacles, it provides us an execution time decrease via its bounding box volume hierarchy structure. 
In this method, you will build an R-Tree of the given obstacles. You may use the functionalities of `shapely` here, including `STRTree`.
You are also free to implement your own R-Tree if you wish.

In this step, you will implement `path_collision_check_r_tree` function which returns the `Segment` indices of the given `Path` which collides with any of the given obstacles. 
This function takes `Path` *t*, radius of the robot *r*, and list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which collides with any of the obstacles.

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

In this step, you will implement `collision_check_robot_frame` function which returns the *True* if robot will collide with any of the fixed obstacles during its movement until its next pose. 
This function takes radius of the robot *r*, current pose `SE2transform`, next pose `SE2transform`, and list of observed obstacles in robot frame as arguments. 

#### Step 8: Collision Checking via Safety Certificates

The aim and all of the assumptions are same as `Step 8`. 
Like the previous steps, the aim is to find the segments of the path in which our circular robot will collide. 
However, in this step you will use a different optimization method called Safety Certificates. 
For environments with small number of obstacles but high number of points to be checked, it provides us an execution time decrease via the approach it uses on collision check. 
To obtain detailed information about the algorithm, you can check the part that is related to the safety certificates from the [given paper](https://journals.sagepub.com/doi/full/10.1177/0278364915625345) (`Algorithm 1`). You may use the functionalities of `shapely` here (the distance between a point and an obstacle can be easily calculated in `shapely` via the `distance` function).

In this step, you will implement `path_collision_check_safety_certificate` function which returns the `Segment` indices of the given `Path` which collides with any of the given obstacles. 
This function takes `Path` *t*, radius of the robot *r*, and list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which collides with any of the obstacles.

### Evaluation

For this exercise our performance metric is accuracy and execution time.
As test data, for each step random inputs are generated with the algorithm provided in `src/pdm4ar/exercises_def/ex06/data.py`. 
For each step, there are multiple test cases. 
The accuracies of steps 1-6 are calculated by the ratio of the correct answers. 
For the steps 6-11, List of indices are converted into a boolean list which represents whether there is a collision on each line segment of the path.
The accuracies of steps 6-11 are calculated by the average of the accuracy of test cases. 
Execution time of each step is calculated as an average of its test cases.
Lastly, accuracies and execution times of each step are aggregated as weighted average. 

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

There are also times where you may be dealing with calculations involving lots of floating point numbers and you may wish to compare the result against a certain value. The `math.isclose` method might be helpful as a direct `==` comparison will likely return *False* more often than not.

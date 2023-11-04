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

We start off by implementing some collision check primitives for basic geometric shapes. 
These will come in handy later on. 
The first step is to implement the functions inside the `CollisionPrimitives` class in `src/pdm4ar/exercises/ex06/collision_primitives.py` file.

In this section, we suggest the use of linear algebra modules like `numpy`. 
However, you are not allowed to use modules that implement collision checking directly such as `shapely`.

#### Step 1: Point-Circle Collision Checking Procedure

In this step, you will implement a function that checks whether a point is inside the given circle or not.
To represent a point and circle, the following data structures (`src/pdm4ar/exercises_def/ex06/structures.py`) will be used:

```python
@dataclass(frozen=True)
class Point(GeoPrimitive):
    x: float
    y: float

@dataclass(frozen=True)
class Circle(GeoPrimitive):
    center: Point
    radius: float
```

In this part, you will implement the `circle_point_collision` function of `CollisionPrimitives` class. 
This function takes a `Circle` *c*, and  a `Point` *p* as arguments. It return *True* if given `Point` is inside the `Circle`.

#### Step 2: Point-Triangle Collision Checking Procedure

In this step, you will implement a function that checks whether a point is inside the given triangle or not. 
You can find an explanation of a suggested procedure [here](http://www.jeffreythompson.org/collision-detection/tri-point.php#:~:text=To%20test%20if%20a%20point,the%20corners%20of%20the%20triangle.). 
Please note that you could use the procedure described [here](http://www.jeffreythompson.org/collision-detection/tri-point.php#:~:text=To%20test%20if%20a%20point,the%20corners%20of%20the%20triangle.), but you are also free to use any other algorithms. 
To represent the triangle, the following data structure (`src/pdm4ar/exercises_def/ex06/structures.py`) will be used:

```python
@dataclass(frozen=True)
class Triangle(GeoPrimitive):
    v1: Point
    v2: Point
    v3: Point
```

In this part, you will implement the `triangle_point_collision` function of `CollisionPrimitives` class. 
This function takes a `Triangle` *t*, and  a `Point` *p* as arguments. 
It returns *True* if given `Point` is inside the `Triangle`.

#### Step 3: Point-Polygon Collision Checking Procedure

In this step, you will implement a function that checks whether a point is inside the given polygon or not. 
A possible approach is to use Delaunay triangulation to decompose a `Polygon` into a set of `Triangle`s. 
For triangulation, the `triangulate` function of the [triangle module](https://github.com/drufat/triangle) may be used. 
The documentation of this module can be found [here](https://rufat.be/triangle/). The `triangle_point_collision` function can then be reused to check for collisions. To represent the polygon, the following data structure (`src/pdm4ar/exercises_def/ex06/structures.py`) will be used:

```python
@dataclass(frozen=True)
class Polygon(GeoPrimitive):
    vertices: List[Point]
```

In this part, you will implement the `polygon_point_collision` function of `CollisionPrimitives` class. 
This function takes a `Polygon` *poly*, and  a `Point` *p* as arguments. 
It returns *True* if given `Point` is inside the `Polygon`.

#### Step 4: Segment-Circle Collision Checking Procedure

In this step, you will implement a function that checks whether a segment collides with a circle or not. 
There is an explanation of a suggested procedure [here](https://www.jeffreythompson.org/collision-detection/line-circle.php). 
Please note that you could use the procedure described [here](http://www.jeffreythompson.org/collision-detection/tri-point.php#:~:text=To%20test%20if%20a%20point,the%20corners%20of%20the%20triangle.), but you are also free to use any other algorithms. 
To represent the segment, the following data structure (`src/pdm4ar/exercises_def/ex06/structures.py`) will be used:

```python
@dataclass(frozen=True)
class Segment:
    p1: Point
    p2: Point
```

In this part, you will implement the `circle_segment_collision` function of `CollisionPrimitives` class. 
This function takes a `Cricle` *c*, and  a `Segment` *s* as arguments. 
It returns *True* if given `Segment` collides with `Circle`.

#### Step 5: Segment-Triangle Collision Checking Procedure

In this step, you will implement a function that checks whether a segment collides with a triangle or not. 
One idea is to sample points on the segment and check whether they collide or not via the `triangle_point_collision` function. You may also use any other algorithms.

In this part, you will implement the `triangle_segment_collision` function of `CollisionPrimitives` class. 
This function takes a `Triangle` *t*, and  a `Segment` *s* as arguments. 
It returns *True* if given `Segment` collides with `Triangle`.

#### Step 6: Segment-Polygon Collision Checking Procedure

In this step, you will implement a function that checks whether a segment collides with a polygon or not. 
One idea is to sample points on the segment and check whether they collide or not via the `polygon_point_collision` function. You may also use any other algorithms.

In this part, you will implement the `polygon_segment_collision` function of `CollisionPrimitives` class. 
This function takes a `Polygon` *poly*, and  a `Segment` *s* as arguments. 
It returns *True* if given `Segment` collides with `Polygon`.

#### Step 7: Optimization via Axis-Aligned Bounding Boxes

The execution time performance of the collision checker is quite important for an efficient planning module. 
In this step, the aim is to improve the execution time performance of the `polygon_segment_collision` primitive. 
We can achieve the same result by simply checking the points within the AABB. In this way, we can get rid of unnecessary collision checks. 
As a result, we aim to see the execution time difference between pure collision checking and collision checking with the optimization.

In this part, you will implement `polygon_line_collision_aabb` function of `CollisionPrimitives` class. 
This function takes a `Polygon` *poly*, and  a `Segment` *s* as arguments. 
It returns *True* if given `Segment` collides with `Polygon`. 
You are free to use or not to use `_poly_to_aabb` function to calculate `AABB`.

### Collision check module

In the second part of this exercise, we leverage the collision primitives that were just implemented to perform collision checking for our robot. 
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

#### Step 8: Collision Checking Procedure for Circle-shaped Robot

In this step, you will implement a baseline version for collision checking by using the primitives implemented before. You should not use `shapely` here. 
The aim of this part is to check if a candidate path for our circular robot is collision-free. 

You will implement `path_collision_check` function which returns the `Segment` indices of the given `Path` which are in collision with the given obstacles. 
This function takes a `Path` *t*, the radius of the robot's occupancy *r*, and a list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which are in collision with any of the obstacles.

#### Step 9: Collision Checking via Occupancy Grid

The aim and all the assumptions are same as `Step 8`. 
However, in this step, you will use different approach for collision checking. 
You are asked to implement collision checking via occupancy grids. 
You will initially create an occupancy grid of the given environment. 
Then using the occupancy grid, you will find the segments of the path in which our robot will collide. You may use the functionalities of `shapely` here. Note that you will have to convert the `GeoPrimitives` to `shapely` geometries in order to work with `shapely`.

In this step, you will implement `path_collision_check_occupancy_grid` function which returns the `Segment` indices of the given `Path` which collides with any of the given obstacles. 
This function takes `Path` *t*, radius of the robot *r*, and list of obstacles as arguments. 
It returns the list of indices which represents the `Segment`s of the `Path` which collides with any of the obstacles. Note that due to the discrete nature of an occupancy grid, it is completely reasonable that the method might not result in a perfect accuracy of 1.0.

#### Step 10: Collision Checking using R-Trees

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

#### Step 11: Collision Checking in Robot Frame

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

#### Step 12: Collision Checking via Safety Certificates

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
| 01          | 15                       | 5                 | 0                     |
| 02          | 15                       | 10                | 0                     |
| 03          | 15                       | 10                | 0                     |
| 04          | 15                       | 10                | 0                     |
| 05          | 15                       | 10                | 0                     |
| 06          | 15                       | 5                 | 0                     |
| 07          | 15                       | 5                 | 0                     |
| 08          | 10                       | 20                | 20                    |
| 09          | 10                       | 20                | 20                    |
| 10          | 10                       | 30                | 30                    |
| 11          | 10                       | 20                | 20                    |
| 12          | 10                       | 30                | 30                    |

### How to run

Make sure to update your repo before running the exercise. 
Please refer to [Hello World](01-helloworld.md) for instructions. If you wish to use the `triangle` module, you should run `pip install triangle` in the terminal.

### Advice

Be cautious of clashing class names between our self-defined `GeoPrimitive` classes and the `shapely` classes. It is not recommended to run the following: `import triangle` or `from shapely import *` as these will result in errors due to identical class/module names. You may instead choose to use aliases for your imported modules (e.g. `import triangle as tr` or `from shapely.geometry import Point as shapelyPoint`) or to just import the methods that you need (e.g. `from triangle import triangulate`).

There are also times where you may be dealing with calculations involving lots of floating point numbers and you may wish to compare the result against a certain value. The `math.isclose` method might be helpful as a direct `==` comparison will likely return *False* more often than not.

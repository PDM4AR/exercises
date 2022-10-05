# Collision check :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Collision check

In this module, we implement collision checking routines. 
The overall goal is to obtain a module that can perform collision check for a given robot and one of its candidate paths.

### Collision check primitives

We start off by implementing some collision check primitives for basic geometric shapes. These will come handy later on. 
The first task is to implement the functions inside the `CollisionPrimitives` class in `src/pdm4ar/exercises/ex_collision_check/collision_primitives.py` file. In this section, you are allowed to use linear algebra modules like `numpy`. But you are not allowed to use modules that implements collision check directly such as `shapely`.

#### Step 1: Point-Circle Collision Checking Procedure

The aim of this step is to implement a function that checks whether a point is inside the given circle or not. To represent a point and circle, the following data structures (`src/pdm4ar/exercises_def/ex_collision_check/data.py`) will be used:

```python
@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float):
        return Point(self.x * scalar, self.y * scalar)

    def rotate(self, theta, origin):
        qx = (
            origin.x
            + np.cos(theta) * (self.x - origin.x)
            - np.sin(theta) * (self.y - origin.y)
        )
        qy = (
            origin.y
            + np.sin(theta) * (self.x - origin.x)
            + np.cos(theta) * (self.y - origin.y)
        )

        return Point(qy, qx)

@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float
```

In this part, you will implement the `circle_point_collision` function of `CollisionPrimitives` class.

#### Step 2: Point-Triangle Collision Checking Procedure

The aim of this step is to implement a function that checks whether a point is inside the given triangle or not. The algorithm which will be implemented is explained [here](http://www.jeffreythompson.org/collision-detection/tri-point.php#:~:text=To%20test%20if%20a%20point,the%20corners%20of%20the%20triangle.). To represent the triangle, the following data structure (`src/pdm4ar/exercises_def/ex_collision_check/data.py`) will be used:

```python
@dataclass(frozen=True)
class Triangle:
    p1: Point
    p2: Point
    p3: Point
```

In this part, you will implement the `triangle_point_collision` function of `CollisionPrimitives` class.

#### Step 3: Point-Polygon Collision Checking Procedure

The aim of this step is to implement a function that checks whether a point is inside the given polygon or not. Initially polygon will be decomposed into triangles via the `triangulate` function of the [triangle module](https://github.com/drufat/triangle). Then a collision check will be applied with the function implemented in `Step 2`. To represent the polygon, the following data structure (`src/pdm4ar/exercises_def/ex_collision_check/data.py`) will be used:

```python
@dataclass(frozen=True)
class Polygon:
    corners: List[Point]
```

In this part, you will implement the `polygon_point_collision` function of `CollisionPrimitives` class.

#### Step 4: Line-Circle Collision Checking Procedure

The aim is to implement a function that checks whether a line collides with a circle or not. The algorithm which will be implemented is explained [here](https://www.jeffreythompson.org/collision-detection/line-circle.php). To represent the line, the following data structure (`src/pdm4ar/exercises_def/ex_collision_check/data.py`) will be used:

```python
@dataclass(frozen=True)
class Line:
    p1: Point
    p2: Point
```

In this part, you will implement the `circle_line_collision` function of `CollisionPrimitives` class.

#### Step 5: Line-Polygon Collision Checking Procedure

The aim is to implement a function that checks whether a line collides with a polygon or not. The idea is to sample points on the line and check whether they collide or not via the function implemented on `Step 3`.

In this part, you will implement the `polygon_line_collision` function of `CollisionPrimitives` class.

#### Step 6: Optimization via Axis-Aligned Bounding Boxes

Inside a planning algorithm, one of the most computationally expensive part is collision check due to ist frequent usage. Therefore, it is important to have a collision checking algorithm as optimized as possible. In this step, the aim is to remove unnecessary collision check operations via axis-aligned bounding boxes. As a result, we aim to see the execution time difference between pure collision checking and collision checking with the optimization.

In this part, you will implement the `polygon_line_collision_aabb` function of `CollisionPrimitives` class.

### Collision check module

In the second part of this exercise, you will implement the collision check module by using the collision primitives that are implemented in the first part. In this part, you will implement the functions in `CollisionChecker` class in `src/pdm4ar/exercises/ex_collision_check/collision_checker.py` file.

#### Step 7: Collision Checking Procedure for Circle-shaped Robot

In this step, the aim is to find the collisions inside a 2D world that contains fixed polygon-shaped obstacles. The input will be a set of points that represents the path of the robot. Because, robot is a circle-shaped collision check can be done by extending the sizes of obstacles by the radius of the robot, and checking line-obstacle collisions. To represent the path of robot, the following data structure (`src/pdm4ar/exercises_def/ex_collision_check/data.py`) will be used:

```python
@dataclass(frozen=True)
class Path:
    waypoints: List[Point]
```
In this step, two functions will be implemented. Initially, a function (`extend_obstacle`) that takes a polygon and a radius as input and returns the extended version of that polygon. And the second function (`path_collision_check`) will firstly extend all obstacles inside the environment, and check the collision via functions that were implemented before. As a result, `path_collision_check` function returns the list of line segment indices where there exists a collision.

#### Step 8: Collision Checking via Occupancy Grid

In this step, the aim is same as `Step 7`. However, in this step, you will use different approach for collision checking. Initially, you will extend all obstacles inside the environment. Then, you will create an occupancy grid of the map. Lastly, you will return the list of line segment indices where there exists a collision. In this step, you will implement `path_collision_check_occupancy_grid` function.

#### Step 9: Collision Checking on Robot Frame

In this step, the function (`collision_check_robot_frame`) which will be implemented will be called sequentially. At each step, current pose of the robot, and currently observed obstacles in the robot frame will be given as input. At each step, observed obstacles will be converted into the fixed frame and stored. Then, at each step, the function will return whether there exists a collision of not between current position and next position of the robot with the function implemented at the first part of the exercise. 

#### Step 10: Collision Checking via R-Tree

The aim of the last step is to improve the execution time performance of the collision checking module via r-tree.

#### Step 11: Collision Checking via Safety Certificates

In this section, the aim is to implement a more efficient collision check procedure by using safety certificates. To obtain detailed information about the algorithm, you can check the part that is related to the safety certificates from the [given paper](https://journals.sagepub.com/doi/full/10.1177/0278364915625345) (`Algorithm 1`). To calculate the distance between a point and an obstacle, you are allowed to use `shapely` module.

##### Update your repo

Update your repo using

```shell
make update
```

this will pull the new exercises in your forked repo. If you get some merge conflicts it is because you might have
modified/moved files that you were not supposed to touch (i.e., outside the `exercises` folder).

###### Run the exercise

```shell
make run-collision_check
```

After running the exercise, a report will be generated that shows your results.
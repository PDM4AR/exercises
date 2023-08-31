# Dubins' Path :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>


## Problem overview
The self-driving taxi startup **ERMETH-ON-WHEELS** :red_car: has tasked you to write a path generation procedure for its fleet of autonomous vehicles.
The navigation team has already found an algorithm for constructing waypoints which the 
car needs to follow for completing different tasks (e.g. parking, lane following, etc.). 
Unfortunately, they are completely clueless (since they have not attended PDM4AR) on how to connect the points with physically realistic paths.
The car fleet is grounded and the investors are furious. 
Your job is it to use Dubins' method to construct a path between two waypoints,
representing the start and end configuration $(x,y,\theta)$ of the vehicle, so that the cars
can finally reach their desired locations and the investors are happy again.
We assume that the kinematics of the car can be described by a simplified bicycle model (also known as Dubins' car):

$\dot{x} = v \cos (\theta)$

$\dot{y} = v \sin (\theta)$

$\dot{\theta} = \frac{v}{L} \tan (\delta)$

$\theta$ is the heading angle of the base, $v$ is the current car velocity in the car's reference frame. 
The wheelbase of the car is given by L, the steering angle of the car is $\delta$.
Note that for this project we are only interested in the path the car needs to follow not the trajectory (i.e., the path as function of time).

### Structures
Please have a look at the files  `structures.py` to familiarize yourself with the data structures and the algorithms.
It is important that you return the right data structures (specified in the method definition). 
You will do your implementations in `algo.py`. The Dubins path planner class looks as follows:

```python
class Dubins(PathPlanner):
    def __int__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[SE2Transform]:
        """ Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        self.path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(self.path) 
        return se2_list
```
The position and orientation of an `SE2Transform` object can be accessed via the attributes `<SE2_object>.p` and `<SE2_object>.theta` respectively.

The final goal of this exercise is to implement the ```calculate_dubins_path``` method which should generate the optimal Dubins path for a given initial and end configuration. The method ```extract_path_points```, which is already implemented, will then generate a list of points $(x,y,\theta)$ which are on the returned path. These points can then be used e.g., in a reference tracking control pipeline. 

The problem is split into multiple (individually graded) subtask, to give you some guidance on how to eventually implement ```calculate_dubins_path```:

### Task
1. [5%] Given the above dynamics and parameters, calculate the minimum turning radius of the back wheel for a generic car.
Please implement this calculation in
 ```python 
 def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
  # TODO implement here your solution
  return DubinsParam(min_radius=0)
  ``` 
2. [5%] In order to generate a Dubins' path, we need to be able to compute the possible turning circles for a given configuration. Implement this functionality in the method
```python
def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    return TurningCircle(left_circle=Curve.create_circle(), right_circle=Curve.create_circle())
```
3. [20%] As a next step, we need to be able to connect two turning circles with a straight line segment which is tangent to the two circles.  To simplify computation for the next step, only return tangents which are possible for a car to complete starting from `circle_start` to `circle_end` (i.e., ignore tangents back to the start). Note in principle ```circle_start``` and `circle_end` can have different radii, however we will only check the case when the radii are equal, you free to implement a more general method.
 Only return the valid tangent line(s) which are physically possible, if no tangent exists return an empty ```List```. 
The order of the lines in the List is not important.
 Write your code in:
 ```python
 def calculate_tangent_btw_circles(circle_start: Curve, circle_start: Curve) -> List[Line]:
    # TODO implement here your solution
    return [] # i.e., [Line(),]
 ``` 

4. [60%] Use the helper methods implemented in the previous task to come up with the complete Dubins' path generation between two configurations. Please always return a valid Dubins' path (never an empty list, use the fact that an optimal Dubin's path has always a **fixed** number of segments). Keep segments with zero length (e.g., line with length = 0) in the returned list.
Implement it in:
```python
def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!
    return [] # e.g., [Curve(), Line(),..]
```

5. [10%] Thanks to your work the taxis are finally able to drive between waypoints. However, customers complain that the cars cannot
park backwards and sidewards when they should pick them up. Instead, they wait in the middle of the street...
In the following, extend the code implemented in task 4 to allow also for situation when the car needs to drive backwards. For simplicity, we will **only** consider cases with **three** path segments all completed in reverse (i.e., $C^{-}S^{-}C^{-}$ type paths) + all optimal forward dubins paths coming from ```calculate_dubins_path``` (don't forget to call this function in the new method). Use the `Gear.REVERSE` enum value to indicate that the car drives backwards. For example, the following reverse path is a $R^{-}S^{-}L^{-}$ path with the `start_config.theta` and `end_config.theta` values corresponding to the direction that the car is facing towards.

<p align="center">
  <img alt="img-name" src="https://github-production-user-asset-6210df.s3.amazonaws.com/92320167/264764780-23ed014d-9ebf-41b8-8216-e12c4600c3f4.jpg">
  <br>
    <em>R-S-L Reeds-Shepp path</em>
</p>

Hint: You may be able to reuse some functions you implemented before. Write your code in:
```python
def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid reeds/dubins path!
    return [] # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]
```

### Test cases and performance criteria

All of the described subtasks are individually graded on different test cases. For each task, we use an **accuracy** metric which we compute by counting the number of *correctly* computed test cases divided by the total number of test cases, i.e., for task $i$: $\frac{N_{correct,i}}{N_{task,i}}$. We define a test case to be computed *correctly*, if:

- For task 1,2,3: The computed return values match the ones of the solution up to some numerical tolerance.
- For task 4,5: The computed `Path` is in the set of **optimal** (i.e.,minimum distance) paths and follows the specification made in the problem description.


We provide some example test cases for each subtask. After running the exercise locally, you will find the report in the folder `out/ex05`. The provided test cases are not the same as the ones run on the test server used for grading, we advise you to additionally test your implementation using your own defined test cases, e.g., by modifying the existing ones in `src/pdm4ar/exercises_def/ex05/data.py`.

The final evaluation result is the normalized, weighted (see [%] in each description) sum of all the individual accuracy results of the subtasks and lies between [0,1]. 

### Update & Run

Please refer to [Hello World](01-helloworld.md) for instructions.

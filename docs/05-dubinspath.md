# Dubins' Path :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-helloworld.html" target="_top">Hello-world</a></td>
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
Note that for this project we are only interested in the path the car needs to follow not the trajectory (i.e. the path as function of time).

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

The final goal of this exercise is to implement the ```calculate_dubins_path``` method which should generate the optimal Dubins path for a given initial and end configuration. The method ```extract_path_points```, which is already implemented, will then generate a list of points $(x,y,\theta)$ which are on the returned path. These points can then be used e.g. in a reference tracking control pipeline. 

The problem is split into multiple (individually graded) subtask, to give you some guidance on how to eventually implement ```calculate_dubins_path```:

### Tasks
### 1. [5%] Computing minimum turning radius
Given the above dynamics and parameters, calculate the minimum turning radius of the back wheel for a generic car.
Please implement this calculation in:
 ```python 
 def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
  # TODO implement here your solution
  return DubinsParam(min_radius=0)
  ``` 
### 2. [5%] Computing turning circles
In order to generate a Dubins' path, we need to be able to compute the possible turning circles for a given configuration. Implement this functionality in the method:
```python
def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    return TurningCircle(left_circle=Curve.create_circle(), right_circle=Curve.create_circle())
```
### 3. [20%] Connecting the turning circles
As a next step, we need to be able to connect two turning circles with a straight line segment which is tangent to the two circles.  To simplify computation for the next step, only return tangents which are possible for a car to complete starting from `circle_start` to `circle_end` (i.e. ignore tangents back to the start). Additionally, only return the valid tangent line(s) in which the car follows the directions of the turning circles. Note in principle ```circle_start``` and `circle_end` can have different radii, however we will only check the case when the radii are equal, you free to implement a more general method. If no tangent exists return an empty ```List```. 
The order of the lines in the List is not important. Write your code in:
 ```python
 def calculate_tangent_btw_circles(circle_start: Curve, circle_start: Curve) -> List[Line]:
    # TODO implement here your solution
    return [] # i.e. [Line(),]
 ``` 

### 4. [50%] Generating Dubin's path
Use the helper methods implemented in the previous task to come up with the complete Dubins' path generation between two configurations. Please always return a valid Dubins' path (never an empty list, use the fact that an optimal Dubin's path has always a **fixed** number of segments). Keep segments with zero length (e.g. line with length = 0) in the returned list.
```python
def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!
    return [] # e.g. [Curve(), Line(),..]
```
### 5. [10%] Spline vs. Dubins: Short-range Maneuver Comparison

Even though your Dubins planner from Task 4 generates physically valid paths, the navigation team now wants to experiment with smoother alternatives for short‑range maneuvers and parking. In particular, they are curious to know how a **cubic Hermite spline** would compare to the optimal Dubins path in terms of:

- The path length  
- The feasibility with respect to the same curvature constraint used for Dubins  
- The qualitative difference in shape  

Your task is to **implement a function** that:  
1. Computes the optimal Dubins path between two given configurations using your `calculate_dubins_path` from Task 4.  
2. Constructs a **cubic Hermite spline** between the same start and end configurations, using the vehicle headings to define the spline tangents.  
3. Computes:
   - The length of the Dubins path  
   - The length of the spline  
   - Whether the spline is **feasible** (i.e., its curvature never exceeds `1/radius`)  
4. Returns the above quantities **together with** the spline parameters:
   - Tangent vector at the start (`t0`)  
   - Tangent vector at the end (`t1`)  
   - Start position (`p0`)  
   - End position (`p1`)  

We provide the method signature below. You must implement it in:

```python
def compare_spline_to_dubins(
    start_config: SE2Transform, end_config: SE2Transform, radius: float
) -> tuple[float, float, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare the Dubins path and a cubic Hermite spline between two configurations.

    Returns:
        dubins_length: length of optimal Dubins path
        spline_length: length of Hermite spline
        is_feasible: True if spline curvature <= 1/radius everywhere
        t0: tangent vector at start 
        t1: tangent vector at end 
        p0: start position (2D)
        p1: end position (2D)
    """
    # TODO implement here your solution
    return 0.0, 0.0, True, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
```

**Hint**  
- A cubic Hermite spline is a piecewise polynomial interpolation method where each segment is defined by two endpoints and their corresponding tangents. Given two points $p_0, p_1 \in \mathbb{R}^2$ and their tangents $t_0, t_1 \in \mathbb{R}^2$, the cubic Hermite spline for $s \in [0, 1]$ is:

  $h(s) = h_{00}(s) p_0 + h_{10}(s) t_0 + h_{01}(s) p_1 + h_{11}(s) t_1$

  where the Hermite basis functions are:  
  - $h_{00}(s) = 2s^3 - 3s^2 + 1$
  - $h_{10}(s) = s^3 - 2s^2 + s$  
  - $h_{01}(s) = -2s^3 + 3s^2$  
  - $h_{11}(s) = s^3 - s^2$

- To **approximate curvature geometrically**, sample a dense set of points along the spline. Then for each triplet of consecutive points $a, b, c$:
  - Compute $\vec{ba} = b - a$, $\vec{bc} = c - b$
  - Skip the triplet if any vector is near-zero (to avoid instability)
  - Compute the angle $\Delta\theta$ between $\vec{ba}$ and $\vec{bc}$
  - Estimate curvature locally using:

    $\kappa \approx \frac{\Delta\theta}{\|\vec{ba}\|}$

  - Keep track of the maximum curvature along the spline and mark the spline as **feasible** if $\max \kappa \leq 1/\text{radius}$

- Special case: when the start and end **positions** are the same:
  - Same heading → **feasible**  
  - Different heading → **infeasible** (turn-in-place is not allowed for Dubins motion)

**Note on tangent vector scale**

The tangent vectors $t_0$ and $t_1$, which define the direction of the Hermite spline at the start and end points, must be **scaled by the distance between the start and end positions**. This ensures that the shape of the spline is consistent and comparable across different queries. The team has decided to use this distance as the standard scale for all tangent vectors.


### 6. [10%] Computing reversing path
Thanks to your work the taxis are finally able to drive between waypoints. However, customers complain that the cars cannot
park backwards and sidewards when they should pick them up. Instead, they wait in the middle of the street...
In the following, extend the code implemented in task 4 to allow also for situation when the car needs to drive backwards. For simplicity, we will **only** consider cases with **three** path segments all completed in reverse (i.e. $C^{-}S^{-}C^{-}$ and $C^{-}C^{-}C^{-}$ type paths) + all optimal forward dubins paths coming from ```calculate_dubins_path``` (don't forget to call this function in the new method). Use the `Gear.REVERSE` enum value to indicate that the car drives backwards. For example, the following reverse path is a $R^{-}S^{-}L^{-}$ path (i.e. the direction of steering wheel input) with the `start_config.theta` and `end_config.theta` values corresponding to the direction that the car is facing towards.

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
    return [] # e.g. [Curve(..,gear=Gear.REVERSE), Curve(),..]
```

### Test cases and performance criteria

All of the described subtasks are individually graded on different test cases. For each task, we use an **accuracy** metric which we compute by counting the number of *correctly* computed test cases divided by the total number of test cases, i.e. for task $i$: $\frac{N_{correct,i}}{N_{task,i}}$. We define a test case to be computed *correctly*, if:

- For task 1,2,3: The computed return values match the ones of the solution up to some numerical tolerance. 
- For task 4,6: The computed `Path` is in the set of **optimal** (i.e.minimum distance) paths and follows the specification made in the problem description. 
- For task 5: The computed values for Dubins length, spline length, and feasibility must match the reference solution within a given numerical tolerance.


We provide some example test cases for each subtask. After running the exercise locally, you will find the report in the folder `out/ex05`. The provided test cases are not the same as the ones run on the test server used for grading, we advise you to additionally test your implementation using your own defined test cases, e.g. by modifying the existing ones in `src/pdm4ar/exercises_def/ex05/data.py`.

The final evaluation result is the normalized, weighted (see [%] in each description) sum of all the individual accuracy results of the subtasks and lies between [0,1]. 


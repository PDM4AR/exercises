# Dubins' Path :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Dubins' Path

In this programming exercises you will learn how to implement Dubins' path.

### Overview
The self-driving taxi startup **ERMETH-ON-WHEELS** :red_car: has tasked you to write a path generation procedure for its fleet of 
autonomous vehicles. The navigation and planning teams have already found an algorithm for constructing waypoints which the 
cars need to follow for completing different tasks (e.g., parking). 
Unfortunately, they are completely clueless (since they have not attended PDM4AR) on how to construct physically realistic paths in between.
The car fleet is grounded - the investors are furious. Your job is it to use Dubins' method to construct a path between two waypoints,
which represent the start and end configuration $(x,y,\theta)$ of the vehicle, so that the cars
can finally reach their desired locations and the investors are happy again ($).
You can assume that the provided waypoints are physically possible and are reachable.
We assume that the kinematics of the car can be described by a simplified bicycle model (also known as Dubins' car):

$\dot{x} = v \cos (\theta)$

$\dot{y} = v \sin (\theta)$

$\dot{\theta} = \frac{v}{L} \tan (\delta)$

$\theta$ is the heading angle of the base, $v$ is the current car velocity in the cars reference frame. 
The wheel-base of the car is given by L, the steering angle of the car is $\delta$.
Note that for this project we are only interested in the path the car needs to follow not the trajectory (i.e., the path as function of time).
### Structure
Please have a look at the files  `structures.py` to familiarize yourself with the data structures and the algorithms. It is important that you return the right data structures (specified in the method definition). You will do your implementations 
in `algo.py`. First, have a look at the Dubins PathPlanner class:


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
The goal of this exercise is to implement the ```calculate_dubins_path``` method which should generate the optimal Dubins path for a given initial and end configuration. The method ```extract_path_points```, which is already implemented, will then generate a list of points $(x,y,\theta)$ which are on the returned path. These points can then be used e.g., in a reference tracking control pipeline. 

We have split the problem into multiple (individually graded) subtask, to give you some guidance on how to finally implement ```calculate_dubins_path```:

### Task
1. [xx%] Given the above dynamics and parameters, calculate the minimum turning radius of a generic car.
Please implement this calculation in
 ```python 
 def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
  # TODO
  return DubinsParam(min_radius=0)
  ``` 
2. [xx%] In order to generate a Dubins' path, we need to be able to compute the possible turning circles for a given configuration. Implement this functionality in the method
```python
def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    return TurningCircle(left_circle=Curve.create_circle(), right_circle=Curve.create_circle())
```
3. [xx%] As a next step, we need to be able to connect two turning circles with a straight line segment which is tangent to the two circles.
 Write your code in:
 ```python
 def calculate_tangent_btw_circles(circle1: Curve, circle2: Curve) -> List[Line]:
    # TODO implement here your solution
    return [] # i.e., [Line(),]
 ``` 
 Note in principle ```curve1``` and `curve2` can have different radii, however we will only check the case when the radii are equal, you free to implement a more general method however.
 Only return the valid tangent line(s) which are physically possible, if no tangent exists return an empty ```List```. 
The order of the lines in the List is not important.

4. [xx%,xx%] Use the helper methods implemented in the previous task to come up with the complete Dubins' path generation between two configurations. Please always return a valid Dubins' path (never an empty list, always the same length). Keep segments with zero length (e.g., line with length = 0) in the returned list.
Implement it in:
```python
def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Important: Please keep segments with zero length in the return list!
    return [] # e.g., [Curve(), Line(),..]

```

5. [xx%, bonus] Thanks to your work the taxis are finally able to drive between waypoints. However, customers complain that the cars cannot
park backwards and sidewards when they should pick them up. Instead, they wait in the middle of the street...
In the following, extend the code implemented in task 4 to allow also for situation when the car needs to drive backwards. For simplicity, we will only consider cases with three path segments all completed in reverse (i.e., $C^{-}S^{-}C^{-}$ type paths) + all optimal forward dubins paths coming from ```calculate_dubins_path``` (don't forget to call this function in the new method). Use the `Gear.REVERSE` enum to indicated segments were car needs to move backwards. Hint: You may be able to resuse some functions you implemented before. Write your code in:
```python

def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Important: Please keep segments with zero length in the return list!
    return [] # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]
```




#### Update & Run

Please refer to [Hello World](01-helloworld.md) for instructions.



#### Food for thoughts

* 

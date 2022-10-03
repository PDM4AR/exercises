# Dubins' Path :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Dubins' Path

In this programming exercises you will learn how to implement Dubins' path.

### Overview
The self-driving taxi startup *ERMETH-ON-WHEELS* :red_car: has tasked you to write a path generation procedure for its fleet of 
autonomous vehicles. The people from navigation and planning have already found an algorithm for constructing waypoints which the 
cars need to follow for completing different tasks (e.g. parking). 
Unfortunately, they are completely clueless (since they have not attended PDM4AR) on how to construct physically realistic paths in between.
The car fleet is grounded - the investors are furious. Your job is it to use Dubins' method to construct a path between two waypoints,
which represent the start and end configuration ($x,y,\theta$) of the vehicle, so that the cars
can finally reach their desired locations and the investors are happy again ($).
You can assume that the provided waypoints are physically possible and are reachable.
We assume that the kinematics of the car can be described by a simplified bicycle model (also known as Dubins' car):

$\begin{align}
 \dot{x} =& v \cos (\theta) \\
\dot{y} =& v \sin (\theta) \\
\dot{\theta} =& \frac{v}\{L} \tan (\delta) 
\end{align}$

$\theta$ is the heading angle of the base, $v$ is the current car velocity in the cars reference frame. 
The wheel-base of the car is given by L = 3 [m], the steering angle of the car is $\delta \in [-30^\circ, 30^\circ]$. 
Note that for this project we are only interested in the path the car needs to follow not the trajectory (i.e. the path as function of time).
### Structure
Please have a look at the files  `structures.py` and `algo_structures.py` to familiarize yourself with the data structures and the algorithms. It is important that you return the right data structures (specified in the method definition). Most (TODO or all) of your implementation 
will be done `algo.py`. 

### Task
1. [xx%] Given the above dynamics and parameters, calculate the minimum turning radius of a generic car.
Please implement this calculation in `calculate_car_turning_radius(wheel_base, max_steering_angle)` method so that we can test your implementation.
2. [xx%] In order to generate a Dubins' path, we need to be able to compute the possible turning circles for a given configuration. Implement this functionality in the method
`calculate_turning_circles(current_config: X, radius: float)`. Note that your free to choose whether
the right or left turn will be returned first as element (TODO or should we make it fix). Just remember your choice for the following tasks.
3. [xx%] As a next step, we need to be able to connect two turning circles with a straight line segment which is tangent to the two circles. Note that the circles can have different radii.
 Write your code in `calculate_tangent_btw_curves(curve1: Curve, curve2: Curve)`. Only return the valid tangent line(s) which are physically possible, if no tangent exists return an empty ```List```. 
Again, the order of the lines in the List/Tuple(TODO) is not important (TODO). Do not forget edge cases (0-length tangents when circles touch; TODO not sure if should give this hint)
4. [xx%,xx%] Use the helper methods implemented in the previous task to come up with the complete Dubins' path generation between two configurations.
Implement it in `calculate_dubins_path(start_config: X, end_config: X)`. 
5. [xx%] Thanks to your work the taxis are finally able to drive between waypoints. However, customers complain that the cars cannot
park backwards and sidewards when they should pick them up. Instead, they wait in the middle of the street...
In the following, extend the code implemented in task 4 to allow also for situation when the car needs to drive a straight line backwards. For simplicity, driving backwards
and turning at the sametime will be ignored (TODO TBD). Write your code in `calculate_dubins_path(start_config: X, end_config: X)`



#### Update your repo

Update your repo using

```shell
make update
```

this will pull the new exercises in your forked repo. If you get some merge conflicts it is because you might have
modified/moved files that you were not supposed to touch (i.e., outside the `exercises` folder).

###### Run the exercise

```shell
make run-exercise-xx
```

#### Food for thoughts

* 

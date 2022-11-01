# Optimization - Mixed Integer Linear Programming :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

##  Optimize your voyage to the *Two Slices*

You are the captain of a pirate ship attempting to travel through the *Short Route*, a misleading name for a dangerous area of the oceans, to reach the *Two Slices*, the legendary pirate treasure which lies on the last island of the *Short Route*.

The *Short Route* is an agglomerate of islands divided into *N* different groups, which will be called archipelagos from now on. Apart from the first and last archipelago which contains only one island each, all of the other archipelagos have the same *k* number of islands. 


## Task - TBD better

The map is an agglomerate of islands divided into *N* different groups, which will be called archipelagos from now on, identified with a number from *0* to *N-1*. Archipelagos n. 0 (first archipelago) and n. *N-1* (last archipelago) are composed of one island each. All of the other archipelagos (from n. 1 to n. *N-2*) are composed of the same amount *k* of islands.
Hence, the total number of islands is 1+(*N-2*)*k*+1 = (*N-2*)*k*+2. The islands are identified by a unique id, ranging from 0 to (*N-2*)*k*+1. They also have another id to know to which archipelagos they belong. Note that the distance of an island to other islands, or to the centroid of the archipelagos, is irrelevant in determining its archipelago of 
Your job is to compute a voyage plan (an ordered list of islands to be visited) that starts from the first archipelago and ends at the last archipelago, optimizing for some specific costs while satifying some constraints.

Here you can see two examples of correct planned voyages:

![example 1](https://user-images.githubusercontent.com/79461707/193420646-368a6b22-6271-420b-bbec-6afe73f6bb68.png)
![example 2](https://user-images.githubusercontent.com/79461707/193420649-e604125d-4781-4058-b17f-376d60ba687e.png)

You have to implement the optimization inside the function `solve_milp` in [src/pdm4ar/exercises/ex07/ex07.py](../src/pdm4ar/exercises/ex07/ex07.py). The input you receive is a `ProblemVoyage` structure, and you have to output back a `ProblemSolution` structure. Through the `ProblemVoyage` input you have access to the different active constraints and to the particular cost your voyage plan must satisfy and optimize.

---

### **Constraints**

> Due to the rare magnetic fields, the size of the ship, the distances, and the weather, you will experience different combinations of constraints during your voyage. You should plan your voyage accordingly, because the ocean does not forgive incautios pirates: failure to abide by the *Shourt Route*'s constraints can mean death.

The input of your function is a `ProblemVoyage` structure, in which you can find information about which constraints you should enforce. When a costraint has a **None** value, it means you don't have to take care of it. Multiple constraints can be active at the same time.


#### **Voyage order**

> The *Short Route* presents a weird and dangerous magnetic field. Travelling through it is not possible using a normal compass, but special compasses are needed. These special compases show the direction of the islands of the next archipelago: when you are at a specific island of an archipelago, they will tune with the magnetic field of the next archipelago, and so on, until you reach the last archipelago where the treasure is located. Trying to directly reach the last archipelago, or moving to an archipelago that is not the next in order is impossible and you will get lost in the ocean.

This constraint is always active. Your voyage plan must start from the first archipelago (where your ship is docked) and end at the last archipelago (where the pirate treasure awaits).  visiting one and only one island of the other archipelagos in between, following the order of the archipelago's identification number. This means that your ordered voyage plan must contain *N* islands and that these islands must belong to archipelagos with increasing identification numbers. Since the constraint is active in every test case, it is not listed within `constraints` to avoid reduntant information.

#### **Minimum nights**

> Everytime you arrive in a new island, the special compasses should start to tune to the new magnetic fields pointing to the islands of the next archipelagos. The number of nights to tune the special compasses varies depending on the island. Unfortunately, the fastest the tuning time of a magnetic field, the more technologically complex a special compass should be. Hence, you are only able to visit islands where the magnetic field pointing to the next archipelago is tunable by your special compass version. You cannot visit islands where the magnetic field requires less nights to tune than the minimum amount of nights specified by your compass. You cannot simply wait more time: if your compass is able to tune to magnetic field that requires 4 nights of waiting, you cannot visit an island with a 2-nights magnetic field tunining and wait for 2 more nights. The compass is simply not working with these faster tuning magnetic field.

When this constraint is active, when you arrive in an island you have to wait a specific amount of nights before departing towards the next island. Each island has its specific value of how many night you should stay in it before departing. Only the starting island an the ending island have a value of 0. You can only visit island whose number of nights is at least `min_nights_individual_island`.

#### **Minimum crew size**

> Pirate ships can be as small as life rafts or as big as vessels: since you are the captain, its dimensions depends on your wealth and ostentatiousness. Based on its size, the ship will always require a crew with a specific minimum amount of people (you included) to steer it and take care of it. 

When this constraint is active, the crew size cannot be smaller than `min_total_crew`, whichever island you are in during any moment of your voyage.

#### **Maximum crew size**

> Pirate ships can be as small as life rafts or as big as vessels: since you are the captain, its dimensions depends on your wealth and ostentatiousness. Based on its size, the ship will always allow a crew with a specific maximum amount of people (you included), since there is no room for everyone.

When this constraint is active, the crew size cannot be greater than `max_total_crew`, whichever island you are in during any moment of your voyage.

#### **Maximum duration individual journey**

> Your pirate ship presents another problem related to its dimensions: it is able to sail the sea only only up to a certain amount of hours in a row. The weather and the strenght of the currents that rule the *Short Route* make the autonomy span of the ships very limited. When you sail from an island you can reach only islands that are within this journey duration, or the ship will definetely sink. 

When this constraint is active, every journey to move among two islands must last at maximum `max_duration_individual_journey` hours. The time needed to go from one island to the next one can be inferred from the respective departure and arrival timetable. 

#### **Maximum L1-norm distance individual journey**

> Yet another problem for your pirate ship related to its dimensions: it is able to sail the sea only only up to a certain distance in a row. The weather and the strenght of the currents that rule the *Short Route* make the autonomy span of the ships very limited. When you sail from an island you can reach only islands that are within this journey distance, or the ship will definetely sink. 

When this constraint is active, the L1-norm length of every journey to move among two islands (so the L1-norm distance between the two islands) must be at maximum `max_L1_distance_individual_journey`. The L1-norm distance to go from one island to the next one can be inferred from the their *x* and *y* positions.

---

### **Cost functions**

> You want to conquer the *Two Slices*, the legendary pirate treasure that lies on the last island of the *Short Route*. It is a complicated voyage in itself, but that's not all! This is a gold age for piracy, and many other pirate ships are looking for the treasure. You don't only have to plan a voyage that will not sink your ship or get you lost (taking care of the magnetic fields, the crew, the weather...) but also plan a voyage that will get you to the treasure as fast as possible to leave behind the other ships, or with a crew as big as possible if there's a need to fight against other crews, or without wearing too much the ship.

The input of your function is a `ProblemVoyage` structure, in which you can find information about which cost (one cost only for each test) you should optimize (while enforcing the active constraints).


#### **Minimum nights to complete the voyage** (`min_total_nights`)

> The easiest way to seize the treasure: reach it as soon as possible, before every other pirate ship. Show to the other pirates the meaning of *haste*.

Plan a voyage such that the total number of nights spent is minimized.

#### **Maximum final crew size** (`max_final_crew`)

> You believe in the power of muscles and gunpowder: better reach the treasure with a large crew, to fight and win over other pirates who could reach the tresure before you.

Plan a voyage such that the final crew size is maximized.

#### **Minimize total sailing time** (`min_total_sailing_time`)

> You, a captain of a pirate ship? Puah! Since you are a seasick wimp, you want to spend as little time as possible on a ship in the middle of the sea under the effect of waves and currents.

Plan a voyage such that the total sailing time, i.e. the total number of hours spent in the sea between departing from an island to arrive to the next one, is minimized.

#### **Minimize total L1-norm travelled distance** (`min_total_travelled_L1_distance`)

> You are not the brightest person out there, and you think that travelling the shortest distance possible is the best way to reach the tresure before everyone else.

Plan a voyage such that the total L1-norm travelled distance, i.e. the sum of the L1-norm distances of the multiple journeys to depart from an island to arrive to another island, is minimized.

#### **Minimize the maximum individual sailing time** (`min_max_sailing_time`)

> In these dangerous waters, caution is never too much. Better not to sail the ship for too many hours in a row, close to its limit, to avoid unnecessary wearing.

Plan a voyage such that the maximum individual sailing time of the voyage, i.e. the maximum number of hours spent in a single journey to depart from an island to arrive to another island, is minimized.

---

## Data structures of the exercise

The task is to implement the `solve_optimization` function inside [src/pdm4ar/exercises/ex07/ex07.py](../src/pdm4ar/exercises/ex07/ex07.py), which takes as input a `ProblemVoyage` data structure and outputs a `ProblemSolution`.

The various data structures needed for the development of the exercise can be inspected in [src/pdm4ar/exercises_def/ex07/structures.py](../src/pdm4ar/exercises_def/ex07/structures.py). 

### **Island**

<details>
<summary><b>Detailed description</b></summary>

- The `id` integer attribute identifies uniquely the islands. The island of the first archipelago has always `id` = 0 while the island of the last archipelago has always `id` = *number of islands - 1*. If the archipelagos in between have 5 islands each, then the `id` of the islands of the second archipelago ranges from 1 to 5, the ones of the third archipelagos from 6 to 10, the ones of the fourth archipelago from 11 to 15, and so on. When you submit your optimized voyage plan, you are submitting an ordered list of the `id` of the islands you planned to visit.
- The `arch` integer attribute tells you to which of the *N* archipelagos the island belongs (*0*, ..., *N-1*).
- The `x` and `y` float attribute specifies the *x* and *y* position of the island in a cartesian reference system. The *Short Route* can be approximated as a flat plane, so you don't have to consider the the Earth's curvature.
- The `departure` and `arrival` float attributes are a timetable of the exact time you have to depart from or to arrive to the island, to exploit its specific weather to being able to set sail or to dock. Note that to keep things simple the decimal places are not representing the minutes in *mod* 60. A value of 8.43 doesn't mean 43 minutes past 8, but that it's 43% of an hour past 8. Treat it as a normal float value.
To keep things simple, the arrival times of all the islands are later than the departure times of all the islands. This means you are always departing and arriving around on the same day.
- The `nights` integer attribute specifies how many nights you have to spend on the island before you can depart to the next archipelago. If `nights` is 1, it means you arrive in the island and you depart the next day, irrelevant of the arrival/departure timetable.
- The `delta_crew` integer attribute specifies how many people will leave the crew (negative value) or how many join the crew (positive value) if you visit the island.

</details>

```python
@dataclass(frozen=True)
class Island:
    id: int
    arch: int
    x: float
    y: float
    departure: float
    arrival: float
    nights: int
    delta_crew: int
```

---

### **ProblemVoyage**

Input of the function `solve_milp` you have to implement.

<details>
<summary><b>Detailed description</b></summary>

- The `start_crew` integer attribute specify how many people are in the crew (included the captain) at the beginning of the voyage.
- The `islands` attribute is a tuple containing the islands' data. Since the islands in the tuple ar eordered based on then`id` and since each archipelago has the same amount of islands, you can use a smart indexing to access islands of the same archipelago.
- The `constraints` attribute:
    - The `cost_to_optimize` attribute contains a value amon
    - The `min_nights_individual_island` integer attribute is a constraint specifing the minimum amount of nights you have to spend in every island to get the ship fixed before departing again to a new island. The ocean currents are badly damaging the ship every time you set sail.
    - The `max_total_crew` and `min_total_crew` integer attributes specify the minimum and the maximum amount of people who can be in the crew at the same time. A small number of people are not adequate for the danger of the *Short Route*, and the ship is not big enough to host too many people.
    - The `max_duration_individual_journey` float attribute is a constraint specifing the maximum amount of hours each voyage from one island to the next can last, otherwise the damage of the ship will be too much and it will sink.
    - The `max_L1_distance_individual_journey` float attribute is a constraint specifing the maximum amount of hours each voyage from one island to the next can last, otherwise the damage of the ship will be too much and it will sink.

</details>

```python
@dataclass(frozen=True)
class ProblemVoyage:
    optimization_cost: CostType
    start_crew: int
    islands: Tuple[Island]
    constraints: Constraints  
```
with

```python
CostType = Literal[OptimizationCost.min_total_nights, 
                   OptimizationCost.max_final_crew,
                   OptimizationCost.min_total_sailing_time,
                   OptimizationCost.min_total_travelled_L1_distance,
                   OptimizationCost.min_max_sailing_time]

@dataclass(frozen=True)
class Constraints:
    min_nights_individual_island: Optional[int]
    min_total_crew: Optional[int]
    max_total_crew: Optional[int]
    max_duration_individual_journey: Optional[float]
    max_L1_distance_individual_journey: Optional[float]
```
---

### **ProblemSolution**

```python
@dataclass(frozen=True)
class ProblemSolution:
    feasibility: FeasibilityType
    voyage_plan: Optional[VoyagePlan]
```

with 

```python3
FeasibilityType = Literal[MilpFeasibility.feasible, MilpFeasibility.unfeasible]

VoyagePlan = List[int]
```
---

Used to store the optimal solution of a MILP problem. The `status` attributes specifies if the MILP problem was found unfeasible or feasible, using the `MilpFeasibility` attribute values. If the problem is feasible, `voyage_plan` is set with a list of the `id`s of the island in the order you plan to visit them. If it is unfeasible, the content of `voyage_plan` doesn't matter.

---

## Report

You can choose between five different levels of report types: 

- `ReportType.none`: no evalutation information at all, no report is generated.
- `ReportType.terminal`: print evaluation information on the terminal, no report is generated.
- `ReportType.report_txt`: print evaluation information on the terminal and in a textual report.
- `ReportType.report_viz`: print evaluation information on the terminal and in a visual report, with text and figures.
- `ReportType.report_viz_extra`: print evaluation information on the terminal and in a visual report, with text and figures and data of each island.

In [src/pdm4ar/exercises_def/ex07/visualization.py](../src/pdm4ar/exercises_def/ex07/visualization.py), you can set the *REPORT_TYPE* global variable. 

Note that depending on the number of islands, `report_viz` starts to be slower by a non-negligible amount of time, while `report_viz_extra` will be even slower.

Feel free to make your modifications to the visualization to match your debugging needs.

If your terminal is not correctly printing the colors (very improbable), or if you need a a more visually impaired

---

## Run the exercise

```shell
pip3 install -e [path/to/exercises_repo]
python3 [path/to/]src/pdm4ar/main.py --exercise 07
```

After running the exercise, a report will be generated that shows your results (if you enabled the report generation).

## Hints
To model the problem notice that in the environment there are already powerful libraries to solve optimization problems. For instance, [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) and [pulp](https://coin-or.github.io/pulp/))

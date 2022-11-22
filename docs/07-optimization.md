# Optimization (preview) :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>
<table>
  <tr>
    <th><i>No storytelling version:</i></th><td><a href="./07-optimization-clean.html" target="_top">here</a></td>
  </tr>
</table>


## Problem overview

> You are the captain of a pirate ship and you desire to travel through the *Short Route*, a misleading name for a dangerous area of the oceans filled with islands, to attempt to reach the *Two Slices*, the legendary pirate treasure which lies on the last island of the *Short Route*.

In this exercise you will learn how to solve an optimization problem translating concepts from the textual to the mathematical and programming realm.

The environment of the problem is a 2D map composed of islands each with its own features. The islands are divided into *N* different groups, which will be called archipelagos henceforth, identified with a number from *0* to *N-1*. Archipelagos n. *0* (first archipelago) and n. *N-1* (last archipelago) are composed of one island each. All of the other archipelagos (from n. *1* to n. *N-2*) are composed of the same amount *k* of islands.
Hence, the total number of islands is 1+(*N-2*)\**k* +1 = (*N-2*)\**k* +2. The islands are identified by a unique id, ranging from *0* to (*N-2*)\**k*+1. They also have another tag to discern to which archipelagos they belong. Note that the belonging of an island in a specific group, called archipelago, is not determined by position, topographic reasons, similarity, etc.: don't make any assumption, just take it as it is.

Your task is to compute a voyage plan (an ordered list of islands to be visited) optimized for a specific cost while satifying some constraints, starting from the first archipelago and ending at the last archipelago, visiting one and only one island of the other archipelagos following the order of the archipelago's identification number.

Here you can see two examples of correct planned voyages:

<!--- | ![Example1](https://user-images.githubusercontent.com/79461707/199726616-4b6eff71-90a0-4416-83de-e5c98260db46.png) | -->
| ![Example1](https://user-images.githubusercontent.com/79461707/199729360-26647058-7399-46fc-a4aa-aa2353171643.jpeg) |
|:--:|
| <b>voyage plan: 0 -> 1 -> 8 -> 12 -> 13</b>|

<!--- | ![Example2](https://user-images.githubusercontent.com/79461707/199726634-d0222b71-c160-42de-b7cb-c877febf7b40.png) | -->
| ![Example2](https://user-images.githubusercontent.com/79461707/199729335-40cfaca9-2256-498f-bd4e-163cb8429dba.jpeg) |
|:--:|
| <b>voyage plan: 0 -> 15 -> 21 -> 38 -> 67 -> 88 -> 91 -> 117 -> 130 -> 151 -> 165 -> 196 -> 203 -> 224 -> 235</b>|

---

You have to implement the optimization inside the function `solve_optimization` in [src/pdm4ar/exercises/ex07/ex07.py](../src/pdm4ar/exercises/ex07/ex07.py). The input you receive is a `ProblemVoyage` structure, and you have to output back a `SolutionVoyage` structure. Through the `ProblemVoyage` input you have access to the different active constraints and the specific cost your voyage plan must satisfy and optimize.

```python
def solve_optimization(problem: ProblemVoyage) -> SolutionVoyage:
    """
    Solve the optimization problem enforcing the active constraints

    Parameters
    ---
    problem : ProblemVoyage
        Contains the problem data: cost to optimize, starting crew, tuple of islands, 
        and information about active constraints.

    Returns
    ---
    out : SolutionVoyage
        Contains the feasibility status of the problem, and the optimal voyage plan
        as a list of ints if problem is feasible, else None
    """

    # toy examples with random voyage plans
    np.random.seed(None)
    if np.random.random() > 0.3:
        feasibility = Feasibility.feasible
        voyage_plan = list(np.random.randint(0,len(problem.islands), size=(min(7,len(problem.islands)),)))
    else:
        feasibility = Feasibility.unfeasible
        voyage_plan = None

    return SolutionVoyage(feasibility, voyage_plan)
```

---

### **Constraints**

> Due to the rare magnetic fields, the size of the ship, the distances, and the weather, you will experience different combinations of constraints during your adventure. You should plan your voyage accordingly, because the *Shourt Route* does not forgive incautios pirates.

The input of your function is a `ProblemVoyage` structure, in which you can find information about which constraints you should enforce. When a costraint has a **None** value, it means you don't have to take care of it. Multiple constraints can be active at the same time.


#### **Voyage order** (`voyage_order`)

> The *Short Route* presents a weird and dangerous magnetic field. Travelling through it is not possible using a normal compass, but special compasses are needed. These special compases show the direction of the islands of the next archipelago: when you are at a specific island of an archipelago, they will tune with the magnetic field of the next archipelago, and so on, until you reach the last archipelago where the treasure is located. Trying to directly reach the last archipelago or moving to an archipelago that is not the next in order is impossible and you will get lost in the ocean.

This constraint is always active. Your voyage plan must start from the first archipelago and end at the last archipelago, visiting one and only one island of the other archipelagos, following the order of the archipelago's identification number. This means that your ordered voyage plan must contain *N* islands and that these islands must belong to archipelagos with increasing identification numbers. Since the constraint is active in every test case, it is not listed within the `constraints` attribute of the `ProblemVoyage` input, to avoid reduntant information.

#### **Minimum nights** (`min_nights_individual_island`)

> Everytime you arrive in a new island, the special compasses should start to tune to the new magnetic fields pointing to the islands of the next archipelagos. The number of nights to tune the special compasses varies depending on the island. Unfortunately, the fastest the tuning time of a magnetic field, the more technologically complex a special compass should be to be able to tune it. Hence, you are only able to visit islands where the magnetic field pointing to the next archipelago is tunable by your special compass version. You cannot visit islands where the magnetic field requires less nights to tune than the minimum amount of nights specified by your compass. You cannot simply wait more time: if your compass is able to tune to magnetic field that requires 4 nights of waiting, you cannot visit an island with a 1, 2, or 3 nights magnetic field tunining and just wait for the remaining nights. The compass is simply not working with these faster tuning magnetic field.

When this constraint is active, you can only visit island whose number of waiting nights is at least `min_nights_individual_island`. When you arrive in an island you have to wait a specific amount of nights before departing towards the next island. Each island has its specific value of how many night you should stay in it before departing. Only the starting island an the ending island have a value of 0.

#### **Minimum crew size** (`min_total_crew`)

> Pirate ships can be as small as life rafts or as big as vessels: since you are the captain, its dimensions depends on your wealth and ostentatiousness. Based on its size, the ship will always require a crew with a specific minimum amount of people (you included) to steer it and take care of it. 

When this constraint is active, the crew size cannot be smaller than `min_total_crew`, whichever island you are in during any moment of your voyage.

#### **Maximum crew size** (`max_total_crew`)

> Pirate ships can be as small as life rafts or as big as vessels: since you are the captain, its dimensions depends on your wealth and ostentatiousness. Based on its size, the ship will always allow a crew with a specific maximum amount of people (you included), since there is no room for everyone.

When this constraint is active, the crew size cannot be greater than `max_total_crew`, whichever island you are in during any moment of your voyage.

#### **Maximum duration individual journey** (`max_duration_individual_journey`)

> Your pirate ship presents structural limitations: it is able to sail the sea only only up to a certain amount of hours during the same island-to-island journey. The weather and the strenght of the currents that rule the *Short Route* make the autonomy span of the ships very limited. When you sail from an island you can reach only islands that are within this journey duration, or the ship will definetely sink. 

When this constraint is active, every island-to-island journey to move among two islands must last at maximum `max_duration_individual_journey` hours. The time needed to go from one island to the next one can be inferred from the respective departure and arrival timetable.

#### **Maximum L1-norm distance individual journey** (`max_L1_distance_individual_journey`)

> Your pirate ship presents structural limitations: it is able to sail the sea only only up to a certain distance during the same island-to-island journey. The weather and the strenght of the currents that rule the *Short Route* make the autonomy span of the ships very limited. When you sail from an island you can reach only islands that are within this journey distance, or the ship will definetely sink. 

When this constraint is active, the L1-norm length of every island-to-island journey to move among two islands (so the L1-norm distance between the two islands) must be at maximum `max_L1_distance_individual_journey`. The L1-norm distance to go from one island to the next one can be inferred from the their *x* and *y* positions.

---

### **Cost functions**

> You want to conquer the *Two Slices*, the legendary pirate treasure that lies on the last island of the *Short Route*. It is a complicated voyage in itself, but that's not all! This is a gold age for piracy, and many other pirate ships are looking for the treasure. You don't only have to plan a voyage that will not sink your ship or get you lost (taking care of the magnetic fields, the crew, the weather...) but that is also optimized for your personal needs and priorities.

The input of your function is a `ProblemVoyage` structure, in which you can find information about which cost (one cost only for each test) you should optimize (while enforcing the active constraints).


#### **Minimum nights to complete the voyage** (`min_total_nights`)

> The easiest way to seize the treasure: reach it as soon as possible, before every other pirate ship. Show to the other pirates the meaning of *haste*.

Plan a voyage such that the total number of nights spent is minimized.

#### **Maximum final crew size** (`max_final_crew`)

> You believe in the power of muscles and gunpowder: better reach the treasure with a large crew, to attack and win over other pirates who could reach the tresure before you and to defend it afterwards.

Plan a voyage such that the final crew size is maximized.

#### **Minimize total sailing time** (`min_total_sailing_time`)

> You, a captain of a pirate ship? Puah! Since you are a seasick wimp, you want to spend as little time as possible on a ship in the middle of the sea under the effect of waves and currents.

Plan a voyage such that the total sailing time, i.e. the total number of hours spent in the sea during the multiple island-tosialnd journeys, is minimized.

#### **Minimize total L1-norm travelled distance** (`min_total_travelled_L1_distance`)

> Your cousin's friend says that the best way to reach the tresure before everyone else is to travel the shortest total distance possible.

Plan a voyage such that the total L1-norm travelled distance, i.e. the sum of the L1-norm distances of the multiple island-to-island journeys, is minimized.

#### **Minimize the maximum individual sailing time** (`min_max_sailing_time`)

> In these dangerous waters, caution is never too much. Better not to sail the ship for too many hours in a row, close to its limit, to avoid unnecessary wearing.

Plan a voyage such that the maximum individual sailing time of the voyage, i.e. the maximum number of hours spent in a single island-to-island journey, is minimized.

---

## Data structures of the exercise

The various data structures needed for the development of the exercise can be inspected in [src/pdm4ar/exercises_def/ex07/structures.py](../src/pdm4ar/exercises_def/ex07/structures.py). 

### **Island**

Structure storing the individual features of an island.

<details>
<summary><b>Detailed description</b></summary>

- The `id` integer attribute identifies uniquely the island. The island of the first archipelago has always `id` = *0* while the island of the last archipelago has always `id` = *number of islands - 1*. If the archipelagos in between have 5 islands each, then the `id` of the islands of the second archipelago ranges from *1* to *5*, the ones of the third archipelagos from *6* to *10*, the ones of the fourth archipelago from *11* to *15*, and so on. When you submit your optimized voyage plan, you are submitting an ordered list of the `id` of the islands you plan to visit.
- The `arch` integer attribute tells you to which of the *N* archipelagos the island belongs (*0*, ..., *N-1*).
- The `x` and `y` float attributes specify the *x* and *y* position of the island in a cartesian reference system. The 2D map is a flat plane.
- The `departure` and `arrival` float attributes are a timetable of the exact time you have to depart from or to arrive to the island, to exploit its specific weather to being able to set sail or to dock. Note that to keep things simple the decimal places are not representing the minutes in *mod* 60. A value of 8.43 doesn't mean 43 minutes past 8, but that it's 43% of an hour past 8. Treat it as a normal float value.
To keep things simple, the arrival times of all the islands are later than the departure times of all the islands. This means in every possible journey between two island you depart and arrive later on the same day, always.
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

Structure storing the data of an optimization problem. Input of the function `solve_optimization`.

<details>
<summary><b>Detailed description</b></summary>

- The `optimization_cost` CostType attribute declares the cost you have to optimize.
- The `start_crew` integer attribute specifies how many people are in the crew (including the captain) at the beginning of the voyage.
- The `islands` attribute is a tuple containing a sequence of `Island`. The islands are ordered based on their `id` attribute.
- The `constraints` attribute contains the following:
    - The `min_nights_individual_island` integer attribute is a constraint specifing the minimum amount of nights you have to spend in every island to get the ship fixed before departing again to a new island. The ocean currents are badly damaging the ship every time you set sail.
    - The `max_total_crew` integer attributes specify the minimum amount of people who can be in the crew at the same time.
    - The `min_total_crew` integer attributes specify the maximum amount of people who can be in the crew at the same time.
    - The `max_duration_individual_journey` float attribute is a constraint specifing the maximum amount of hours each island-to-island jounrey can last. Treat it as a normal float value.
    - The `max_L1_distance_individual_journey` float attribute is a constraint specifing the maximum L1-norm distance length of each island-to-island journey.

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

### **SolutionVoyage**

Structure storing the solution of an optimization problem. Output of the function `solve_optimization`. A solution not compliant with the expected structure types will raise a **TestCaseSanityCheckException**.

<details>
<summary><b>Detailed description</b></summary>

- The `feasibility` FeasibilityType attribute specifies if the problem is found unfeasible or feasible. 
- The `voyage_plan` VoyagePlan attribute stores the list of the `id`s of the island in the order you plan to visit them if the problem is feasible, else it should be set to **None**.

</details>

```python
@dataclass(frozen=True)
class SolutionVoyage:
    feasibility: FeasibilityType
    voyage_plan: Optional[VoyagePlan]
```

with 

```python3
FeasibilityType = Literal[Feasibility.feasible, Feasibility.unfeasible]

VoyagePlan = List[int]
```

## Test cases and performance criteria

#### **Test cases**

We provide some example test cases with ground truth to test the feasibility, the constraint violations and the cost optimizations of your solution. The provided test cases are not the same as the ones run on the test server used for grading, we advise you to additionally test your implementation using your own defined test cases, e.g., by modifying the random test case generation in [src/pdm4ar/exercises_def/ex07/data.py](../src/pdm4ar/exercises_def/ex07/data.py), and then setting `test_type = CaseVoyage.random` within the function `get_exercise7` in [src/pdm4ar/exercises_def/ex07/ex07.py](../src/pdm4ar/exercises_def/ex07/ex07.py) to load your test cases and not the provided ones. 

Note that the constraints have different difficulties, but if you do not implement or violate one constraint, this will not affect the performance score of the other constraints (except if you violate `voyage_order` or if you mistakenly state that the problem is *unfeasible*). However, not implementing/violating a constraint can degrade the performance score of the feasibility and of the optimization costs.
Therefore, the test server will also test your code on a number of problems with the minimum number of active constraints at the same time, so as not to penalize the other performances score too much if you have not implemented all constraints correctly.

---

#### **Feasibility performance**

For the feasibility performance, we use an **accuracy** metric which we compute by counting the number of *correctly* computed test cases divided by the total number of test cases: $\frac{N_{correct,i}}{N_{task,i}}$. A test case is *correctly* computed if you match the ground truth feasibility status of the problem. If by any chance the ground truth status is *unfeasible* but your status is *feasible* and your solution is not violating any constraints, the test case is considered *correctly* computed.

#### **Constraints performance**

For the constraints performance, we use multiple **accuracy** metrics, one for each constraint, which we compute by counting the number of test cases where the specific constraint was *correctly* enforced divided by the total number of test cases where that constraint was active: $\frac{N_{correct,i}}{N_{task,i}}$. A constraint is *correctly* enforced if it is not violated up to some numerical tolerance, or if you correctly state the status of a ground truth *unfeasible* problem. Violating the `voyage_order` constraint counts as a violation also for all of the other active constraints.

#### **Costs performance**

For the costs scores, we use multiple **accuracy** metrics, one for each cost, which we compute by summing the scores of the test cases where the specific cost was *correctly* optimized divided by the total number of feasible test cases where that cost should have been optimized: $\frac{\sum score_{i}}{N_{task,i}}$. The score is 0 if you violate any constraint. If you don't violate any constraint, the score is 0 if the cost of your solution is worse than the ground truth optimal cost by more than **tol** = *max(5% of the ground truth optimal cost, min_abs_tol)*, linearly intepolated from 0 to 1 if it is within **tol**, and 1 if it matches it up to some numerical tolerance. The score is also 1 if by any chance your solution is more optimal than our ground truth, given that it is not violating any active constraint. Mistaking a ground truth *feasible* problem as *unfeasible* is scored with a 0.


## Report

The visualization of the exercise is handled within [src/pdm4ar/exercises_def/ex07/visualization.py](../src/pdm4ar/exercises_def/ex07/visualization.py).

- We provide five different levels of report types (solely for your own debugging): 
    - `ReportType.none`: no evalutation information at all, no report is generated.
    - `ReportType.terminal`: print evaluation information on the terminal, no report is generated.
    - `ReportType.report_txt`: print evaluation information on the terminal and in a textual report.
    - `ReportType.report_viz`: print evaluation information on the terminal and in a visual report, with text and figures.
    - `ReportType.report_viz_extra`: print evaluation information on the terminal and in a visual report, with text and figures and data of each island.

- You can choose your preferred terminal/report output setting the ***`REPORT_TYPE`*** global variable.
Note that depending on the number of islands, `report_viz` can be really slow, while `report_viz_extra` will be even slower.

- The speed of the report generation is also greatly influenced by the size of the images generated with `report_viz` and `report_viz_extra`. The image size (due to the map size) unfortunately affects also the non-overlapping of the figures, and in case you selected `report_viz_extra`, the readability of the extra text.
    You can choose your preferred size setting the ***`FIGURE_WIDTH`*** global variable (note that this specifies the size in *points*, not *pixels*. Think of it as if you are choosing the [DPI](https://en.wikipedia.org/wiki/Dots_per_inch), while the actual pixel size depends on the map size). Bigger the value, the slower the images generation but the better the readability.

- If your terminal is not correctly printing the colors (very improbable), or if you want/need better contrastive readability in both the terminal and the report, set the ***`ACTIVATE_COLORS`*** global variable to `False`.

Remember that the images shown in the pdf report are "compressed" (*rasterized*): to see the "uncompressed" (*vector graphic*) version click the **main** link below each image.

Feel free to make your own modifications to the visualization to match your debugging needs.

## Run the exercise
If needed, update your repository with `make update`, and/or install the *pdm4ar* module with `pip3 install -e [path/to/exercises_repo]`.

Run the exercise with:
```shell
python3 [path/to/]src/pdm4ar/main.py --exercise 07
```
or:
```shell
python3 [path/to/]src/pdm4ar/main.py -e 07
```

After running the exercise, a report will be generated in the folder `out/ex07` that shows your results (if you enabled the report generation).


## Hints
- Since the islands stored in the `islands` tuple of `ProblemVoyage` are ordered based on their `id` and since each archipelago has the same amount of islands (apart from the first and the last one), you can use a smart indexing to access islands of the same archipelago.
- When working with distances among islands, consider the islands as dimensionless points.
- You might want to model the problem as a Mixed Integer Linear Program.
- To model the problem notice that in the environment there are already powerful libraries to solve optimization problems. For instance, [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html), [PuLP](https://coin-or.github.io/pulp/), [Google OR-Tools](https://developers.google.com/optimization/introduction/overview), and [cvxpy](https://www.cvxpy.org/) (we tested *scipy.optimize* and *PuLP*). To be able to use *PuLP*, *Google OR-Tools* or *cvxpy*, please rebuild the container, running the VS Code command (click Ctrl+Shift+P) `Remote-Containers: Rebuild Container` or `Remote-Containers: Rebuild and Reopen in Container`.

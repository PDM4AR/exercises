# Optimization :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>


## Problem overview

In this exercise you will learn how to solve an optimization problem translating concepts from the textual to the mathematical and programming realm.

The environment of the problem is a 2D map composed of islands each with its own features. The islands are divided into *N* different groups, which will be called archipelagos henceforth, identified with a number from *0* to *N-1*. Archipelagos n. *0* (first archipelago) and n. *N-1* (last archipelago) are composed of one island each. All of the other archipelagos (from n. *1* to n. *N-2*) are composed of the same amount *k* of islands.
Hence, the total number of islands is 1+(*N-2*)\**k* +1 = (*N-2*)\**k* +2. The islands are identified by a unique id, ranging from *0* to (*N-2*)*k*+1. They also have another tag to discern to which archipelagos they belong. Note that the belonging of an island in a specific group, called archipelago, is not determined by position, topographic reasons, similarity, etc.: don't make any assumption, just take it as it is.

Your job is to compute a voyage plan (an ordered list of islands to be visited) that starts from the first archipelago and ends at the last archipelago, optimized for a specific cost while satifying some constraints.

Here you can see two examples of correct planned voyages:

![example 1](https://user-images.githubusercontent.com/79461707/193420646-368a6b22-6271-420b-bbec-6afe73f6bb68.png)
![example 2](https://user-images.githubusercontent.com/79461707/193420649-e604125d-4781-4058-b17f-376d60ba687e.png)

You have to implement the optimization inside the function `solve_optimization` in [src/pdm4ar/exercises/ex07/ex07.py](../src/pdm4ar/exercises/ex07/ex07.py). The input you receive is a `ProblemVoyage` structure, and you have to output back a `ProblemSolution` structure. Through the `ProblemVoyage` input you have access to the different active constraints and the specific cost your voyage plan must satisfy and optimize.

```python
def solve_optimization(problem: ProblemVoyage) -> ProblemSolution:
    """
    Solve the optimization problem enforcing the active constraints

    Parameters
    ---
    problem : ProblemVoyage
        Contains the problem data: cost to optimize, starting crew, tuple of islands, 
        and information about active constraints.

    Returns
    ---
    out : ProblemSolution
        Contains the feasibility status of the problem, and the optimal voyage plan
        as a list of ints if problem is feasible, else None
    """

    # toy examples with random voyage plans
    np.random.seed(None)
    if np.random.random() > 0.3:
        feasibility = MilpFeasibility.feasible
        voyage_plan = list(np.random.randint(0,len(problem.islands), size=(min(7,len(problem.islands)),)))
    else:
        feasibility = MilpFeasibility.unfeasible
        voyage_plan = None

    return ProblemSolution(feasibility, voyage_plan)
```

---

### **Constraints**

The input of your function is a `ProblemVoyage` structure, in which you can find information about which constraints you should enforce. When a costraint has a **None** value, it means you don't have to take care of it. Multiple constraints can be active at the same time.


#### **Voyage order** (`voyage_order`)

This constraint is always active. Your voyage plan must start from the first archipelago and end at the last archipelago, visiting one and only one island of the other archipelagos in between, following the order of the archipelago's identification number. This means that your ordered voyage plan must contain *N* islands and that these islands must belong to archipelagos with increasing identification numbers. Since the constraint is active in every test case, it is not listed within the `constraints` attribute of the `ProblemVoyage` input, to avoid reduntant information.

#### **Minimum nights** (`min_nights_individual_island`)

When this constraint is active, you can only visit island whose number of waiting nights is at least `min_nights_individual_island`. When you arrive in an island you have to wait a specific amount of nights before departing towards the next island. Each island has its specific value of how many night you should stay in it before departing. Only the starting island an the ending island have a value of 0.

#### **Minimum crew size** (`min_total_crew`)

When this constraint is active, the crew size cannot be smaller than `min_total_crew`, whichever island you are in during any moment of your voyage.

#### **Maximum crew size** (`max_total_crew`)

When this constraint is active, the crew size cannot be greater than `max_total_crew`, whichever island you are in during any moment of your voyage.

#### **Maximum duration individual journey** (`max_duration_individual_journey`)

When this constraint is active, every island-to-island journey to move among two islands must last at maximum `max_duration_individual_journey` hours. The time needed to go from one island to the next one can be inferred from the respective departure and arrival timetable.

#### **Maximum L1-norm distance individual journey** (`max_L1_distance_individual_journey`)

When this constraint is active, the L1-norm length of every island-to-island journey to move among two islands (so the L1-norm distance between the two islands) must be at maximum `max_L1_distance_individual_journey`. The L1-norm distance to go from one island to the next one can be inferred from the their *x* and *y* positions.

---

### **Cost functions**

The input of your function is a `ProblemVoyage` structure, in which you can find information about which cost (one cost only for each test) you should optimize (while enforcing the active constraints).


#### **Minimum nights to complete the voyage** (`min_total_nights`)

Plan a voyage such that the total number of nights spent is minimized.

#### **Maximum final crew size** (`max_final_crew`)

Plan a voyage such that the final crew size is maximized.

#### **Minimize total sailing time** (`min_total_sailing_time`)

Plan a voyage such that the total sailing time, i.e. the total number of hours spent in the sea during the multiple island-tosialnd journeys, is minimized.

#### **Minimize total L1-norm travelled distance** (`min_total_travelled_L1_distance`)

Plan a voyage such that the total L1-norm travelled distance, i.e. the sum of the L1-norm distances of the multiple island-to-island journeys, is minimized.

#### **Minimize the maximum individual sailing time** (`min_max_sailing_time`)

Plan a voyage such that the maximum individual sailing time of the voyage, i.e. the maximum number of hours spent in a single island-to-island journey, is minimized.

---

## Data structures of the exercise

The various data structures needed for the development of the exercise can be inspected in [src/pdm4ar/exercises_def/ex07/structures.py](../src/pdm4ar/exercises_def/ex07/structures.py). 

### **Island**

Used to store the individual features of an island.

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

Input of the function `solve_milp` you have to implement.

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

### **ProblemSolution**

Used to store the solution of the optimization problem.

<details>
<summary><b>Detailed description</b></summary>

- The `feasibility` FeasibilityType attribute specifies if the problem is found unfeasible or feasible. 
- The `voyage_plan` VoyagePlan attribute stores the list of the `id`s of the island in the order you plan to visit them if the problem is feasible, else it should be set to **None**.

</details>

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

## Test cases and performance criteria

#### **Test cases**

We provide some example test cases with ground truth to test the feasibility, the constraint violations and the cost optimizations of your solution. The provided test cases are not the same as the ones run on the test server used for grading, we advise you to additionally test your implementation using your own defined test cases, e.g., by modifying the random test case generation in [src/pdm4ar/exercises_def/ex07/data.py](../src/pdm4ar/exercises_def/ex07/data.py), and then setting `test_type = MilpCase.random_voyage` within the function `get_exercise7` in [src/pdm4ar/exercises_def/ex07/ex07.py](../src/pdm4ar/exercises_def/ex07/ex07.py) to load your test cases and not the provided ones. 

Note that the constraints have different difficulties, but if you do not implement or violate one constraint, this will not affect the performance score of the other constraints (except if you violate `voyage_order` or if you mistakenly state that the problem is *unfeasible*). However, not implementing/violating a constraint can degrade the performance score of the feasibility and of the optimization costs.
Therefore, the test server will also test your code on a number of problems with the minimum number of active constraints at the same time, so as not to penalize the other performances score too much if you have not implemented all constraints correctly.

---

#### **Feasibility performance**

For the feasibility performance, we use an **accuracy** metric which we compute by counting the number of *correctly* computed test cases divided by the total number of test cases: $\frac{N_{correct,i}}{N_{task,i}}$. A test case is *correctly* computed if you match the ground truth feasibility status of the problem. If by any chance the ground truth status is *unfeasible* but your status is *feasible* and your solution is not violating any constraints, the test case is considered *correctly* computed.

#### **Constraints performance**

For the constraints performance, we use multiple **accuracy** metrics, one for each constraint, which we compute by counting the number of test cases where the specific constraint was *correctly* enforced divided by the total number of test cases where that constraint was active: $\frac{N_{correct,i}}{N_{task,i}}$. A constraint is *correctly* enforced if it is not violated up to some numerical tolerance, or if you correctly state the status of a ground truth *unfeasible* problem. Violating the `voyage_order` constraint counts as a violation also for all of the other active constraints. Mistaking the fesibility status of the problem counts as a violation.

#### **Costs performance**

For the costs scores, we use multiple **accuracy** metrics, one for each cost, which we compute by summing the scores of the test cases where the specific cost was *correctly* optimized divided by the total number of feasible test cases where that cost should have been optimized: $\frac{\sum score_{i}}{N_{task,i}}$. The score is 0 if you violate any constraint. If you don't violate any constraint, the score is 0 if the cost of your solution is worse than the ground truth optimal cost by more than **tol** = *max(5% of the ground truth optimal cost, 2)*, linearly intepolated from 0 to 1 if it is within **tol**, and 1 if it matches it up to some numerical tolerance. The score is also 1 if by any chance your solution is more optimal than our ground truth, given that it is not violating any active constraint. Stating that a ground truth *feasible* problem is *unfeasible* is scored with a 0.


## Report

The visualization of the exercise is handled within [src/pdm4ar/exercises_def/ex07/visualization.py](../src/pdm4ar/exercises_def/ex07/visualization.py).

- We provide five different levels of report types (solely for your own debugging): 
    - `ReportType.none`: no evalutation information at all, no report is generated.
    - `ReportType.terminal`: print evaluation information on the terminal, no report is generated.
    - `ReportType.report_txt`: print evaluation information on the terminal and in a textual report.
    - `ReportType.report_viz`: print evaluation information on the terminal and in a visual report, with text and figures.
    - `ReportType.report_viz_extra`: print evaluation information on the terminal and in a visual report, with text and figures and data of each island.

- You can choose your preferred terminal/report output setting the ***`REPORT_TYPE`*** global variable.
Note that depending on the number of islands, `report_viz` starts to be slower by a non-negligible amount of time, while `report_viz_extra` will be even slower.

- The speed of the report generation is also greatly influenced by the size of the images generated with `report_viz` and `report_viz_extra`. The image size (due to the map size) unfortunately affects also the non-overlapping of the figures, and in case you selected `report_viz_extra`, the readability of the extra text.
    You can choose your preferred size setting the ***`FIGURE_WIDTH`*** global variable (note that this specifies the size in *points*, not *pixels*. Think of it as if you are choosing the [DPI](https://en.wikipedia.org/wiki/Dots_per_inch), while the actual pixel size depends on the map size). Bigger the value, the slower the images generation but the better the readability.

- If your terminal is not correctly printing the colors (very improbable), or if you want/need better contrastive readability in both the terminal and the report, set the ***`ACTIVATE_COLORS`*** global variable to `False`.

Feel free to make your own modifications to the visualization to match your debugging needs.

## Run the exercise

```shell
pip3 install -e [path/to/exercises_repo]
python3 [path/to/]src/pdm4ar/main.py --exercise 07
```

After running the exercise, a report will be generated in the folder `out/ex07` that shows your results (if you enabled the report generation).


## Hints
- To model the problem notice that in the environment there are already powerful libraries to solve optimization problems. For instance, [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) and [pulp](https://coin-or.github.io/pulp/).
- Since the islands stored in the `islands` tuple of `ProblemVoyage` are ordered based on their `id` and since each archipelago has the same amount of islands (apart from the first and the last one), you can use a smart indexing to access islands of the same archipelago.
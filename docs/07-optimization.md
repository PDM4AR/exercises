# Optimization :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-helloworld.html" target="_top">Hello-world</a></td>
  </tr>
</table>


In the following instructions, the narrative text in [blockquotes](https://www.markdownguide.org/basic-syntax/#blockquotes-1) is irrelevant to the actual task.
Feel free to skip them.

It is recommended to read the instructions directly from the `docs/07-optimization.md` file on your github repo or VS Code (click Ctrl+Shift+V), since *github.io* is not correctly loading the "Detailed description" dropdown.


## Problem overview

> You are the captain of a pirate ship, and you desire to travel through the *Short Route*, a misleading name for a dangerous area of the oceans filled with islands, to attempt to reach the *Two Slices*, the legendary pirate treasure which lies on the last island of the *Short Route*.

You plan your voyage on a 2D map where all the islands and the corresponding features are annotated. 
The islands are divided into *N* different archipelagos identified with a number from *0* to *N-1*.
The first archipelago (#*0*) and the last one (#*N-1*) are each formed by a single island. 
All the other archipelagos (from #*1* to #*N-2*) are instead composed by *k* islands each.
Hence, the total number of islands is $(N-2)k + 2$.
The islands are identified by a unique id, ranging from 0 to $(N-2)k + 1$. 
Moreover, they another tag indicates to which archipelagos they belong. 
Note that the belonging of an island in a specific group, called archipelago, is not determined by position, topographic reasons, similarity, etc.: don't make any assumption.

Your task is to compute a voyage plan as an ordered list of islands to be visited. 
The plan shall optimize a given cost while satisfying different constraints (described below).
Common to all the plans is that they start from the first archipelago and end at the last archipelago, visiting one and only one island of the other archipelagos following the order of the archipelago's identification number.

Here you can see two examples of such planned voyages:

<p align="center">
  <img alt="img-name" src="https://user-images.githubusercontent.com/79461707/199729360-26647058-7399-46fc-a4aa-aa2353171643.jpeg">
  <br>
    <em>voyage plan: 0 -> 1 -> 8 -> 12 -> 13</em>
</p>

<p align="center">
  <img alt="img-name" src="https://user-images.githubusercontent.com/79461707/199729335-40cfaca9-2256-498f-bd4e-163cb8429dba.jpeg">
  <br>
    <em>voyage plan: 0 -> 15 -> 21 -> 38 -> 67 -> 88 -> 91 -> 117 -> 130 -> 151 -> 165 -> 196 -> 203 -> 224 -> 235</em>
</p>

---

Your task is to implement the function `solve_optimization` in [src/pdm4ar/exercises/ex07/ex07.py](../src/pdm4ar/exercises/ex07/ex07.py) (please remember that your modifications done outside the *src/pdm4ar/exercises/ex??*  folders are discarded during the server evaluation). 
As input, you receive a `ProblemVoyage`, and you need to return a `SolutionVoyage`. 
Through the `ProblemVoyage` structure you have access to the different constraints and the specific cost that your voyage plan must optimize.

```python
def solve_optimization(problem: ProblemVoyage) -> SolutionVoyage:
    """
    Solve the optimization problem enforcing the constraints

    Parameters
    ---
    problem : ProblemVoyage
        Contains the problem data: cost to optimize, starting crew, tuple of islands, 
        and information about the requested constraints.

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

## Plan Optimization

### **Cost functions**

> You want to conquer the *Two Slices*, the legendary pirate treasure that lies on the last island of the *Short Route*. It is a complicated voyage in itself, but that's not all! This is a gold age for piracy, and many other pirate ships are looking for the treasure. You don't only have to plan a voyage that will not sink your ship or get you lost (taking care of the magnetic fields, the crew, the weather...) but that it is also optimized for your personal needs and priorities.

In the `ProblemVoyage` structure you can find information about which cost you should optimize. Note that *only one cost for each test* will be given.
The possible costs to optimize for will be:

#### **Minimum nights to complete the voyage** (`min_total_nights`)

> The easiest way to seize the treasure: reach it as soon as possible, before every other pirate ship. Show to the other pirates the meaning of *haste*.

Plan a voyage such that the total number of nights spent is minimized.

#### **Maximum final crew size** (`max_final_crew`)

> You believe in the power of muscles and gunpowder: better reach the treasure with a large crew, to attack and win over other pirates who could reach the treasure before you and to defend it afterwards.

Plan a voyage such that the final crew size is maximized.

#### **Minimize total sailing time** (`min_total_sailing_time`)

> You, a captain of a pirate ship? Puah! Since you are a seasick wimp, you want to spend as little time as possible on a ship in the middle of the sea under the effect of waves and currents.

Plan a voyage such that the total sailing time, i.e. the total number of hours spent in the sea during the multiple island-to-island journeys, is minimized.

#### **Minimize total L1-norm travelled distance** (`min_total_travelled_L1_distance`)

> Your cousin's friend says that the best way to reach the treasure before everyone else is to travel the shortest total distance possible.

Plan a voyage such that the total L1-norm travelled distance, i.e. the sum of the L1-norm distances of the multiple island-to-island journeys, is minimized.

#### **Minimize the maximum individual sailing time** (`min_max_sailing_time`)

> In these dangerous waters, caution is never too much. Better not to sail the ship for too many hours in a row, close to its limit, to avoid unnecessary wearing.

Plan a voyage such that the maximum individual sailing time of the voyage, i.e. the maximum number of hours spent in a single island-to-island journey, is minimized.

---

### **Constraints**

> Due to the rare magnetic fields, the size of the ship, the distances, and the weather, you will experience different combinations of constraints during your adventure. You should plan your voyage accordingly, because the *Short Route* does not forgive incautious pirates.

In the `ProblemVoyage` structure you will also find information about the constraints that you should enforce.
Note that multiple constraints can be requested for a single test case.
When a constraint has a **None** value, it means you can disregard it for that test case.


#### **Voyage order** (`voyage_order`)

> The *Short Route* presents a weird and dangerous magnetic field. Travelling through it is not possible using a normal compass, but special compasses are needed. These special compasses show the direction of the islands of the next archipelago: when you are at a specific island of an archipelago, they will tune with the magnetic field of the next archipelago, and so on, until you reach the last archipelago where the treasure is located. Trying to directly reach the last archipelago or moving to an archipelago that is not the next in order is impossible, and you will get lost in the ocean.

This constraint is always present. Your voyage plan must start from the first archipelago and end at the last archipelago, visiting one and only one island of the other archipelagos, following the order of the archipelago's identification number. This means that your ordered voyage plan must contain *N* islands and that these islands must belong to archipelagos with increasing identification numbers. Since the constraint is present in every test case, it is not listed within the `constraints` attribute of the `ProblemVoyage` input, to avoid redundant information.

#### **Minimum nights** (`min_nights_individual_island`)

> Everytime you land on a new island, the special compasses should start to tune to the new magnetic fields pointing to the islands of the next archipelagos. The number of nights to tune the special compasses varies depending on the island. Unfortunately, the faster the tuning time of a magnetic field, the more technologically complex a special compass should be to be able to tune it. Hence, you are only able to visit islands where the magnetic field pointing to the next archipelago is tunable by your special compass version. You cannot visit islands where the magnetic field requires fewer nights to tune than the minimum amount of nights specified by your compass. You cannot simply wait more time: if your compass is able to tune to magnetic field that requires 4 nights of waiting, you cannot visit an island with a 1, 2, or 3 nights magnetic field tuning and just wait for the remaining nights. The compass is simply not working with these faster tuning magnetic field.

When this constraint is present, you can only visit islands whose number of waiting *nights* is at least `min_nights_individual_island`. This constraint is not applied to the islands of the first and last archipelago.

#### **Minimum crew size** (`min_total_crew`)

> Pirate ships can be as small as life rafts or as big as vessels: since you are the captain, its dimensions depends on your wealth and ostentatiousness. Based on its size, the ship will always require a crew with a specific minimum amount of people (you included) to steer it and take care of it. 

When this constraint is present, the crew size cannot be smaller than `min_total_crew`, whichever island you are in during any moment of your voyage.

#### **Maximum crew size** (`max_total_crew`)

> Pirate ships can be as small as life rafts or as big as vessels: since you are the captain, its dimensions depends on your wealth and ostentatiousness. Based on its size, the ship will always allow a crew with a specific maximum amount of people (you included), since there is no room for everyone.

When this constraint is present, the crew size cannot be greater than `max_total_crew`, whichever island you are in during any moment of your voyage.

#### **Maximum duration individual journey** (`max_duration_individual_journey`)

> Your pirate ship presents structural limitations: it is able to sail the sea only up to a certain amount of hours during the same island-to-island journey. The weather and the strength of the currents that rule the *Short Route* make the autonomy span of the ships very limited. When you sail from an island you can reach only islands that are within this journey duration, or the ship will definitely sink. 

When this constraint is present, every island-to-island journey to move among two islands must last at maximum `max_duration_individual_journey` hours. The time needed to go from one island to the next one can be inferred from the respective *departure* and *arrival* timetable.

#### **Maximum L1-norm distance individual journey** (`max_L1_distance_individual_journey`)

> Your pirate ship presents structural limitations: it is able to sail the sea only up to a certain distance during the same island-to-island journey. The weather and the strength of the currents that rule the *Short Route* make the autonomy span of the ships very limited. When you sail from an island you can reach only islands that are within this journey distance, or the ship will definitely sink. 

When this constraint is present, the L1-norm length of every island-to-island journey to move among two islands (so the L1-norm distance between the two islands) must be at maximum `max_L1_distance_individual_journey`. 
The L1-norm distance to go from one island to the next one can be inferred from the *x* and *y* positions.

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
- The `nights` integer attribute specifies the exact amount of nights you have to spend on the island before you can depart to the next island. If `nights` is 1, it means you land on the island, and you depart the next day, irrelevant of the arrival/departure timetable. The islands of the first and last archipelagos have `nights` = 0.
- The `delta_crew` integer attribute specifies how many people will leave the crew (negative value) or how many join the crew (positive value) if you visit the island. The islands of the first and last archipelago have `delta_crew` = 0.

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
    - The `min_nights_individual_island` integer attribute is a constraint specifying the minimum amount of `nights` an island should have to be able to visit it.
    - The `min_total_crew` integer attributes specify the maximum amount of people who can be in the crew at the same time.
    - The `max_total_crew` integer attributes specify the minimum amount of people who can be in the crew at the same time.
    - The `max_duration_individual_journey` float attribute is a constraint specifying the maximum amount of hours each island-to-island journey can last. Treat it as a normal float value.
    - The `max_L1_distance_individual_journey` float attribute is a constraint specifying the maximum L1-norm distance length of each island-to-island journey.

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

## Available Optimization Tools 
To model the problem, note that we have added powerful libraries in the container to solve optimization problems ([rebuild the container to use them](#run-the-exercise)). 
For instance, [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html), [PuLP](https://coin-or.github.io/pulp/), [cvxpy](https://www.cvxpy.org/), and [Google OR-Tools](https://developers.google.com/optimization/introduction/overview). We tested *scipy.optimize* and *PuLP* (and *cvxpy*). While *scipy.optimize* and *PuLP* will work without any additional extra steps, *cvxpy* requires you to set the **CBC** solver when calling [solve()](https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.solve):  `prob.solve(solver="CBC")`.

The final goal is to find an optimal feasible solution, but you are free to choose how to solve the problem, how to model it (i.e. modeling constraints and costs) and which library to exploit, among those in the container.

## Test cases and performance criteria

#### **Test cases**

We provide some example test cases with a solution to test the feasibility, the constraint violations and the cost optimizations of your solution. The provided test cases are not the same as the ones run on the test server used for grading, we advise you to additionally test your implementation using your own defined test cases, e.g., by modifying the random test case generation in [src/pdm4ar/exercises_def/ex07/data.py](../src/pdm4ar/exercises_def/ex07/data.py), and then setting `test_type = CaseVoyage.random` within the function `get_exercise7` in [src/pdm4ar/exercises_def/ex07/ex07.py](../src/pdm4ar/exercises_def/ex07/ex07.py) to load your test cases and not the provided ones. 

Note that the constraints have different difficulties, but if you do not implement or violate one constraint, this will not affect the performance score of the other constraints (except if you violate `voyage_order` or if you mistakenly state that the problem is *unfeasible*). However, not implementing/violating a constraint can degrade the performance score of the feasibility and of the optimization costs.
Therefore, the test server will also test your code on a number of problems with the minimum number of present constraints at the same time, so as not to penalize too much the other performance scores if you have not implemented all the constraints correctly.

---

#### **Feasibility performance**

For the feasibility performance, we use an (*1* ) **accuracy** metric which we compute by counting the number of *correctly* computed test cases divided by the total number of test cases: $\frac{N_{correct}}{N_{total}}$. 
A test case is *correctly* computed if you match our solution feasibility status of the problem. 
If the solution status is *unfeasible*, but you state it is *feasible* while your solution is not violating any constraints, the test case is considered *correctly* computed (congrats, you found a solution better than ours!).

#### **Constraint performances**

For the constraint performances, we use multiple (*6* ) **accuracy** metrics, one for each constraint, which we compute by counting the number of test cases where the specific constraint was *correctly* enforced divided by the total number of test cases where that constraint was present: $\frac{N_{correct,i}}{N_{total,i}}$. 
A constraint is *correctly* enforced if it is not violated up to some numerical tolerance, or if you correctly state the status of an *unfeasible* problem. 
Violating the `voyage_order` constraint counts as a violation also for all the other present constraints.

#### **Cost performances**

For the cost performances, we use multiple (*5* ) **accuracy** metrics, one for each cost, which we compute by calculating the average from the individual cost scores of the *feasible* test cases where that cost should be optimized: $\frac{\sum score_{i}}{N_{total,i}}$.
The score is 0 if you violate any constraint. If you don't violate any constraint, the score is 0 if the cost of your solution is much worse than our solution's cost, namely by more than **tol** = *max(5% of our solution's cost, min_abs_tol)*, linearly interpolated from 0 to 1 if it is within **tol**, and 1 if it matches it up to some numerical tolerance. 
The score is also 1 if the cost of your solution is even better than our optimal cost, given that your solution is not violating any present constraint. 
Mistaking a *feasible* problem as *unfeasible* is scored with a 0.
Note that we are checking the cost of your feasible voyage plan (and not the voyage plan itself) since there is only one optimal cost, but that can be generated by different optimal solutions/voyage plans.

#### **Total performance**

Finally, the total performance of the exercise is a simple average of the previous 12 accuracy metric performances.

## Report

The visualization of the exercise is handled within [src/pdm4ar/exercises_def/ex07/visualization.py](../src/pdm4ar/exercises_def/ex07/visualization.py).

- We provide five different levels of report types (solely for your own debugging): 
    - `ReportType.none`: no evaluation information at all, no report is generated.
    - `ReportType.terminal`: print evaluation information on the terminal, no report is generated.
    - `ReportType.report_txt`: print evaluation information on the terminal and in a textual report.
    - `ReportType.report_viz`: print evaluation information on the terminal and in a visual report, with text and figures.
    - `ReportType.report_viz_extra`: print evaluation information on the terminal and in a visual report, with text and figures and data of each island.

- You can choose your preferred terminal/report output setting the ***`REPORT_TYPE`*** global variable.
Note that depending on the number of islands, `report_viz` can be really slow, while `report_viz_extra` will be even slower.

- The speed of the report generation is also greatly influenced by the size of the images generated with `report_viz` and `report_viz_extra`. The image size (due to the map size) unfortunately affects also the non-overlapping of the figures, and in case you selected `report_viz_extra`, the readability of the extra text.
    You can choose your preferred size setting the ***`FIGURE_WIDTH`*** global variable (note that this specifies the size in *points*, not *pixels*. Think of it as if you are choosing the [DPI](https://en.wikipedia.org/wiki/Dots_per_inch), while the actual pixel size depends on the map size). Bigger the value, the slower the images generation but the better the readability.

- If your report's settings (combined with the performance of your PC) produces a slow report generation, you will probably incur in a *Timeout Exception*: to avoid this, increase the `test_case_timeout` variable at the end of [src/pdm4ar/exercises_def/ex07/ex07.py](../src/pdm4ar/exercises_def/ex07/ex07.py) during your debugging - but remember that the official timeout is the original one. Server-side, the speed of the report generation was already taken into account when the timeout value was chosen.

- If your terminal is not correctly printing the colors (very improbable), or if you want/need better contrastive readability in both the terminal and the report, set the ***`ACTIVATE_COLORS`*** global variable to `False`.

Remember that the images shown in the pdf report are "compressed" (*rasterized*): to see the "uncompressed" (*vector graphic*) version click the **main** link below each image.

Feel free to change the provided report's settings and to make your own modifications to the visualization to match your debugging needs. But please remember that your modifications done outside the *src/pdm4ar/exercises/ex??*  folders are discarded during the server evaluation: your code within [src/pdm4ar/exercises/ex07](../src/pdm4ar/exercises/ex07) should be compatible with the provided [src/pdm4ar/exercises_def/ex07/visualization.py](../src/pdm4ar/exercises_def/ex07/visualization.py) file. 

## Run the exercise
Update your repository running `make update` (refer to [Hello World](01-helloworld.md) for more instructions).

To be able to import and use *PuLP*, *cvxpy*, or *Google OR-Tools*, please rebuild the container running the VS Code command (click Ctrl+Shift+P) `Remote-Containers: Rebuild Container` or `Remote-Containers: Rebuild and Reopen in Container`, and then reinstall the *pdm4ar* module running `pip3 install -e [path/to/exercises_repo]` in the VS Code terminal.

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
- You might want to add additional optimization variables to model some constraints and/or costs.
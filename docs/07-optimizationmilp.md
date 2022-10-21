# Optimization - Mixed Integer Linear Programming :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

##  Optimize your voyage to the *Two Slices*

You are the captain of a pirate ship attempting to travel through the *Short Route*, a misleading name for a dangerous area of the oceans, to reach the *Two Slices*, the legendary treasure which lies at the end of the *Short Route*.
The *Short Route* is an agglomerate of different archipelagos of islands, stretching horizontally from west to east.

The islands are divided into different archipelagos. Due to the distances, the weather, and rare magnetic fields, you can only travel from one archipelago to the next one;
you cannot directly set sail towards the last archipelago.
This is because to travel through the *Short Route* you cannot use a normal compass, but you need special compasses, which when you are in an archipelago, show the direction of the islands of the next archipelago. When you are at a specific island of an archipelago, the special compasses will tune with the magnetic field of the next archipelago, and so on, until you reach the last archipelago where the treasure is located.
Note that each archipelago has the same number of islands.
Here you see two examples of a planned trip:

![example 1](https://user-images.githubusercontent.com/79461707/193420646-368a6b22-6271-420b-bbec-6afe73f6bb68.png)
![example 2](https://user-images.githubusercontent.com/79461707/193420649-e604125d-4781-4058-b17f-376d60ba687e.png)


You start your voyage from the first archipelago of the *Short Route* (the leftmost). You need to find a voyage plan (an ordered list of islands to be visited) to reach the last archipelago (the rightmost) optimizing for some specific costs while satifying some constraints. For example, you can only visit one and only one island of each archipelago, you cannot travel among islands that are too far away from each other, and you need to optimize the duration of the whole voyage.


## Data structures of the exercise

You decide to solve the planning problem with a MILP formulation.
The task is to implement the `solve_milp` function inside [src/pdm4ar/exercises/optimization/milp/milp.py](../src/pdm4ar/exercises/optimization/milp/milp.py), which takes as input a `ProblemVoyage1` data structure and outputs a `ProblemSolutions`.

The various data structures needed for the development of the exercise can be inspected in [src/pdm4ar/exercises_def/optimization/milp/structures.py](../src/pdm4ar/exercises_def/optimization/milp/structures.py). 

---

### Island

<details>
<summary><b>Detailed description</b></summary>

- The `id` integer attribute identifies uniquely the islands. If each archipelago has 5 islands, then the `id` of the islands of the first archipelago ranges from 0 to 4, the ones of the second archipelagos from 5 to 9, the ones of the third archipelago from 10 to 14, and so on. When you will submit your optimized voyage plan, you will submit an ordered list of the `id` of the islands you planned to visit.
- The `arch` integer attribute tells you to which archipelago the island belongs (0, 1, 2...).
- The `x` and `y` float attribute specifies the x and y position of the island in a cartesian reference system. The *Short Route* can be approximated as a flat plane, so you don't have to consider the the Earth's curvature.
- The `departure` and `arrival` float attributes are a timetable of the exact time  you have to depart from or to arrive to the island, to exploit its specific weather to being able to set sail or to dock. Note that to keep things simple the decimal places of a number are not representing the minutes in mod 60. A value of 8.43 doesn't mean it's 43 minutes past 8, but that it's 43% of an hour past 8. Treat it a normal float value.
To keep things simple, the arrival times of all the islands are later than the departure times of all the islands. This means you are always departing around morning and arriving around the evening of the same day.
- The `time_compass` integer attribute specifies how many nights you have to spend on the island before you can depart to the next archipelagos. If `time_compass` is 1, it means you arrive in the island and you depart the next day, irrelevant of the hour you arrive or you depart. DEVELOPMENT NOTE: YOU SHOULD ACCOUNT FOR TIME_COMPASS VALUE OF THE LAST ISLAND YOU ARRIVE TOO. I could remove it, or keep it and add a single final island after the last archipelago at end of the voyage.
- The `crew` integer attribute specifies how many people leave the crew (negative value) or how many join the crew (positive value) if you visit the island.

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
    time_compass: int
    crew: int
```

---

### ProblemVoyage1

<details>
<summary><b>Detailed description</b></summary>

- The `start_crew` integer attribute specify how many people are in the crew (included the captain) at the beginning of the voyage.
- The `islands` attribute is a tuple containing the islands' data. Since the islands in the tuple ar eordered based on then`id` and since each archipelago has the same amount of islands, you can use a smart indexing to access islands of the same archipelago.
- The `min_fix_time_individual_island` integer attribute is a constraint specifing the minimum amount of nights you have to spend in every island to get the ship fixed before departing again to a new island. The ocean currents are badly damaging the ship every time you set sail.
- The `max_crew` and `min_crew` integer attributes specify the minimum and the maximum amount of people who can be in the crew at the same time. A small number of people are not adequate for the danger of the *Short Route*, and the ship is not big enough to host too many people.
- The `max_duration_individual_journey` float attribute is a constraint specifing the maximum amount of hours each voyage from one island to the next can last, otherwise the damage of the ship will be too much and it will sink.
- The `max_distance_individual_journey` float attribute is a constraint specifing the maximum amount of hours each voyage from one island to the next can last, otherwise the damage of the ship will be too much and it will sink.

</details>

```python
@dataclass(frozen=True)
class ProblemVoyage1:
    start_crew: int
    islands: Tuple[Island]
    min_fix_time_individual_island: int
    min_crew: int
    max_crew: int
    max_duration_individual_journey: float
    max_distance_individual_journey: float    
```

Input of the function `solve_milp` you have to implement.

---

### ProblemVoyage2

```python
@dataclass(frozen=True)
class ProblemVoyage2(ProblemVoyage1):
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float] 
```

Same as `ProblemVoyage1`, but with the additional `start_pos` and `end_pos` attributes. In this version of the problem, two extra islands are present: the start island and the end island, not belonging to any archipelaos. The voyage start from the start island, which is located before the first archipelagos, and ends at the last island, which is located after the last archipelagos.

---

### MilpFeasibility

```python
@dataclass(frozen=True)
class MilpFeasibility(Enum):
    feasible = "feasible"
    unfeasible = "unfeasible"
```

Used to indicate the `status` attribute of the `MilpSolution` data structure.

---

### MilpSolution

```python
@dataclass(frozen=True)
class MilpSolution:
    status: Literal[MilpFeasibility.feasible, MilpFeasibility.unfeasible]
    voyage_plan: Optional[List[int]] = None

```

Used to store the optimal solution of a MILP problem. The `status` attributes specifies if the MILP problem was found unfeasible or feasible, using the `MilpFeasibility` attribute values. If it is feasible, save in `voyage_plan` the `id`s of the island you plan to visit in order. If it is unfeasible, the content of `voyage_plan` doesn't matter.

---

### ProblemSolutions

<details>
<summary><b>Detailed description</b></summary>

- The `min_total_compass_time` attribute stores the `MilpSolution` of the voyage associated with the minimum total amount of nights waiting for the compasses to tune.
- The `max_final_crew` attribute stores the `MilpSolution` of the voyage associated with the maximum amount of people in the crew at the end of the voyage.
- The `min_total_sail_time` attribute stores the `MilpSolution` of the voyage associated with the minimum total amount of time actually spent sailing the ocean with the ship, i.e. the minimum total amount of time spent in the intermediate journeys to go from one island to the next one. 
- The `min_total_travelled_distance` attribute stores the `MilpSolution` of the voyage associated with the minimum total amount of distance travelled by the ship. 
- The `min_max_sail_time` attribute store the `MilpSolution` of the voyage associated with the minimum value of the individual maximum individual amount of time spent sailing the ocean with the ship, i.e. minimize the maximum individual amount of time spent in the intermediate journeys to go from one island to the next one. 

</details>

```python
@dataclass(frozen=True)
class ProblemSolutions:
    min_total_compass_time: Optional[MilpSolution] = None
    max_final_crew: Optional[MilpSolution] = None
    min_total_sail_time: Optional[MilpSolution] = None
    min_total_travelled_distance: Optional[MilpSolution] = None
    min_max_sail_time: Optional[MilpSolution] = None
```

Output of the function `solve_milp` you have to implement.


## Tasks - TBD better

For now, I am just listing the possible constraints and cost functions. The tasks should be a combinations of them, with an increasing difficulty.

---

### **Constraints**

#### Voyage order

In you voyage plan, you have to visit one and only one island of each archipelago, and the archipelagos must be visited in order one after the other, based on the archipelago's id (from left to right).

#### Minimum nights to fix a ship

When you arrive to an island, before departing to another island you have to wait at least a minimum amount of nights to being able to fix the ship. This value is the same for every island.

#### Minimum and maximum crew size

During the whole voyage, the crew size must be within a specific min-max range. In other words, whichever island you are in, the crew size must always be within the range. These values are the same for every island.

#### Maximum duration individual journey

Everytime you depart from an island and you arrive to another island, the duration of the individual journey must be less than the maximum duration threshold. This value is the same for every island.

#### Maximum L1-norm distance individual journey

Everytime you depart from an island and you arrive to another island, the distance travelled during the individual journey must be less than the maximum amount of travelled distance. This value is the same for every island. Since we are modeling a linear program, the threshold value represents the L1-norm maximum distance.


### **Cost functions**

#### Minimum nights to complete the voyage

Find the voyage plan that minimize the total duration of the voyage, i.e. minimize the total number of nights.

#### Maximum final crew size

Find the voyage plan that maximize the crew size at the end of the voyage.

#### Minimize total sailing time

Find the voyage plan that minimize the total sailing time of the voyage, i.e. minimize the total number of hours spent in the journeys to depart from an island to arrive to another island.

#### Minimize total L1-norm travelled distance

Find the voyage plan that minimize the total L1-norm travelled distance of the voyage, i.e. minimize the total L1-norm distances of the journeys to depart from an island to arrive to another island.

#### Minimize the maximum individual sailing time

Find the voyage plan that minimize the maximum individual sailing time of the voyage, i.e. minimize the maximum number of hours spent in a single journey to depart from an island to arrive to another island.

---

## Report

You can choose between four different levels of report types: 

- `ReportType.none`: no evalutation information at all, no report is generated.
- `ReportType.terminal`: print evaluation information on the terminal, no report is generated.
- `ReportType.report_txt`: print evaluation information on the terminal and in a textual report.
- `ReportType.report_viz`: print evaluation information on the terminal and in a visual report, with text and figures.

The `report_txt` report displays the costs and the violations of your voyage plans. The `report_viz` report adds an archipelagos map with the islands data and highlights in the map your voyage plan, for an optimal visualization to better comprehend and debug the problem/solution. But the report generation is way slower and could take a long time, especially if the map is big.

In [src/pdm4ar/exercises_def/optimization/milp/visualization.py](../src/pdm4ar/exercises_def/optimization/milp/visualization.py), you can set the *REPORT_TYPE* global variable.

Feel also free to make your modifications to the visualization file to match your debugging needs.

---

## Run the exercise

```shell
pip3 install -e [path/to/exercises_repo]
python3 [path/to/]src/pdm4ar/main.py --exercise 07
```

After running the exercise, a report will be generated that shows your results (if you enabled the report generation).

## Hints
To model the problem notice that in the environment there are already powerful libraries to solve optimization problems.For instance, [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) and [pulp](https://coin-or.github.io/pulp/))

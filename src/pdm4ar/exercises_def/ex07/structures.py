from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Tuple, Literal, Optional, Union
from frozendict import frozendict
import numpy as np

from pdm4ar.exercises_def.structures import PerformanceResults


@dataclass(frozen=True)
class OptimizationCost(IntEnum):
    """Used to specify the optimization cost."""

    min_total_nights = 0
    max_final_crew = 1
    min_total_sailing_time = 2
    min_total_travelled_L1_distance = 3
    min_max_sailing_time = 4

    @classmethod
    def get_costs(cls):
        return [cost for cost in cls]

    def __eq__(self, other: Enum) -> bool:
        return self.value == other.value


@dataclass(frozen=True)
class Feasibility(IntEnum):
    """Used to specify feasibility/unfeasibility status."""

    unfeasible = 0
    feasible = 1

    def __eq__(self, other: IntEnum) -> bool:
        return self.value == other.value


CostType = Literal[
    OptimizationCost.min_total_nights,
    OptimizationCost.max_final_crew,
    OptimizationCost.min_total_sailing_time,
    OptimizationCost.min_total_travelled_L1_distance,
    OptimizationCost.min_max_sailing_time,
]

FeasibilityType = Literal[Feasibility.feasible, Feasibility.unfeasible]

VoyagePlan = List[int]


@dataclass(frozen=True)
class Island:
    """From the doc:\n
    Structure storing the individual features of an island.
    - The `id` integer attribute identifies uniquely the island. The island of the first archipelago has always `id` = 0 while the island of the last archipelago has always `id` = number of islands - 1. If the archipelagos in between have 5 islands each, then the `id` of the islands of the second archipelago ranges from 1 to 5, the ones of the third archipelagos from 6 to 10, the ones of the fourth archipelago from 11 to 15, and so on. When you submit your optimized voyage plan, you are submitting an ordered list of the `id` of the islands you plan to visit.
    - The `arch` integer attribute tells you to which of the N archipelagos the island belongs (0, ..., N-1).
    - The `x` and `y` float attributes specify the x and y position of the island in a cartesian reference system. The 2D map is a flat plane.
    - The `departure` and `arrival` float attributes are a timetable of the exact time you have to depart from or to arrive to the island, to exploit its specific weather to being able to set sail or to dock. Note that to keep things simple the decimal places are not representing the minutes in mod 60. A value of 8.43 doesn't mean 43 minutes past 8, but that it's 43% of an hour past 8. Treat it as a normal float value.
    To keep things simple, the arrival times of all the islands are later than the departure times of all the islands. This means in every possible journey between two island you depart and arrive later on the same day, always.
    - The `nights` integer attribute specifies how many nights you have to spend on the island before you can depart to the next archipelago. If `nights` is 1, it means you arrive in the island and you depart the next day, irrelevant of the arrival/departure timetable.
    - The `delta_crew` integer attribute specifies how many people will leave the crew (negative value) or how many join the crew (positive value) if you visit the island.
    """

    id: int
    arch: int
    x: float
    y: float
    departure: float
    arrival: float
    nights: int
    delta_crew: int


@dataclass(frozen=True)
class Constraints:
    """Structure storing the constraints data. A constraint with a value of `None` is not active.\n
    From the doc:\n
    - The `min_nights_individual_island` integer attribute is a constraint specifing the minimum amount of nights you have to spend in every island to get the ship fixed before departing again to a new island. The ocean currents are badly damaging the ship every time you set sail.
    - The `max_total_crew` integer attributes specify the minimum amount of people who can be in the crew at the same time.
    - The `min_total_crew` integer attributes specify the maximum amount of people who can be in the crew at the same time.
    - The `max_duration_individual_journey` float attribute is a constraint specifing the maximum amount of hours each island-to-island jounrey can last. Treat it as a normal float value.
    - The `max_L1_distance_individual_journey` float attribute is a constraint specifing the maximum L1-norm distance length of each island-to-island journey.
    """

    min_nights_individual_island: Optional[int]
    min_total_crew: Optional[int]
    max_total_crew: Optional[int]
    max_duration_individual_journey: Optional[float]
    max_L1_distance_individual_journey: Optional[float]


@dataclass(frozen=True)
class ProblemVoyage:
    """Used to specify the preferred report type.\n
    ---
    From the doc:\n
    Structure storing the data of an optimization problem. Input of the function `solve_optimization`.
    - The `optimization_cost` CostType attribute declares the cost you have to optimize.
    - The `start_crew` integer attribute specifies how many people are in the crew (including the captain) at the beginning of the voyage.
    - The `islands` attribute is a tuple containing a sequence of `Island`. The islands are ordered based on their `id` attribute.
    - The `constraints` attribute contains the following:
        - The `min_nights_individual_island` integer attribute is a constraint specifing the minimum amount of nights you have to spend in every island to get the ship fixed before departing again to a new island. The ocean currents are badly damaging the ship every time you set sail.
        - The `max_total_crew` integer attributes specify the minimum amount of people who can be in the crew at the same time.
        - The `min_total_crew` integer attributes specify the maximum amount of people who can be in the crew at the same time.
        - The `max_duration_individual_journey` float attribute is a constraint specifing the maximum amount of hours each island-to-island jounrey can last. Treat it as a normal float value.
        - The `max_L1_distance_individual_journey` float attribute is a constraint specifing the maximum L1-norm distance length of each island-to-island journey.
    """

    optimization_cost: CostType
    start_crew: int
    islands: Tuple[Island]
    constraints: Constraints


@dataclass(frozen=True)
class SolutionVoyage:
    """From the doc:\n
    Structure storing the solution of an optimization problem. Output of the function `solve_optimization`. A solution not compliant with the expected structure types will raise a `TestCaseSanityCheckException`.
    - The `feasibility` FeasibilityType attribute specifies if the problem is found unfeasible or feasible.
    - The `voyage_plan` VoyagePlan attribute stores the list of the `id`s of the island in the order you plan to visit them if the problem is feasible, else it should be set to `None`.
    """

    feasibility: FeasibilityType
    voyage_plan: Optional[VoyagePlan]




















# Following structures are not important for the student's exercise development.

@dataclass(frozen=True)
class ReportType(IntEnum):
    """Used to specify preferred report type\n
    From the doc:\n
    - `ReportType.none`: no evalutation information at all, no report is generated.
    - `ReportType.terminal`: print evaluation information on the terminal, no report is generated.
    - `ReportType.report_txt`: print evaluation information on the terminal and in a textual report.
    - `ReportType.report_viz`: print evaluation information on the terminal and in a visual report, with text and figures.
    - `ReportType.report_viz_extra`: print evaluation information on the terminal and in a visual report, with text and figures and data of each island.
    """

    none = 0
    terminal = 1
    report_txt = 2
    report_viz = 3  # slow
    report_viz_extra = 4  # very slow

    def __eq__(self, other: IntEnum) -> bool:
        return self.value == other.value

    def __ge__(self, other: IntEnum) -> bool:
        return self.value >= other.value

    def __le__(self, other: IntEnum) -> bool:
        return self.value <= other.value


CostTolerances = frozendict(
    {
        OptimizationCost.min_total_nights: (0.05, 2),
        OptimizationCost.max_final_crew: (0.05, 2),
        OptimizationCost.min_total_sailing_time: (0.05, 2),
        OptimizationCost.min_total_travelled_L1_distance: (0.05, 10),
        OptimizationCost.min_max_sailing_time: (0.05, 0.4),
    }
)


@dataclass(frozen=True)
class Violations:
    voyage_order: Optional[bool]
    min_nights_individual_island: Optional[bool]
    min_total_crew: Optional[bool]
    max_total_crew: Optional[bool]
    max_duration_individual_journey: Optional[bool]
    max_L1_distance_individual_journey: Optional[bool]


@dataclass(frozen=True)
class Cost:
    feasibility: FeasibilityType
    cost: Optional[Union[int, float]]


@dataclass(frozen=True)
class CostScore:
    type: CostType
    cost: Cost


@dataclass(frozen=True)
class OptimizationPerformance(PerformanceResults):
    feasibility: int
    constraints: Violations
    cost: CostScore


@dataclass(frozen=True)
class ConstraintsPerformance:
    voyage_order: Optional[float]
    min_nights_individual_island: Optional[float]
    min_total_crew: Optional[float]
    max_total_crew: Optional[float]
    max_duration_individual_journey: Optional[float]
    max_L1_distance_individual_journey: Optional[float]


@dataclass(frozen=True)
class CostsPerformance:
    min_total_nights: Optional[float]
    max_final_crew: Optional[float]
    min_total_sailing_time: Optional[float]
    min_total_travelled_L1_distance: Optional[float]
    min_max_sailing_time: Optional[float]


@dataclass(frozen=True)
class Ex07FinalPerformance(PerformanceResults):
    feasibility_performance: float
    constraints_performance: ConstraintsPerformance
    costs_performance: CostsPerformance


@dataclass(frozen=True)
class Tolerance:
    tol: float

    def compare(self, value_1: float, value_2: float) -> bool:
        return np.isclose(value_1, value_2, rtol=0.0, abs_tol=self.tol)

    def is_greater(self, value_1: float, value_2: float) -> bool:
        return value_1 > value_2 + self.tol
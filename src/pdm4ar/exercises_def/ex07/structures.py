from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Tuple, Literal, Optional, Union
import numpy as np

from pdm4ar.exercises_def.structures import PerformanceResults

@dataclass(frozen=True)
class OptimizationCost(IntEnum):
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
class MilpFeasibility(IntEnum):
    unfeasible = 0
    feasible = 1

    def __eq__(self, other: IntEnum) -> bool:
        return self.value == other.value

CostType = Literal[OptimizationCost.min_total_nights, 
                   OptimizationCost.max_final_crew,
                   OptimizationCost.min_total_sailing_time,
                   OptimizationCost.min_total_travelled_L1_distance,
                   OptimizationCost.min_max_sailing_time]
FeasibilityType = Literal[MilpFeasibility.feasible, MilpFeasibility.unfeasible]

VoyagePlan = List[int]


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


@dataclass(frozen=True)
class Constraints:
    min_nights_individual_island: Optional[int]
    min_total_crew: Optional[int]
    max_total_crew: Optional[int]
    max_duration_individual_journey: Optional[float]
    max_L1_distance_individual_journey: Optional[float]

@dataclass(frozen=True)
class ProblemVoyage:
    optimization_cost: CostType
    start_crew: int
    islands: Tuple[Island]
    constraints: Constraints  


@dataclass(frozen=True)
class ProblemSolution:
    feasibility: FeasibilityType
    voyage_plan: Optional[VoyagePlan]






# Following structures are not important for the student's exercise development


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
class MilpPerformance(PerformanceResults):
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


@dataclass(frozen=True)
class ReportType(IntEnum):
    none = 0
    terminal = 1
    report_txt = 2
    report_viz = 3 # slow
    report_viz_extra = 4 # very slow

    def __eq__(self, other: IntEnum) -> bool:
        return self.value == other.value
        
    def __ge__(self, other: IntEnum) -> bool:
        return self.value >= other.value

    def __le__(self, other: IntEnum) -> bool:
        return self.value <= other.value


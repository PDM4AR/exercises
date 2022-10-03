from dataclasses import dataclass
from enum import Enum
from random import random, seed
from typing import List, Tuple, Any, Literal, Optional
import numpy as np

@dataclass(frozen=True)
class MilpFeasibility(Enum):
    feasible = "feasible"
    unfeasible = "unfeasible"

    def __eq__(self, other):
        return self.value == other.value

@dataclass(frozen=True)
class MilpSolution:
    status: Literal[MilpFeasibility.feasible, MilpFeasibility.unfeasible]
    voyage_plan: Optional[List[int]] = None

@dataclass(frozen=True)
class ProblemSolutions:
    min_total_compass_time: Optional[MilpSolution] = None
    max_final_crew: Optional[MilpSolution] = None
    min_total_sail_time: Optional[MilpSolution] = None
    min_total_travelled_distance: Optional[MilpSolution] = None


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

@dataclass(frozen=True)
class PirateProblem1:
    start_crew: int
    islands: Tuple[Island]
    min_fix_time_individual_island: int
    min_crew: int
    max_crew: int
    max_duration_individual_journey: float
    max_distance_individual_journey: float    

@dataclass(frozen=True)
class PirateProblem2(PirateProblem1):
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]   

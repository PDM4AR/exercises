import numpy as np
from pdm4ar.exercises_def.ex07.structures import ProblemVoyage, OptimizationCost, Island, Constraints, MilpFeasibility, ProblemSolution


def solve_milp(problem: ProblemVoyage) -> ProblemSolution:
    """
    Solve the MILP problem optimizing for the different costs while enforcing the given constraints
    """

    # toy data for random voyage plan
    # np.random.seed(None)
    if np.random.random() > 0.5:
        feasibility = MilpFeasibility.feasible
        voyage_plan = list(np.random.randint(0,len(problem.islands), size=(min(7,len(problem.islands)),)))
    else:
        feasibility = MilpFeasibility.unfeasible
        voyage_plan = None

    return ProblemSolution(feasibility, voyage_plan)
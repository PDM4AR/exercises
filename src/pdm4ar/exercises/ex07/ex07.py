import numpy as np
from pdm4ar.exercises_def.ex07.structures import ProblemVoyage, OptimizationCost, Island, Constraints, MilpFeasibility, ProblemSolution


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
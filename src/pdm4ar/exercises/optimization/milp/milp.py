import numpy as np

from pdm4ar.exercises_def.optimization.milp.structures import Island, MilpSolution, MilpFeasibility, PirateProblem1, ProblemSolutions


def solve_milp(milp_problem_1: PirateProblem1) -> ProblemSolutions:
    """
    Solve the MILP problem
    """

    # example results
    voyage_plan = list(np.random.randint(0,len(milp_problem_1.islands), size=(min(6,len(milp_problem_1.islands)),)))
    solution_min_nights = MilpSolution(MilpFeasibility.feasible, voyage_plan)
    solution_max_crew = MilpSolution(MilpFeasibility.feasible, voyage_plan)
    solution_min_sail_time = MilpSolution(MilpFeasibility.feasible, voyage_plan)
    solution_min_travelled_distance = MilpSolution(MilpFeasibility.unfeasible, None)
    return ProblemSolutions(solution_min_nights, solution_max_crew, solution_min_sail_time, solution_min_travelled_distance)

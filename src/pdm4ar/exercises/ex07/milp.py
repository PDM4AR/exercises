import numpy as np
from pdm4ar.exercises_def.ex07.structures import Island, MilpSolution, MilpFeasibility, ProblemVoyage1, ProblemSolutions


def solve_milp(milp_problem_1: ProblemVoyage1) -> ProblemSolutions:
    """
    Solve the MILP problem optimizing for the different costs while enforcing the given constraints
    """

    # toy data for random voyage plans

    voyage_plan = list(np.random.randint(0,len(milp_problem_1.islands), size=(min(6,len(milp_problem_1.islands)),)))
    solution_min_nights = MilpSolution(MilpFeasibility.feasible, voyage_plan)

    voyage_plan = list(np.random.randint(0,len(milp_problem_1.islands), size=(min(10,len(milp_problem_1.islands)),)))
    solution_max_crew = MilpSolution(MilpFeasibility.feasible, voyage_plan)

    voyage_plan = list(np.random.randint(0,len(milp_problem_1.islands), size=(min(4,len(milp_problem_1.islands)),)))
    solution_min_sail_time = MilpSolution(MilpFeasibility.feasible, voyage_plan)

    solution_min_travelled_distance = MilpSolution(MilpFeasibility.unfeasible, None)

    voyage_plan = list(np.random.randint(0,len(milp_problem_1.islands), size=(min(7,len(milp_problem_1.islands)),)))
    solution_min_max_sail_time = MilpSolution(MilpFeasibility.feasible, voyage_plan)

    return ProblemSolutions(solution_min_nights, solution_max_crew, 
            solution_min_sail_time, solution_min_travelled_distance, solution_min_max_sail_time)
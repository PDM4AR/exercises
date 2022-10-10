from itertools import product
from dataclasses import asdict, dataclass
import pickle
import traceback
from typing import Any, Callable, Dict, Optional
from reprep import Report
import numpy as np
import timeit
from pdm4ar.exercises_def.optimization.milp.structures import MilpFeasibility, MilpFinalPerformance, MilpPerformance, \
    MilpSolution, PerformanceWeight, ProblemSolutions, SolutionViolations, SolutionsCosts, aViolations

from pdm4ar.exercises_def.optimization.milp.visualization import visualize_journey_plan
from pdm4ar.exercises_def.structures import PerformanceResults

from .data import *
from pdm4ar.exercises.optimization.milp.milp import solve_milp
from pdm4ar.exercises_def import Exercise, ExIn


@dataclass
class TestMilp(ExIn):
    type: str
    difficulty: str
    number_of_test_cases: int
    seed: int
    sample_generator: Callable
    visualizer: Callable
    ex_function: Callable

    def str_id(self) -> str:
        return f"{self.type.value}-{self.difficulty.value}"


def milp_performance_aggregator(performances: List[MilpPerformance]) -> MilpFinalPerformance:

    performance_weights = PerformanceWeight(0.2, 0.2, 0.2, 0.2, 0.2)
    mean_performances = []
    overall_performance = 0
    for cost_name in MilpPerformance.__annotations__.keys():
        cost_score = 0
        if all([getattr(performance, cost_name) is None for performance in performances]):
            mean_performances.append(np.nan)
            continue
        for performance in performances:
            cost_score += getattr(performance, cost_name)
        cost_score /= len(performances)
        mean_performances.append(cost_score)
        overall_performance += getattr(performance_weights, cost_name)*cost_score

    return MilpFinalPerformance(*mean_performances, overall_performance)

def compute_individual_violations(problem: ProblemVoyage1, solution: MilpSolution) -> aViolations:
    islands = problem.islands
    voyage_plan = solution.voyage_plan

    violation = None
    n_arch = max(islands, key= lambda x: x.arch).arch+1
    n_arch_visited = len(set([islands[island_id].arch for island_id in voyage_plan]))
    if n_arch_visited != n_arch:
        violation = n_arch_visited - n_arch
    all_arch_violation = violation

    violation = None
    if len(voyage_plan) != n_arch:
        violation = len(voyage_plan) - n_arch
    n_islands_violation = violation

    violation = None
    if len(set(voyage_plan)) != len(voyage_plan):
        violation = len(voyage_plan) - len(set(voyage_plan))
    single_visit_violation= violation
    
    violation = None
    for arch_id, island_id in zip(range(n_arch),voyage_plan):
        if islands[island_id].arch != arch_id:
            violation = 1 if violation is None else violation+1
    order_visit_violation = violation

    violation = None
    for island_id in voyage_plan:
        diff = islands[island_id].time_compass - problem.min_fix_time_individual_island
        if diff < 0:
            violation = diff if violation is None else min(diff, violation)
    min_fix_time_violation = violation
    
    violation = None
    crew = problem.start_crew
    for island_id in voyage_plan:
        crew += islands[island_id].crew
        diff = crew - problem.min_crew
        if diff < 0:
            violation = diff if violation is None else min(diff, violation)
    min_crew_violation = violation

    violation = None
    crew = problem.start_crew
    for island_id in voyage_plan:
        crew += islands[island_id].crew
        diff = problem.max_crew - crew
        if diff < 0:
            violation = -diff if violation is None else max(-diff, violation)
    max_crew_violation = violation

    violation = None
    for voyage_idx in range(len(voyage_plan)-1):
        dep = islands[voyage_plan[voyage_idx]].departure
        arr = islands[voyage_plan[voyage_idx+1]].arrival
        diff = problem.max_duration_individual_journey - (arr-dep)
        if diff < 0:
            violation = -diff if violation is None else max(-diff, violation)
    max_duration_violation = violation

    violation = None
    for voyage_idx in range(len(voyage_plan)-1):
        island_id = voyage_plan[voyage_idx]
        next_island_id = voyage_plan[voyage_idx+1]
        island_pos = np.array([islands[island_id].x, islands[island_id].y])
        next_island_pos = np.array([islands[next_island_id].x, islands[next_island_id].y])
        diff = problem.max_distance_individual_journey - np.linalg.norm(next_island_pos-island_pos, ord=1)
        if diff < 0:
            violation = -diff if violation is None else max(-diff, violation)
    max_distance_violation = violation

    return aViolations(all_arch_violation, n_islands_violation, single_visit_violation, 
                      order_visit_violation, min_fix_time_violation, min_crew_violation, 
                      max_crew_violation, max_duration_violation, max_distance_violation)

def compute_violations(problem: ProblemVoyage1, solutions: ProblemSolutions) -> SolutionViolations:

    violations = [None for _ in solutions.__dict__.items()]

    for idx_cost, (name_cost, solution) in enumerate(solutions.__dict__.items()):
        status = solution.status
        voyage_plan = solution.voyage_plan

        if status == MilpFeasibility.feasible and \
            voyage_plan is not None:
            violations[idx_cost] = compute_individual_violations(problem, solution)
        
    return SolutionViolations(*violations)



def compute_costs(problem: ProblemVoyage1, solutions: ProblemSolutions) -> Dict:

    est_costs = [None for _ in solutions.__dict__.items()]
    islands = problem.islands

    for idx_cost, (name_cost, solution) in enumerate(solutions.__dict__.items()):

        status = solution.status
        voyage_plan = solution.voyage_plan

        if status == MilpFeasibility.feasible and \
            voyage_plan is not None:

            if name_cost == "min_total_compass_time":
                cost = 0
                for island_id in voyage_plan:
                    cost += islands[island_id].time_compass

            elif name_cost == "max_final_crew":
                cost = problem.start_crew
                for island_id in voyage_plan:
                    cost += islands[island_id].crew

            elif name_cost == "min_total_sail_time":
                cost = 0
                for idx in range(len(voyage_plan)-1):
                    island_id = voyage_plan[idx]
                    next_island_id = voyage_plan[idx+1]
                    cost += islands[next_island_id].arrival - islands[island_id].departure

            elif name_cost == "min_total_travelled_distance":
                cost = 0
                for idx in range(len(voyage_plan)-1):
                    island_id = voyage_plan[idx]
                    next_island_id = voyage_plan[idx+1]
                    island_pos = np.array([islands[island_id].x, islands[island_id].y])
                    next_island_pos = np.array([islands[next_island_id].x, islands[next_island_id].y])
                    cost += np.linalg.norm(next_island_pos-island_pos, ord=1)

            elif name_cost == "min_max_sail_time":
                cost = 0
                for idx in range(len(voyage_plan)-1):
                    island_id = voyage_plan[idx]
                    next_island_id = voyage_plan[idx+1]
                    sail_time = islands[next_island_id].arrival - islands[island_id].departure
                    if sail_time > cost:
                        cost = sail_time

            else:
                raise ValueError(f"The attribute {name_cost} of {type(solutions).__name__} is not recognized.")

        elif status == MilpFeasibility.unfeasible:
            cost = MilpFeasibility.unfeasible.value
        else:
            cost = None
        
        est_costs[idx_cost] = cost

    return SolutionsCosts(*est_costs)

# TODO remove this function when development completed. It checks if a bug is present in gt solution
def debug_violations_gt(problem: ProblemVoyage1, est_solution: ProblemSolutions) -> None:
    gt_violations = compute_violations(problem, est_solution)
    for name_cost in est_solution.__annotations__.keys():
        violations = getattr(gt_violations, name_cost)
        assert all([getattr(violations, violation_name) is None for violation_name in violations.__annotations__.keys()]), \
            f"gt solution is violating some constraints {asdict(getattr(gt_violations, name_cost))}"

# TODO remove this function when development completed. It checks if a bug is present in gt solution
def debug_cost_gt(performance_violation_score, est_cost, gt_optimal_cost) -> None:
    assert performance_violation_score == 1 and est_cost < gt_optimal_cost, \
            f"est cost {est_cost} smaller than GT cost {gt_optimal_cost}, no violations"


def milp_report(
    algo_in: TestMilp, expected_out: Any
) -> Tuple[MilpPerformance, Report]:

    img_width_px = 200

    # database = {}

    title = f"MILP Exercise {algo_in.difficulty.value}"
    print('\033[33;46m'+title+'\033[0m'+"\n")

    r = Report(title)

    algo_in_type = algo_in.type
    algo_in_difficulty = algo_in.difficulty
    algo_in_number_of_test_cases = algo_in.number_of_test_cases
    algo_in_seed = algo_in.seed

    weight_cost_score = 0.35
    weight_violation_score = 0.65

    assert weight_cost_score + weight_violation_score == 1

    epsilon = 1e-3
    success_string = '\033[92mSUCCESS\033[0m'
    fail_string = '\033[91mFAIL\033[0m'

    for ex_num in range(algo_in_number_of_test_cases):
        title_section = f"{algo_in_type.value}-test {ex_num}"
        rsec = r.section(title_section)
        print('\033[94;103m'+title_section+'\033[0m'+"\n")
        
        seed_bias = ex_num if algo_in_seed != 0 else 0
        if algo_in_type == MilpCase.voyage1 or algo_in_type == MilpCase.voyage2:
            problem_data = algo_in.sample_generator(algo_in_type, algo_in_difficulty, algo_in_seed+seed_bias)
        elif algo_in_type == MilpCase.testvoyage1 or algo_in_type == MilpCase.testvoyage2:
            assert expected_out is not None
            problem_data = database_reader(expected_out, ex_num)
        else:
            raise ValueError(algo_in_type)

        problem, gt_optimal_costs, slack_violations, slack_costs = problem_data

        start = timeit.default_timer()
        try:
            est_solution = algo_in.ex_function(problem)
        except Exception as e:
            print(f"Failed because of:\n {e.args} \n{''.join(traceback.format_tb(e.__traceback__))}")
            raise e

        stop = timeit.default_timer()
        timing = stop-start

        # database[ex_num] = {'problem': problem, 'solution': est_solution, 'slack_violations': slack_violations, 'slack_costs': slack_costs}

        est_costs = compute_costs(problem, est_solution)
        violations = compute_violations(problem, est_solution)

        if timing > 30: # TODO change timing
            if gt_optimal_costs is None:
                # debug_violations_gt(problem, est_solution)
                performance = MilpPerformance(np.nan, np.nan, np.nan, np.nan, np.nan)
            else:         
                performance =  MilpPerformance(0, 0, 0, 0, 0)
        else:
            timing = None
            if gt_optimal_costs is None:
                # debug_violations_gt(problem, est_solution)
                performance = MilpPerformance(np.nan, np.nan, np.nan, np.nan, np.nan)
            else:

                performance_scores = [None for _ in range(len(gt_optimal_costs.__annotations__))]      
                n_slack_violations = len([_ for _ in slack_violations])   

                for performance_idx, name_cost in enumerate(gt_optimal_costs.__annotations__.keys()):
                    gt_optimal_cost = getattr(gt_optimal_costs, name_cost)
                    if gt_optimal_cost is None:
                        continue
                    if getattr(gt_optimal_costs, name_cost) == MilpFeasibility.feasible and getattr(est_costs, name_cost) == MilpFeasibility.feasible:
                        max_cost_error = getattr(slack_costs, name_cost)
                        if max_cost_error != 0:
                            performance_cost_score = 1-min(abs(getattr(est_costs, name_cost) - getattr(gt_optimal_costs, name_cost))/max_cost_error,1)
                        else:
                            performance_cost_score = 1 if getattr(est_costs, name_cost) - getattr(gt_optimal_costs, name_cost) == 0 else 0
                        performance_violation_score = 1
                        violation = getattr(violations, name_cost)
                        for name_constraint in slack_violations.__annotations__.keys():
                            max_violation_error = getattr(slack_violations, name_constraint)
                            if max_violation_error != 0:
                                performance_violation_score -= min(abs(getattr(violation, name_constraint))/max_violation_error,1)/n_slack_violations
                            else:
                                performance_violation_score -= 0 if getattr(violation, name_constraint) == 0 else 1/n_slack_violations

                        # debug_cost_gt(performance_violation_score, getattr(est_costs, name_cost), getattr(gt_optimal_costs, name_cost))

                        performance_score = max(min(weight_cost_score*performance_cost_score + weight_violation_score*performance_violation_score, 1), 0)

                    elif getattr(gt_optimal_costs, name_cost) == MilpFeasibility.unfeasible and getattr(est_costs, name_cost) == MilpFeasibility.unfeasible:
                        performance_score = 1
                    else:
                        performance_score = 0
                    performance_scores[performance_idx] = performance_score

                performance = MilpPerformance(*performance_scores)

        algo_in.visualizer(rsec, problem, est_solution, est_costs, violations, timing, img_width_px)


    # with open(f'{algo_in_difficulty.value}_database_tests.pkl', 'wb') as f:
    #     pickle.dump(database, f)

    # with open(f'{algo_in_difficulty.value}_database_tests.pkl', 'rb') as f:
    #     database = pickle.load(f)

    return performance, r


def get_exercise_optimization_milp() -> Exercise:
    test_types = MilpCase.get_test_milp_cases_types()
    test_difficulties = MilpCase.get_test_milp_cases_difficulties()

    test_numbers = 1
    seed = 1

    if MilpCase.testvoyage1 in test_types or MilpCase.testvoyage2 in test_types:
        with open('database_tests', 'rb') as f:
            complete_database = pickle.load(f)

    test_values = []
    expected_results = []

    test_difficulties = [MilpCase.easy]

    for test_type, test_difficulty in product(test_types, test_difficulties):

        test_values.append(TestMilp(
            test_type,
            test_difficulty,
            test_numbers,
            seed,
            milp_generator,
            visualize_journey_plan,
            solve_milp
        ))

        if test_type in (MilpCase.testvoyage1, MilpCase.testvoyage2):
            database = complete_database[test_difficulty][test_type]
        else:
            database = None
        expected_results.append(database)

    return Exercise[TestMilp, Any](
        desc="This exercise solves MILPs.",
        evaluation_fun=milp_report,
        perf_aggregator=milp_performance_aggregator,
        test_values=test_values,
        expected_results=expected_results,
        test_case_timeout = 60*6
    )

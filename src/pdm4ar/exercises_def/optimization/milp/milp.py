from itertools import product
from dataclasses import dataclass
from typing import Any, Callable, Dict
from reprep import Report
import numpy as np
import timeit
from pdm4ar.exercises_def.optimization.milp.structures import MilpFeasibility, ProblemSolutions

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
        return f"test-{self.type}-{self.difficulty}"

@dataclass(frozen=True)
class MilpPerformance(PerformanceResults):
    nights_score: float
    crew_score: float
    sail_score: float
    distance_score: float

def milp_performance_aggregator(performances: List[MilpPerformance]) -> MilpPerformance:
    final_performance = []
    for cost_name in MilpPerformance.__annotations__.keys():
        cost_score = 0
        for performance in performances:
            cost_score += getattr(performance, cost_name)
        cost_score /= len(performances)
        final_performance.append(cost_score)
    
    return MilpPerformance(*final_performance)

def compute_costs(problem: PirateProblem1, solutions: ProblemSolutions) -> Dict:

    est_costs = {}
    islands = problem.islands

    for name_cost, solution in solutions.__dict__.items():
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

            else:
                raise ValueError(f"The attribute {name_cost} of {type(solutions).__name__} is not recognized.")

        elif status == MilpFeasibility.unfeasible:
            cost = MilpFeasibility.unfeasible
        else:
            cost = None
        
        est_costs[name_cost] = cost

    return est_costs

def milp_report(
    algo_in: TestMilp, expected_out: Any
) -> Tuple[MilpPerformance, Report]:

    img_width_px = 2000

    r = Report("MILP Exercise")
    r.section(algo_in.type)
    algo_in_type = algo_in.type
    algo_in_difficulty = algo_in.difficulty
    algo_in_number_of_test_cases = algo_in.number_of_test_cases
    algo_in_seed = algo_in.seed

    performance_cost_weights = {'min_compass': 0.25, 'max_crew': 0.25, 'min_sail': 0.25, 'min_distance': 0.25}
    assert sum(list(performance_cost_weights.values())) == 1

    epsilon = 1e-3
    success_string = '\033[92mSUCCESS\033[0m'
    fail_string = '\033[91mFAIL\033[0m'

    for ex_num in range(algo_in_number_of_test_cases):
        title = f"{algo_in.str_id()}-{ex_num}"
        
        seed_bias = ex_num if algo_in_seed != 0 else 0
        problem, gt_optimal_costs = algo_in.sample_generator(algo_in_type, algo_in_difficulty, algo_in_seed+seed_bias)

        start = timeit.default_timer()
        est_solution = algo_in.ex_function(problem)
        stop = timeit.default_timer()

        est_costs = compute_costs(problem, est_solution)

        if gt_optimal_costs is None:
            performance = MilpPerformance(np.nan, np.nan, np.nan, np.nan)
        else:
            # TODO change the saturation errors based on the specific problem setting
            if gt_optimal_costs['min_nights'] == MilpFeasibility.feasible and est_costs['min_nights'] == MilpFeasibility.feasible:
                saturation_supply_error = 3
                perf_score_nights = 1-max(abs(est_costs['min_nights'] - gt_optimal_costs['min_nights'])/saturation_supply_error,1)
            else:
                perf_score_nights = 1 if (gt_optimal_costs['min_nights'] == MilpFeasibility.unfeasible and est_costs['min_nights'] == MilpFeasibility.unfeasible) else 0

            if gt_optimal_costs['max_crew'] == MilpFeasibility.feasible and est_costs['max_crew'] == MilpFeasibility.feasible:
                saturation_crew_error = 5
                perf_score_crew = 1-max(abs(est_costs['max_crew'] - gt_optimal_costs['max_crew'])/saturation_crew_error,1)
            else:
                perf_score_crew = 1 if (gt_optimal_costs['max_crew'] == MilpFeasibility.unfeasible and est_costs['max_crew'] == MilpFeasibility.unfeasible) else 0

            if gt_optimal_costs['min_sail'] == MilpFeasibility.feasible and est_costs['min_sail'] == MilpFeasibility.feasible:
                saturation_sail_error = 10
                perf_score_sail = 1-max(abs(est_costs['min_sail'] - gt_optimal_costs['min_sail'])/saturation_sail_error,1)
            else:
                perf_score_sail = 1 if (gt_optimal_costs['min_sail'] == MilpFeasibility.unfeasible and est_costs['min_sail'] == MilpFeasibility.unfeasible) else 0
                
            if gt_optimal_costs['min_distance'] == MilpFeasibility.feasible and est_costs['min_distance'] == MilpFeasibility.feasible:
                saturation_distance_error = 100
                perf_score_distance = 1-max(abs(est_costs['min_distance'] - gt_optimal_costs['min_distance'])/saturation_distance_error,1)
            else:
                perf_score_distance = 1 if (gt_optimal_costs['min_distance'] == MilpFeasibility.unfeasible and est_costs['min_distance'] == MilpFeasibility.unfeasible) else 0

            performance = MilpPerformance(perf_score_nights, perf_score_crew, perf_score_sail, perf_score_distance)


        # data = [gt_solution. est_solution]
        algo_in.visualizer(r, title, problem, est_solution, est_costs, img_width_px)

        evaluation_completed = False
        if evaluation_completed:      
            if gt_solution.status != MilpCase.unfeasible and est_solution.status != MilpCase.unfeasible:
                gt_cost = problem.cost_function.dot(gt_solution.solution)
                est_cost = problem.cost_function.dot(est_solution.solution)
                cost_error = np.abs(gt_cost-est_cost)
                # LN_error = np.linalg.norm(gt_solution.solution-est_solution.solution, ord=len(gt_solution.solution))
            elif gt_solution.status == MilpCase.unfeasible and est_solution.status == MilpCase.unfeasible:
                cost_error = 0
            else:
                cost_error = float('inf')
            
            r.text(
                f"{algo_in.str_id()}-{ex_num}",
                f" {f'{success_string}' if cost_error < epsilon else f'{fail_string} with cost error {cost_error}'} | "
                f"Optimal: cost= {gt_cost if gt_solution.status == MilpCase.feasible else None}, "
                f"x= {gt_solution.solution if gt_solution.status == MilpCase.feasible else gt_solution.status} | "
                f"Estimate: cost= {est_cost if est_solution.status == MilpCase.feasible else None}, "
                f"x= {est_solution.solution if est_solution.status == MilpCase.feasible else est_solution.status} | "
                f"Execution Time = {round(stop - start, 5)}",
            )
            print(f"{algo_in.str_id()}-{ex_num}"
                f" {f'{success_string}' if cost_error < epsilon else f'{fail_string} with cost error {cost_error}'} | "
                f"Optimal: cost= {gt_cost if gt_solution.status == MilpCase.feasible else None}, "
                f"x= {gt_solution.solution if gt_solution.status == MilpCase.feasible else gt_solution.status} | "
                f"Estimate: cost= {est_cost if est_solution.status == MilpCase.feasible else None}, "
                f"x= {est_solution.solution if est_solution.status == MilpCase.feasible else est_solution.status} | "
                f"Execution Time = {round(stop - start, 5)}")

    return performance, r


def algo_placeholder(ex_in):
    return None


def get_exercise_optimization_milp() -> Exercise:
    test_types = MilpCase.get_test_milp_cases_types()
    test_difficulties = MilpCase.get_test_milp_cases_difficulties()
    test_numbers = 3
    seed = 0

    test_values = list()
    test_difficulties = [MilpCase.easy]
    for test_type, test_difficulty in product(test_types, test_difficulties):
        milp_solver = solve_milp
        test_values.append(TestMilp(
            test_type,
            test_difficulty,
            test_numbers,
            seed,
            milp_generator,
            visualize_journey_plan,
            milp_solver
        ))

    return Exercise[TestMilp, Any](
        desc="This exercise solves MILPs exploiting the branch and bound algorithm.",
        evaluation_fun=milp_report,
        perf_aggregator=milp_performance_aggregator,
        test_values=test_values,
        expected_results=None
    )

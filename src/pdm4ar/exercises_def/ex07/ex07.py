from itertools import product
from dataclasses import asdict, dataclass
import pathlib
import pickle
from tkinter import E
import traceback
from typing import Any, Callable, Dict, Optional, Type
from reprep import Report
import numpy as np
import timeit
from pdm4ar.exercises_def.ex07.structures import ConstraintsPerformance, Cost, CostScore, CostType, CostsPerformance, MilpFeasibility, \
    Ex07FinalPerformance, MilpPerformance, OptimizationCost, ProblemSolution, Tolerance, Violations

from pdm4ar.exercises_def.ex07.visualization import Viz
from pdm4ar.exercises_def.structures import PerformanceResults
from pdm4ar.exercises_def.structures_time import TestCaseTimeoutException

from .data import *
from pdm4ar.exercises.ex07.ex07 import solve_optimization
from pdm4ar.exercises_def import Exercise, ExIn

class SanityCheckError(Exception):
    pass

@dataclass
class TestMilp(ExIn):
    type: IntEnum
    optimization_cost: CostType
    test_id: int
    problem: Optional[ProblemVoyage]
    seed: Optional[int]
    timeout: int

    def str_id(self) -> str:
        return f"{self.optimization_cost.name} - test {self.test_id}"


def ex07_performance_aggregator(performances: List[MilpPerformance], voyage_type: MilpCase) -> Ex07FinalPerformance:

    overall_feasibility = 0.0
    overall_constraints = {key: 0.0 for key in Violations.__annotations__.keys()}
    overall_costs = {key: 0.0 for key in OptimizationCost.__members__.keys()}
    overall_n_test_feasibility = 0
    overall_n_test_constraints = {key: 0 for key in Violations.__annotations__.keys()}
    overall_n_test_costs = {key: 0 for key in OptimizationCost.__members__.keys()}
    
    for performance in performances:

        feasibility_score = performance.feasibility
        violations = performance.constraints
        cost = performance.cost
        feasibility = cost.cost.feasibility

        # FEASIBILITY
        if voyage_type == MilpCase.test_voyage:
            overall_feasibility += feasibility_score
            overall_n_test_feasibility += 1

        # VIOLATIONS
        for violation_name in violations.__annotations__.keys():
            violation = getattr(violations, violation_name)
            if feasibility_score == 1:
                if feasibility == MilpFeasibility.feasible:
                    if violation is False:
                        overall_constraints[violation_name] += 1
                    overall_n_test_constraints[violation_name] += 1
            else:
                if feasibility == MilpFeasibility.feasible:
                    if violation is False:
                        overall_constraints[violation_name] += 1
                    overall_n_test_constraints[violation_name] += 1

        # COSTS
        if voyage_type == MilpCase.test_voyage:
            if feasibility_score == 1:
                if feasibility == MilpFeasibility.feasible:
                    overall_costs[cost.type.name] += cost.cost.cost
                    overall_n_test_costs[cost.type.name] += 1
        else:
            overall_costs[cost.type.name] = cost.cost.cost
            overall_n_test_costs[cost.type.name] += 1
            

    if overall_n_test_feasibility > 0:
        overall_feasibility /= overall_n_test_feasibility
    for cost_name in OptimizationCost.__members__.keys():
        if overall_n_test_costs[cost_name] > 0:
            overall_costs[cost_name] /= overall_n_test_costs[cost_name]
    for constraint_name in Violations.__annotations__.keys():
        if overall_n_test_constraints[constraint_name] > 0:
            overall_constraints[constraint_name] /= overall_n_test_constraints[constraint_name]

    feasibility_performance = overall_feasibility if voyage_type == MilpCase.test_voyage else np.nan

    constraints_performance = ConstraintsPerformance(*[None for _ in ConstraintsPerformance.__annotations__.keys()])
    for constraint_name in ConstraintsPerformance.__annotations__.keys():
        object.__setattr__(constraints_performance, constraint_name, overall_constraints[constraint_name])

    costs_performance = CostsPerformance(*[None for _ in OptimizationCost.__members__.keys()])
    for cost_name in CostsPerformance.__annotations__.keys():
        object.__setattr__(costs_performance, cost_name, overall_costs[cost_name])

    return Ex07FinalPerformance(feasibility_performance, constraints_performance, costs_performance)


def compute_violations(problem: ProblemVoyage, solution: ProblemSolution) -> Violations:

    constraints = problem.constraints
    feasibility = solution.feasibility
    voyage_plan = solution.voyage_plan

    tolerance = Tolerance(1e-3)

    if feasibility == MilpFeasibility.feasible:

        if voyage_plan is not None:
        
            islands = problem.islands
            voyage_plan = solution.voyage_plan

            violation = False
            n_arch = max(islands, key= lambda x: x.arch).arch+1
            if len(voyage_plan) != n_arch:
                violation = True
            else:
                for id_arch, id_island in enumerate(voyage_plan):
                    if islands[id_island].arch != id_arch:
                        violation = True
                        break
            order_voyage_violation = violation

            if order_voyage_violation:
                if constraints.min_nights_individual_island is not None:
                    min_fix_time_violation = True
                if constraints.min_total_crew is not None:
                    min_crew_violation = True

                violations = Violations(*[None for _ in Violations.__annotations__.keys()])
                object.__setattr__(violations, "voyage_order", True)
                for violation_name in constraints.__annotations__.keys():
                    if getattr(constraints, violation_name) is not None:
                        object.__setattr__(violations, violation_name, True)
            
            else:
                violation = None
                if constraints.min_nights_individual_island is not None:
                    violation = False
                    for island_id in voyage_plan:
                        if islands[island_id].arch not in (0, n_arch-1):
                            if tolerance.is_greater(constraints.min_nights_individual_island, islands[island_id].nights):
                                violation = True
                                break
                min_fix_time_violation = violation
                
                violation = None
                if constraints.min_total_crew is not None:
                    violation = False
                    crew = problem.start_crew
                    for island_id in voyage_plan:
                        crew += islands[island_id].delta_crew
                        if tolerance.is_greater(constraints.min_total_crew, crew):
                            violation = True
                            break
                min_crew_violation = violation

                violation = None
                if constraints.max_total_crew is not None:
                    violation = False
                    crew = problem.start_crew
                    for island_id in voyage_plan:
                        crew += islands[island_id].delta_crew
                        if tolerance.is_greater(crew, constraints.max_total_crew):
                            violation = True
                            break
                max_crew_violation = violation

                violation = None
                if constraints.max_duration_individual_journey is not None:
                    violation = False
                    for voyage_idx in range(len(voyage_plan)-1):
                        island_id = voyage_plan[voyage_idx]
                        next_island_id = voyage_plan[voyage_idx+1]
                        dep = islands[island_id].departure
                        arr = islands[next_island_id].arrival
                        if tolerance.is_greater(arr-dep, constraints.max_duration_individual_journey):
                            violation = True
                            break
                max_duration_violation = violation

                violation = None
                if constraints.max_L1_distance_individual_journey is not None:
                    violation = False
                    for voyage_idx in range(len(voyage_plan)-1):
                        island_id = voyage_plan[voyage_idx]
                        next_island_id = voyage_plan[voyage_idx+1]
                        island_pos = np.array([islands[island_id].x, islands[island_id].y])
                        next_island_pos = np.array([islands[next_island_id].x, islands[next_island_id].y])
                        if tolerance.is_greater(np.linalg.norm(next_island_pos-island_pos, ord=1), 
                                                constraints.max_L1_distance_individual_journey):
                            violation = True
                            break
                max_distance_violation = violation

                violations = Violations(order_voyage_violation, min_fix_time_violation, 
                            min_crew_violation, max_crew_violation, 
                            max_duration_violation, max_distance_violation)
        else:
            violations = Violations(*[True for _ in Violations.__annotations__.keys()])
            
    elif feasibility == MilpFeasibility.unfeasible:
        violations = Violations(*[None for _ in Violations.__annotations__.keys()])
        
    return violations


def compute_cost(problem: ProblemVoyage, solution: ProblemSolution) -> Cost:

    islands = problem.islands
    optimization_cost = problem.optimization_cost
    feasibility = solution.feasibility
    voyage_plan = solution.voyage_plan

    if feasibility == MilpFeasibility.feasible and voyage_plan is not None:

        if optimization_cost == OptimizationCost.min_total_nights:
            cost = 0
            for island_id in voyage_plan:
                cost += islands[island_id].nights

        elif optimization_cost == OptimizationCost.max_final_crew:
            cost = problem.start_crew
            for island_id in voyage_plan:
                cost += islands[island_id].delta_crew

        elif optimization_cost == OptimizationCost.min_total_sailing_time:
            cost = 0
            for idx in range(len(voyage_plan)-1):
                island_id = voyage_plan[idx]
                next_island_id = voyage_plan[idx+1]
                cost += islands[next_island_id].arrival - islands[island_id].departure

        elif optimization_cost == OptimizationCost.min_total_travelled_L1_distance:
            cost = 0
            for idx in range(len(voyage_plan)-1):
                island_id = voyage_plan[idx]
                next_island_id = voyage_plan[idx+1]
                island_pos = np.array([islands[island_id].x, islands[island_id].y])
                next_island_pos = np.array([islands[next_island_id].x, islands[next_island_id].y])
                cost += np.linalg.norm(next_island_pos-island_pos, ord=1)

        elif optimization_cost == OptimizationCost.min_max_sailing_time:
            cost = 0
            for idx in range(len(voyage_plan)-1):
                island_id = voyage_plan[idx]
                next_island_id = voyage_plan[idx+1]
                sail_time = islands[next_island_id].arrival - islands[island_id].departure
                if sail_time > cost:
                    cost = sail_time

        else:
            raise ValueError(optimization_cost)

    elif feasibility == MilpFeasibility.unfeasible:
        cost = None

    else:
        raise ValueError(feasibility)
    
    return Cost(feasibility, cost)

def compute_feasibility_score(est_cost: Cost, gt_cost: Optional[Cost]) -> int:
    if gt_cost is not None:
        if est_cost.feasibility == gt_cost.feasibility:
            feasibility_score = 1
        else:
            feasibility_score = 0
    else:
        feasibility_score = np.nan

    return feasibility_score

def compute_cost_score(cost_type: CostType, est_cost: Cost, gt_cost: Optional[Cost], violations: Violations) -> CostScore:
    if gt_cost is not None:
        if any(getattr(violations, violation_name) for violation_name in violations.__annotations__.keys()):
            cost_score = 0
        elif est_cost.feasibility == gt_cost.feasibility:
            if est_cost.feasibility == MilpFeasibility.feasible:
                rel_tol_cost = 0.05
                min_abs_tol_cost = 2
                tol = max(rel_tol_cost*gt_cost.cost, min_abs_tol_cost)
                tolerance = Tolerance(tol)
                if cost_type != OptimizationCost.max_final_crew:
                    if tolerance.is_greater(est_cost.cost, gt_cost.cost):
                        cost_score = 0
                    else:
                        rel_diff = min(est_cost.cost - gt_cost.cost, 0)/tol
                        cost_score = 1 - rel_diff
                else:
                    if tolerance.is_greater(gt_cost.cost, est_cost.cost):
                        cost_score = 0
                    else:
                        rel_diff = min(gt_cost.cost, est_cost.cost, 0)/tol
                        cost_score = 1 - rel_diff
            else:
                cost_score = 1
        else:
            cost_score = 0

        cost_score = max(0, min(cost_score, 1))
        if cost_score > 0.99:
            cost_score = 1

    else:
        cost_score = np.nan

    cost = Cost(est_cost.feasibility, cost_score)
    cost_score = CostScore(cost_type, cost)

    return cost_score

def sanity_check(solution: ProblemSolution):
    if isinstance(solution, ProblemSolution):
        if isinstance(solution.feasibility, MilpFeasibility):
            if solution.voyage_plan is None:
                return
            elif isinstance(solution.voyage_plan, list):
                if all([isinstance(island_id, (int, np.integer)) for island_id in solution.voyage_plan]):
                    return
    raise SanityCheckError(f"The output {solution!r} of {solve_optimization.__name__}"
                            " is not respecting the required types.")

def ex07_evaluation(
    algo_in: TestMilp, expected_out: Any, visualizer: Viz = Viz()
) -> Tuple[MilpPerformance, Report]:

    algo_in_type = algo_in.type
    algo_in_optimization_cost = algo_in.optimization_cost
    algo_in_probem = algo_in.problem
    algo_in_seed = algo_in.seed
    algo_in_timeout = algo_in.timeout

    title = algo_in.str_id()
    r = Report(title)
    visualizer.print_title(title)

    if algo_in_type == MilpCase.test_voyage:
        problem: ProblemVoyage = algo_in_probem
        gt_optimal_cost: ProblemSolution = expected_out
    elif algo_in_type == MilpCase.random_voyage:
        # individual probability of each constraint to be active
        p_constraints = Constraints(0.5,0.5,0.5,0.5,0.5)
        problem = milp_generator(algo_in_seed, algo_in_optimization_cost, p_constraints)
        gt_optimal_cost = None
    else:
        raise ValueError(algo_in_type)

    start = timeit.default_timer()
    est_solution = solve_optimization(problem)
    stop = timeit.default_timer()
    timing = stop-start

    sanity_check(est_solution)

    feasibility_score = compute_feasibility_score(est_solution, gt_optimal_cost)
    violations, feasibility_score = compute_violations(problem, est_solution, feasibility_score)
    est_cost = compute_cost(problem, est_solution)
    cost_score = compute_cost_score(problem.optimization_cost, est_cost, gt_optimal_cost, feasibility_score, violations)

    if timing > algo_in_timeout:
        raise TestCaseTimeoutException(f"Exceeded test case timeout: {algo_in_timeout} seconds.")
    else:
        timing = None
        performance = MilpPerformance(feasibility_score, violations, cost_score)

    visualizer.visualize(r, problem, feasibility_score, gt_optimal_cost, cost_score, est_solution.voyage_plan, est_cost, violations, timing)

    return performance, r


def get_exercise7() -> Exercise:

    test_type = MilpCase.random_voyage
    
    test_values = []
    expected_results = []

    if test_type == MilpCase.test_voyage:
        path = pathlib.Path(__file__).parent.resolve()
        with open(f"{path}/local_tests_GT.pkl", 'rb') as f:
            database_tests = pickle.load(f)
            for opt_cost in database_tests:
                for test_id in database_tests[opt_cost]:
                    test_value = database_tests[opt_cost][test_id]["test_value"]
                    expected_result = database_tests[opt_cost][test_id]["expected_result"]
                    test_values.append(test_value)
                    expected_results.append(expected_result)
    
    elif test_type == MilpCase.random_voyage:

        seed = 0
        n_tests = 3
        timeout = 20

        for test_cost in OptimizationCost.get_costs():

            seed = int((seed**1.4-seed**1.3)**0.5+seed)+1
            if seed > 2**25:
                seed = int(seed/2**18)

            for test_id in range(n_tests):
                
                test_seed = min(seed*(1+(test_id*2)), 2**31)

                test_values.append(TestMilp(
                    test_type,
                    test_cost,
                    test_id,
                    None,
                    test_seed,
                    timeout
                ))
                expected_results.append(None)

    else:
        raise ValueError(test_type)

    return Exercise[TestMilp, Any](
        desc="This exercise solves MILPs.",
        evaluation_fun= ex07_evaluation,
        perf_aggregator= lambda x: ex07_performance_aggregator(x, test_type),
        test_values=test_values,
        expected_results=expected_results,
        test_case_timeout = 40
    )

from dataclasses import dataclass
from time import process_time
from typing import Any, List, Sequence, Type
from zuper_commons.text import remove_escapes

import numpy as np
from matplotlib import pyplot as plt
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.policy_iteration import PolicyIteration
from pdm4ar.exercises.ex04.value_iteration import ValueIteration
from pdm4ar.exercises_def import Exercise, ExIn
from pdm4ar.exercises_def.ex04.data import get_expected_results, get_test_grids
from pdm4ar.exercises_def.ex04.map import map2image
from pdm4ar.exercises_def.ex04.utils import action2arrow, head_width
from pdm4ar.exercises_def.structures import PerformanceResults
from reprep import MIME_PDF, Report


@dataclass
class TestValueEx4(ExIn):
    algo: Type[GridMdpSolver]
    grid_list: List[GridMdp]

    def str_id(self) -> str:
        return str(self.algo.__name__)

@dataclass(frozen=True)
class Ex04PerformanceResult(PerformanceResults):
    policy_accuracy: float 
    value_func_mspa: float   # mean squared percentage accuracy of value function
    solve_time: float

    def __post__init__(self):
        assert self.policy_accuracy <= 1, self.policy_accuracy
        assert self.solve_time >= 0, self.solve_time


def ex4_evaluation(ex_in: TestValueEx4, ex_out=None) -> Report:
    grid_mdp_list = ex_in.grid_list
    solver: GridMdpSolver = ex_in.algo()
    algo_name = solver.__class__.__name__
    r = Report(f"Ex4-{algo_name}")

    solve_time_list = []
    policy_accuracy_list = []
    value_func_mspa_list = []
    
    for k, grid_mdp in enumerate(grid_mdp_list):
        t = process_time()
        value_func, policy = solver.solve(grid_mdp)
        solve_time = process_time() - t

        MAP_SHAPE = grid_mdp.grid.shape
        font_size = 3 if MAP_SHAPE[0] > 15 else 6

        rfig = r.figure(cols=2)
        with rfig.plot(nid=f"{algo_name}-value-{k}", mime=MIME_PDF, figsize=None) as _:
            ax = plt.gca()
            ax.imshow(value_func, aspect="equal")
            ax.tick_params(axis="both", labelsize=font_size + 3)
            for i in range(MAP_SHAPE[0]):
                for j in range(MAP_SHAPE[1]):
                    ax.text(j, i, f"{value_func[i, j]:.1f}", size=font_size, ha="center", va="center", color="k")

        map_c = map2image(grid_mdp.grid)
        with rfig.plot(nid=f"{algo_name}-policy-{k}", mime=MIME_PDF, figsize=None) as _:
            ax = plt.gca()
            ax.imshow(map_c, aspect="equal")
            ax.tick_params(axis="both", labelsize=font_size + 3)
            for i in range(MAP_SHAPE[0]):
                for j in range(MAP_SHAPE[1]):
                    arrow = action2arrow[policy[i, j]]
                    ax.arrow(j, i, arrow[1], arrow[0], head_width=head_width, color="k")

        if ex_out is not None:
            value_func_gt, policy_gt = ex_out[k]
            # evaluate accuracy
            policy_accuracy = np.sum(policy_gt==policy) / policy_gt.size
            value_func_mspa = 1 - np.sum(np.abs(np.divide(value_func_gt-value_func, value_func_gt))) / value_func_gt.size

            policy_accuracy_list.append(policy_accuracy)
            value_func_mspa_list.append(value_func_mspa)
            solve_time_list.append(solve_time)

            # plot ground truth
            rfig = r.figure(cols=2)
            with rfig.plot(nid=f"GroundTruth-value-{k}", mime=MIME_PDF, figsize=None) as _:
                ax = plt.gca()
                ax.imshow(value_func_gt, aspect="equal")
                ax.tick_params(axis="both", labelsize=font_size + 3)
                for i in range(MAP_SHAPE[0]):
                    for j in range(MAP_SHAPE[1]):
                        ax.text(j, i, f"{value_func_gt[i, j]:.1f}", size=font_size, ha="center", va="center", color="k")

            map_c = map2image(grid_mdp.grid)
            with rfig.plot(nid=f"GroundTruth-policy-{k}", mime=MIME_PDF, figsize=None) as _:
                ax = plt.gca()
                ax.imshow(map_c, aspect="equal")
                ax.tick_params(axis="both", labelsize=font_size + 3)
                for i in range(MAP_SHAPE[0]):
                    for j in range(MAP_SHAPE[1]):
                        arrow = action2arrow[policy_gt[i, j]]
                        ax.arrow(j, i, arrow[1], arrow[0], head_width=head_width, color="k")

            msg = f"policy_accuracy: {policy_accuracy}\n"
            msg += f"value_func_mspa:{value_func_mspa:.3f}\n"

            r.text(f"{algo_name}-query{k}", text=remove_escapes(msg))

    # aggregate performance of each query
    query_perf = list(map(Ex04PerformanceResult, policy_accuracy_list, value_func_mspa_list, solve_time_list))
    perf = ex4_perf_aggregator(query_perf)
    return perf, r


def ex4_perf_aggregator(perf: Sequence[Ex04PerformanceResult]) -> Ex04PerformanceResult:
    # perfomance for valid results
    policy_accuracy = [p.policy_accuracy for p in perf]
    value_func_mspa = [p.value_func_mspa for p in perf]
    solve_time = [p.solve_time for p in perf]
    try:
        # average accuracy and solve_time
        avg_policy_accuracy = sum(policy_accuracy) / float(len(policy_accuracy))
        avg_value_func_mspa = sum(value_func_mspa) / float(len(value_func_mspa)) 
        avg_solve_time = sum(solve_time) / float(len(solve_time))
    except ZeroDivisionError:
        # None if gt wasn't provided
        avg_policy_accuracy = 0
        avg_value_func_mspa = 0
        avg_solve_time = 0

    return Ex04PerformanceResult(policy_accuracy=avg_policy_accuracy,
                                 value_func_mspa=avg_value_func_mspa,
                                 solve_time=avg_solve_time)


def get_exercise4() -> Exercise:
    grid_mdp_list = get_test_grids()
    test_values = [TestValueEx4(algo=ValueIteration, grid_list=grid_mdp_list),
                   TestValueEx4(algo=PolicyIteration, grid_list=grid_mdp_list)]

    expected_results = get_expected_results()

    return Exercise[TestValueEx4, Any](
            desc='This exercise is about dynamic programming',
            evaluation_fun=ex4_evaluation,
            perf_aggregator=ex4_perf_aggregator,
            test_values=test_values,
            expected_results=expected_results,
    )

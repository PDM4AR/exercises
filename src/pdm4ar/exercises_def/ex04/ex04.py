from dataclasses import dataclass
from time import process_time
from typing import Any, Sequence, Type
from zuper_commons.text import remove_escapes

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.policy_iteration import PolicyIteration
from pdm4ar.exercises.ex04.value_iteration import ValueIteration
from pdm4ar.exercises.ex04.structures import Action, AllOptimalActions, Cell
from pdm4ar.exercises_def import Exercise, ExIn
from pdm4ar.exercises_def.ex04.data import get_expected_results, get_test_grids
from pdm4ar.exercises_def.ex04.map import map2image
from pdm4ar.exercises_def.ex04.utils import action2arrow, head_width
from pdm4ar.exercises_def.structures import PerformanceResults
from reprep import MIME_PDF, Report


@dataclass
class TestValueEx4(ExIn):
    algo: Type[GridMdpSolver]
    grid: GridMdp
    testId: int = 0

    def str_id(self) -> str:
        return str(self.algo.__name__) + str(self.testId)


@dataclass(frozen=True)
class Ex04Performance(PerformanceResults):
    policy_accuracy: float
    value_func_r2: float
    solve_time: float

    def __post__init__(self):
        assert self.policy_accuracy <= 1, self.policy_accuracy
        assert self.solve_time >= 0, self.solve_time


@dataclass(frozen=True)
class Ex04PerformanceResult(PerformanceResults):
    # All test cases
    perf_result: Ex04Performance
    # Value Iteration test cases
    value_iteration: Ex04Performance = None
    # Policy Iteration test cases
    policy_iteration: Ex04Performance = None


def get_font_size(grid_mdp: GridMdp) -> int:
    num_row = grid_mdp.grid.shape[0]
    if num_row <= 15:
        return 6
    elif num_row <= 30:
        return 3
    else:
        return 2


def plot_grid_values(rfig, grid_mdp: GridMdp, value_func: np.ndarray, algo_name: str):
    MAP_SHAPE = grid_mdp.grid.shape
    font_size = get_font_size(grid_mdp)
    with rfig.plot(nid=f"{algo_name}-value", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        ax.imshow(value_func, aspect="equal")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=font_size + 3)
        for i in range(MAP_SHAPE[0]):
            for j in range(MAP_SHAPE[1]):
                if grid_mdp.grid[i, j] == Cell.CLIFF:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="k"))
                else:
                    ax.text(j, i, f"{value_func[i, j]:.1f}", size=font_size, ha="center", va="center", color="k")


def plot_grid_policy(rfig, grid_mdp: GridMdp, policy: AllOptimalActions, algo_name: str):
    MAP_SHAPE = grid_mdp.grid.shape
    font_size = get_font_size(grid_mdp)
    map_c = map2image(grid_mdp.grid)
    with rfig.plot(nid=f"{algo_name}-policy", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        ax.imshow(map_c, aspect="equal")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=font_size + 3)
        for i in range(MAP_SHAPE[0]):
            for j in range(MAP_SHAPE[1]):
                # Skip cliff
                if grid_mdp.grid[i, j] == Cell.CLIFF:
                    continue
                # Get optimal actions. If policy is a single action, convert it to a list
                if policy.dtype == object:
                    optimal_actions = policy[i, j]
                elif policy.dtype == int:
                    optimal_actions = [policy[i, j]]
                else:
                    raise ValueError("Invalid policy type")

                for action in optimal_actions:
                    if action == Action.ABANDON:
                        ax.text(j, i, "X", size=2.5 * font_size, ha="center", va="center", color="k", weight="bold")
                    else:
                        arrow = action2arrow[action]
                        ax.arrow(j, i, arrow[1], arrow[0], head_width=head_width, color="k")


def plot_report_figure(r: Report, grid_mdp: GridMdp, value_func: np.ndarray, policy: AllOptimalActions, algo_name: str):
    rfig = r.figure(cols=2)
    plot_grid_values(rfig, grid_mdp, value_func, algo_name)
    plot_grid_policy(rfig, grid_mdp, policy, algo_name)


def ex4_evaluation(ex_in: TestValueEx4, ex_out=None) -> Report:
    grid_mdp = ex_in.grid
    solver: GridMdpSolver = ex_in.algo()
    algo_name = ex_in.str_id()
    r = Report(f"Ex4-{algo_name}")

    t = process_time()
    value_func, policy = solver.solve(grid_mdp)
    solve_time = process_time() - t
    plot_report_figure(r, grid_mdp, value_func, policy, algo_name)

    if ex_out is not None:
        all_states_mask = grid_mdp.grid != Cell.CLIFF
        # ground truth
        value_func_gt, policy_gt = ex_out
        # evaluate accuracy
        if policy_gt.dtype == int:  # policy_gt only contains single optimal action per state
            policy_accuracy = (
                np.sum(policy_gt[all_states_mask] == policy[all_states_mask]) / policy_gt[all_states_mask].size
            )
        elif policy_gt.dtype == object:  # policy_gt contains all optimal actions per state
            correct_policy = 0
            for user_policy, gt_policy in zip(policy[all_states_mask], policy_gt[all_states_mask]):
                correct_policy += 1 if user_policy in gt_policy else 0
            policy_accuracy = float(correct_policy) / policy_gt[all_states_mask].size
        else:
            raise ValueError("Invalid policy_gt type")

        # R2 score - sum of squared errors divided by sum of squared differences from the mean
        value_func_r2 = 1 - np.sum(np.square(value_func_gt[all_states_mask] - value_func[all_states_mask])) / np.sum(
            np.square(value_func_gt[all_states_mask] - np.mean(value_func_gt[all_states_mask]))
        )
        # Clamp negative values to 0
        value_func_r2 = max(0, value_func_r2)

        # plot ground truth
        plot_report_figure(r, grid_mdp, value_func_gt, policy_gt, "GroundTruth")

        msg = f"policy_accuracy: {policy_accuracy}\n"
        msg += f"value_func_r2:{value_func_r2:.3f}\n"

        r.text(f"{algo_name}", text=remove_escapes(msg))

    result = Ex04Performance(policy_accuracy=policy_accuracy, value_func_r2=value_func_r2, solve_time=solve_time)
    perf = Ex04PerformanceResult(perf_result=result)
    if isinstance(solver, PolicyIteration):
        perf = Ex04PerformanceResult(perf_result=result, policy_iteration=result)
    elif isinstance(solver, ValueIteration):
        perf = Ex04PerformanceResult(perf_result=result, value_iteration=result)
    return perf, r


def ex4_single_perf_aggregator(perf: Sequence[Ex04Performance]) -> Ex04Performance:
    # perfomance for valid results
    policy_accuracy = [p.policy_accuracy for p in perf]
    value_func_r2 = [p.value_func_r2 for p in perf]
    solve_time = [p.solve_time for p in perf]
    try:
        # average accuracy and solve_time, rounding to 3 decimal places
        avg_policy_accuracy = round(np.mean(policy_accuracy), 3)
        avg_value_func_r2 = round(np.mean(value_func_r2), 3)
        avg_solve_time = round(np.mean(solve_time), 3)
    except ZeroDivisionError:
        # None if gt wasn't provided
        avg_policy_accuracy = 0
        avg_value_func_r2 = 0
        avg_solve_time = 0

    return Ex04Performance(
        policy_accuracy=avg_policy_accuracy, value_func_r2=avg_value_func_r2, solve_time=avg_solve_time
    )


def ex4_perf_aggregator(perf: Sequence[Ex04PerformanceResult]) -> Ex04PerformanceResult:
    perf_result = [p.perf_result for p in perf]
    policy_iteration = [p.policy_iteration for p in perf if p.policy_iteration is not None]
    value_iteration = [p.value_iteration for p in perf if p.value_iteration is not None]

    return Ex04PerformanceResult(
        perf_result=ex4_single_perf_aggregator(perf_result),
        policy_iteration=ex4_single_perf_aggregator(policy_iteration),
        value_iteration=ex4_single_perf_aggregator(value_iteration),
    )


def get_exercise4() -> Exercise:
    algos = [ValueIteration, PolicyIteration]
    grid_mdps = get_test_grids()
    test_values = [
        TestValueEx4(algo=algo, grid=grid_mdp, testId=i) for algo in algos for i, grid_mdp in enumerate(grid_mdps)
    ]
    expected_results = get_expected_results()

    return Exercise[TestValueEx4, Any](
        desc="This exercise is about dynamic programming",
        evaluation_fun=ex4_evaluation,
        perf_aggregator=ex4_perf_aggregator,
        test_values=test_values,
        expected_results=expected_results,
    )

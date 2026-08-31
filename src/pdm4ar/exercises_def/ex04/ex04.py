from dataclasses import dataclass
from time import process_time
from typing import Any, Sequence, Type, Union, Optional, cast
from zuper_commons.text import remove_escapes

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from pdm4ar.exercises.ex04.mdp import AugmentedGridMdp, GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.policy_iteration import PolicyIteration
from pdm4ar.exercises.ex04.value_iteration import ValueIteration
from pdm4ar.exercises.ex04.structures import Action, OptimalActions, Cell, Policy
from pdm4ar.exercises_def import Exercise, ExIn
from pdm4ar.exercises_def.ex04.data import (
    get_expected_results_algo,
    get_small_test_grids,
    get_expected_results_algo_aug,
    get_expected_results_transition,
    get_expected_results_transition_aug,
    get_test_grids,
    get_test_mdps_aug,
    get_transition_prob_test_cases,
    get_transition_prob_test_cases_aug,
    TestTransitionProbAug,
    TestTransitionProbEx4,
)
from pdm4ar.exercises_def.ex04.map import map2image
from pdm4ar.exercises_def.ex04.utils import action2arrow, head_width
from pdm4ar.exercises_def.structures import PerformanceResults
from reprep import MIME_PDF, Report

ALL_MAPS = False
"""Set to True to additionally run three smaller test maps (6x6, 9x9, 12x12,
report ids 3-5) with published solutions, on top of the three public maps."""


@dataclass
class TestValueEx4(ExIn):
    algo: Type[GridMdpSolver]
    grid: Union[GridMdp, AugmentedGridMdp]
    testId: int = 0
    case_name: str = "base"

    def str_id(self) -> str:
        return f"{self.algo.__name__}-{self.case_name}{self.testId}"

# VI results cached per (case_name, testId) so the PI run of the same MDP can
# be compared against them (automated VI/PI coincidence check).
_VI_CACHE: dict = {}
COINCIDENCE_TOL = 1e-11

AUG_Z_LABELS = {
    "momentum": ["h=-", "h=N", "h=W", "h=S", "h=E"],
    "forecast": ["CLEAR", "FOGGY"],
    "glitch": ["OK", "GLITCHY"],
}


@dataclass(frozen=True)
class Ex04Performance(PerformanceResults):
    policy_accuracy: float
    value_func_r2: float
    solve_time: float

    def __post__init__(self):
        assert self.policy_accuracy <= 1, self.policy_accuracy
        assert self.solve_time >= 0, self.solve_time


@dataclass(frozen=True)
class Ex04TransitionProbPerformance(PerformanceResults):
    transition_prob_accuracy: float

    def __post__init__(self):
        assert self.transition_prob_accuracy <= 1, self.transition_prob_accuracy


@dataclass(frozen=True)
class Ex04PerformanceResult(PerformanceResults):
    # Value Iteration test cases
    value_iteration: Optional[Ex04Performance] = None
    # Policy Iteration test cases
    policy_iteration: Optional[Ex04Performance] = None
    # Transition probability test cases
    transition_prob: Optional[Ex04TransitionProbPerformance] = None


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
        # mask CLIFF cells (not states) so the color scale spans only real
        # values and matches between the student's and the ground-truth panel;
        # the colormap renders the masked cells black (part of the raster, so
        # no seams from patch overlays)
        plot_v = np.where(grid_mdp.grid == Cell.CLIFF, np.nan, np.asarray(value_func, dtype=float))
        cmap = plt.get_cmap().copy()
        cmap.set_bad("k")
        ax.imshow(plot_v, aspect="equal", cmap=cmap)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=font_size + 3)
        for i in range(MAP_SHAPE[0]):
            for j in range(MAP_SHAPE[1]):
                if grid_mdp.grid[i, j] != Cell.CLIFF:
                    ax.text(j, i, f"{value_func[i, j]:.1f}", size=font_size, ha="center", va="center", color="k")


def plot_grid_policy(rfig, grid_mdp: GridMdp, policy: Union[OptimalActions, Policy], algo_name: str):
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
                # Skip cliff cells
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


def plot_report_figure(
    r: Report, grid_mdp: GridMdp, value_func: np.ndarray, policy: Union[OptimalActions, Policy], algo_name: str
):
    rfig = r.figure(cols=2)
    plot_grid_values(rfig, grid_mdp, value_func, algo_name)
    plot_grid_policy(rfig, grid_mdp, policy, algo_name)


def ex4_evaluation(ex_in, ex_out=None) -> tuple[PerformanceResults, Report]:
    if isinstance(ex_in, TestValueEx4):
        if ex_in.case_name == "base":
            return ex4_evaluation_algo(ex_in, ex_out)
        return ex4_evaluation_algo_aug(ex_in, ex_out)
    elif isinstance(ex_in, TestTransitionProbEx4):
        return ex4_transition_prob_evaluation(ex_in, ex_out)
    elif isinstance(ex_in, TestTransitionProbAug):
        return ex4_transition_prob_evaluation_aug(ex_in, ex_out)
    else:
        raise ValueError(f"Unknown test type: {type(ex_in)}")


def ex4_evaluation_algo(ex_in: TestValueEx4, ex_out=None) -> tuple[PerformanceResults, Report]:
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
                if gt_policy is None:
                    gt_policy = [Action.ABANDON]
                if user_policy is None:
                    user_policy = [Action.ABANDON]
                correct_policy += 1 if user_policy in gt_policy else 0
            policy_accuracy = float(correct_policy) / policy_gt[all_states_mask].size
        else:
            raise ValueError("Invalid policy_gt type")

        # R2 score - sum of squared errors divided by sum of squared differences from the mean
        value_func_r2 = 1.0 - np.sum(np.square(value_func_gt[all_states_mask] - value_func[all_states_mask])) / np.sum(
            np.square(value_func_gt[all_states_mask] - np.mean(value_func_gt[all_states_mask]))
        )
        # Clamp negative values to 0
        value_func_r2 = max(0, value_func_r2)

        # plot ground truth
        plot_report_figure(r, grid_mdp, value_func_gt, policy_gt, "GroundTruth")

        msg = f"policy_accuracy: {policy_accuracy}\n"
        msg += f"value_func_r2:{value_func_r2:.3f}\n"
        msg += _coincidence_msg(ex_in, np.asarray(value_func, dtype=float),
                                all_states_mask, solve_time)

        r.text(f"{algo_name}", text=remove_escapes(msg))

    result = Ex04Performance(policy_accuracy=policy_accuracy, value_func_r2=value_func_r2, solve_time=solve_time)
    if isinstance(solver, PolicyIteration):
        perf = Ex04PerformanceResult(policy_iteration=result)
    elif isinstance(solver, ValueIteration):
        perf = Ex04PerformanceResult(value_iteration=result)
    else:
        raise ValueError(f"Unknown solver type: {type(solver)}")
    return perf, r


def ex4_transition_prob_evaluation(ex_in: TestTransitionProbEx4, ex_out=None) -> tuple[PerformanceResults, Report]:
    grid_mdp = ex_in.grid
    test_name = ex_in.str_id()
    r = Report(f"Ex4-{test_name}")

    if ex_out is not None:
        expected_prob = ex_out
        msg = f"State: {ex_in.state}, Action: {ex_in.action.name}, Next State: {ex_in.next_state}\n"
        msg += f"Expected probability: {expected_prob}\n"
        # Get the transition probability from the implemented method
        transition_prob = grid_mdp.get_transition_prob(ex_in.state, ex_in.action, ex_in.next_state)
        
        accuracy = 1.0 if abs(transition_prob - expected_prob) < 1e-6 else 0.0
        msg += f"Computed probability: {transition_prob:.6f}\n"
        msg += f"Accuracy: {accuracy:.1f}\n"

        r.text(f"{test_name}", text=remove_escapes(msg))

    result = Ex04TransitionProbPerformance(transition_prob_accuracy=accuracy)
    return result, r


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
        policy_accuracy=float(avg_policy_accuracy),
        value_func_r2=float(avg_value_func_r2),
        solve_time=float(avg_solve_time),
    )


def ex4_transition_prob_perf_aggregator(perf: Sequence[Ex04TransitionProbPerformance]) -> Ex04TransitionProbPerformance:
    if not perf:
        return Ex04TransitionProbPerformance(transition_prob_accuracy=0.0)

    accuracy_sum = sum(p.transition_prob_accuracy for p in perf)
    avg_accuracy = accuracy_sum / len(perf) if perf else 0.0

    return Ex04TransitionProbPerformance(transition_prob_accuracy=round(avg_accuracy, 3))


def ex4_perf_aggregator(perf: Sequence[Ex04PerformanceResult | Ex04TransitionProbPerformance]) -> Ex04PerformanceResult:
    algo_results = []
    transition_prob_results = []

    for result in perf:
        if isinstance(result, Ex04PerformanceResult):
            algo_results.append(result)
        elif isinstance(result, Ex04TransitionProbPerformance):
            transition_prob_results.append(result)

    policy_iteration_results = [p.policy_iteration for p in algo_results if p.policy_iteration is not None]
    value_iteration_results = [p.value_iteration for p in algo_results if p.value_iteration is not None]

    policy_iteration_aggregated = ex4_single_perf_aggregator(policy_iteration_results)
    value_iteration_aggregated = ex4_single_perf_aggregator(value_iteration_results)
    transition_prob_aggregated = ex4_transition_prob_perf_aggregator(transition_prob_results)

    return Ex04PerformanceResult(
        policy_iteration=policy_iteration_aggregated,
        value_iteration=value_iteration_aggregated,
        transition_prob=transition_prob_aggregated,
    )




# ---------------------------------------------------------------------------
# Part 2: augmented cases
# ---------------------------------------------------------------------------
def _coincidence_msg(ex_in: TestValueEx4, value_func, mask, solve_time) -> str:
    """Cache the VI result; on the matching PI run, compare the converged
    matrices and emit a review label if they agree to bit level."""
    from pdm4ar.exercises.ex04.policy_iteration import PolicyIteration as _PI
    from pdm4ar.exercises.ex04.value_iteration import ValueIteration as _VI

    key = (ex_in.case_name, ex_in.testId)
    if issubclass(ex_in.algo, _VI):
        _VI_CACHE[key] = (value_func.copy(), solve_time)
        return ""
    if issubclass(ex_in.algo, _PI) and key in _VI_CACHE:
        v_vi, t_vi = _VI_CACHE[key]
        d = np.abs(v_vi[mask] - value_func[mask])
        max_diff = float(d.max()) if d.size else 0.0
        ratio = t_vi / solve_time if solve_time > 0 else float("inf")
        if max_diff < COINCIDENCE_TOL:
            return (f"LABEL vi_pi_identical_suspected: the two submitted "
                    f"value functions agree to {max_diff:.1e} (bit level; no "
                    f"honest tolerance explains this). solve_time ratio "
                    f"VI/PI = {ratio:.2f}. Please review manually.\n")
        return f"vi_pi_coincidence_check: clean (max diff {max_diff:.1e})\n"
    return ""


def _plot_aug_slice_values(rfig, mdp, value_slice, title: str):
    font_size = get_font_size(mdp)
    with rfig.plot(nid=f"{title}-value", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        # mask CLIFF cells (not states) so the color scale spans only real
        # values and matches between the student's and the ground-truth panels;
        # the colormap renders the masked cells black (no patch-overlay seams)
        plot_v = np.where(mdp.grid == Cell.CLIFF, np.nan, np.asarray(value_slice, dtype=float))
        cmap = plt.get_cmap().copy()
        cmap.set_bad("k")
        ax.imshow(plot_v, aspect="equal", cmap=cmap)
        ax.tick_params(axis="both", labelsize=font_size + 3)
        ax.set_title(title, fontsize=font_size + 4)
        for i in range(value_slice.shape[0]):
            for j in range(value_slice.shape[1]):
                if mdp.grid[i, j] != Cell.CLIFF and np.isfinite(value_slice[i, j]):
                    ax.text(j, i, f"{value_slice[i, j]:.1f}", size=font_size,
                            ha="center", va="center", color="k")


def _plot_aug_slice_policy(rfig, mdp, policy_slice, title: str):
    font_size = get_font_size(mdp)
    map_img = map2image(mdp.grid)
    with rfig.plot(nid=f"{title}-policy", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        ax.imshow(map_img, aspect="equal")
        ax.tick_params(axis="both", labelsize=font_size + 3)
        ax.set_title(title, fontsize=font_size + 4)
        for i in range(policy_slice.shape[0]):
            for j in range(policy_slice.shape[1]):
                # Skip cliff cells, as in the Part-1 plots
                if mdp.grid[i, j] == Cell.CLIFF:
                    continue
                a = policy_slice[i, j]
                if a < 0:
                    continue
                a = Action(a)
                if a == Action.ABANDON:
                    ax.text(j, i, "X", size=font_size + 2, ha="center",
                            va="center", color="k")
                elif a == Action.STAY:
                    ax.text(j, i, "G", size=font_size + 2, ha="center",
                            va="center", color="k")
                else:
                    arrow = action2arrow[a]
                    ax.arrow(j, i, arrow[1], arrow[0], head_width=head_width,
                             color="k")


def _plot_aug_all_slices(r: Report, mdp, case_name: str, value_func, policy,
                         algo_name: str):
    labels = AUG_Z_LABELS[case_name]
    rfig = r.figure(cols=2)
    for zi in range(mdp.Z):
        _plot_aug_slice_values(rfig, mdp, value_func[:, :, zi],
                               f"{algo_name}-{labels[zi]}")
        _plot_aug_slice_policy(rfig, mdp, policy[:, :, zi],
                               f"{algo_name}-{labels[zi]}")


def ex4_evaluation_algo_aug(ex_in: TestValueEx4, ex_out=None) -> tuple[PerformanceResults, Report]:
    mdp = ex_in.grid
    solver: GridMdpSolver = ex_in.algo()
    algo_name = ex_in.str_id()
    r = Report(f"Ex4-{algo_name}")

    t = process_time()
    value_func, policy = solver.solve(mdp)
    solve_time = process_time() - t
    value_func = np.asarray(value_func, dtype=float)
    policy = np.asarray(policy)
    _plot_aug_all_slices(r, mdp, ex_in.case_name, value_func, policy, algo_name)

    policy_accuracy, value_func_r2 = 0.0, 0.0
    if ex_out is not None:
        value_func_gt, policy_gt = ex_out
        mask2 = mdp.grid != Cell.CLIFF
        mask3 = np.repeat(mask2[:, :, None], mdp.Z, axis=2)

        correct = 0
        for user_a, gt_list in zip(policy[mask3], policy_gt[mask3]):
            if gt_list is None or int(user_a) in gt_list:
                correct += 1
        policy_accuracy = float(correct) / policy_gt[mask3].size

        value_func_r2 = 1.0 - np.sum(
            np.square(value_func_gt[mask3] - value_func[mask3])
        ) / np.sum(
            np.square(value_func_gt[mask3] - np.mean(value_func_gt[mask3]))
        )
        value_func_r2 = max(0.0, float(value_func_r2))

        plot_gt_policy = np.full(policy_gt.shape, -1, dtype=int)
        for idx, gt_list in np.ndenumerate(policy_gt):
            if gt_list:
                plot_gt_policy[idx] = gt_list[0]
        _plot_aug_all_slices(r, mdp, ex_in.case_name, value_func_gt,
                             plot_gt_policy, "GroundTruth")

        msg = f"policy_accuracy: {policy_accuracy}\n"
        msg += f"value_func_r2:{value_func_r2:.3f}\n"
        msg += _coincidence_msg(ex_in, value_func, mask3, solve_time)
        r.text(f"{algo_name}", text=remove_escapes(msg))

    result = Ex04Performance(policy_accuracy=policy_accuracy,
                             value_func_r2=value_func_r2,
                             solve_time=solve_time)
    if isinstance(solver, PolicyIteration):
        return Ex04PerformanceResult(policy_iteration=result), r
    elif isinstance(solver, ValueIteration):
        return Ex04PerformanceResult(value_iteration=result), r
    raise ValueError(f"Unknown solver type: {type(solver)}")


def ex4_transition_prob_evaluation_aug(ex_in: TestTransitionProbAug, ex_out=None) -> tuple[PerformanceResults, Report]:
    test_name = ex_in.str_id()
    r = Report(f"Ex4-{test_name}")
    accuracy = 0.0
    if ex_out is not None:
        expected_prob = ex_out
        msg = (f"Case: {ex_in.case_name}, State: {ex_in.state}, "
               f"Action: {ex_in.action.name}, Next state: {ex_in.next_state}\n")
        msg += f"Expected probability: {expected_prob}\n"
        transition_prob = ex_in.mdp.get_transition_prob(
            ex_in.state, ex_in.action, ex_in.next_state)
        accuracy = 1.0 if abs(transition_prob - expected_prob) < 1e-6 else 0.0
        msg += f"Computed probability: {transition_prob:.6f}\n"
        msg += f"Accuracy: {accuracy:.1f}\n"
        r.text(f"{test_name}", text=remove_escapes(msg))
    return Ex04TransitionProbPerformance(transition_prob_accuracy=accuracy), r


def get_exercise4(all_maps: Optional[bool] = None) -> Exercise:
    algos = [ValueIteration, PolicyIteration]
    if all_maps is None:
        all_maps = ALL_MAPS
    map_ids = (0, 1, 2, 3, 4, 5) if all_maps else (0, 1, 2)
    grid_mdps = get_test_grids() + (get_small_test_grids() if all_maps else [])

    # Part 1: the base MDP
    test_values_algo = [
        TestValueEx4(algo=algo, grid=grid_mdp, testId=mi)
        for algo in algos
        for mi, grid_mdp in zip(map_ids, grid_mdps)
    ]
    expected_results_algo = get_expected_results_algo(map_ids)

    # transition probes always run on the 5x5 example map (fast)
    transition_test_cases = get_transition_prob_test_cases(get_test_grids()[:1])
    transition_expected_results = get_expected_results_transition(transition_test_cases)

    # Part 2: the augmented cases on the same maps
    aug_mdps = get_test_mdps_aug(map_ids)
    test_values_aug = [
        TestValueEx4(algo=algo, grid=mdp, testId=mi, case_name=case_name)
        for algo in algos
        for (case_name, mi, mdp) in aug_mdps
    ]
    expected_results_aug = get_expected_results_algo_aug(map_ids)
    aug_transition_cases = get_transition_prob_test_cases_aug()
    aug_transition_expected = get_expected_results_transition_aug()

    all_test_values = (
        transition_test_cases + aug_transition_cases + test_values_algo + test_values_aug
    )
    all_expected_results = (
        transition_expected_results + aug_transition_expected + expected_results_algo + expected_results_aug
    )

    return Exercise[Any, Any](
        desc="Dynamic programming: the base MDP plus three augmented-state cases",
        evaluation_fun=ex4_evaluation,
        perf_aggregator=cast(Any, ex4_perf_aggregator),
        test_values=all_test_values,
        expected_results=all_expected_results,
    )

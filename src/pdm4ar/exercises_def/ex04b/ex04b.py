from dataclasses import dataclass
from time import process_time
from typing import Any, Optional, Sequence, Type, Union, cast

import numpy as np
from matplotlib import pyplot as plt
from zuper_commons.text import remove_escapes

from pdm4ar.exercises.ex04.structures import Action, Cell
from pdm4ar.exercises.ex04b.mdp import AugmentedGridMdp, GridMdpSolver
from pdm4ar.exercises.ex04b.policy_iteration import PolicyIteration
from pdm4ar.exercises.ex04b.value_iteration import ValueIteration
from pdm4ar.exercises_def import Exercise, ExIn
from pdm4ar.exercises_def.ex04.ex04 import get_font_size
from pdm4ar.exercises_def.ex04.map import map2image
from pdm4ar.exercises_def.ex04.utils import action2arrow, head_width
from pdm4ar.exercises_def.ex04b.data import (TestTransitionProbEx4b,
                                             get_expected_results_algo_ex04b,
                                             get_expected_results_transition_ex04b,
                                             get_test_mdps_ex04b,
                                             get_transition_prob_test_cases_ex04b)
from pdm4ar.exercises_def.structures import PerformanceResults
from reprep import MIME_PDF, Report

Z_LABELS = {
    "momentum": ["h=-", "h=N", "h=W", "h=S", "h=E"],
    "forecast": ["CLEAR", "FOGGY"],
    "glitch": ["OK", "GLITCHY"],
}

# VI results cached per (case_name, testId) so the PI run of the same MDP can
# be compared against them (the automated VI/PI coincidence check).
_VI_CACHE: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
COINCIDENCE_TOL = 1e-11


@dataclass
class TestValueEx4b(ExIn):
    algo: Type[GridMdpSolver]
    mdp: AugmentedGridMdp
    case_name: str
    testId: int = 0

    def str_id(self) -> str:
        return f"{self.algo.__name__}-{self.case_name}{self.testId}"


@dataclass(frozen=True)
class Ex04bPerformance(PerformanceResults):
    policy_accuracy: float
    value_func_r2: float
    solve_time: float

    def __post__init__(self):
        assert self.policy_accuracy <= 1, self.policy_accuracy
        assert self.solve_time >= 0, self.solve_time


@dataclass(frozen=True)
class Ex04bTransitionProbPerformance(PerformanceResults):
    transition_prob_accuracy: float


@dataclass(frozen=True)
class Ex04bPerformanceResult(PerformanceResults):
    value_iteration: Optional[Ex04bPerformance] = None
    policy_iteration: Optional[Ex04bPerformance] = None
    transition_prob: Optional[Ex04bTransitionProbPerformance] = None


def _plot_slice_values(rfig, mdp: AugmentedGridMdp, value_slice, title: str):
    font_size = get_font_size(mdp)
    with rfig.plot(nid=f"{title}-value", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        ax.imshow(value_slice, aspect="equal")
        ax.tick_params(axis="both", labelsize=font_size + 3)
        ax.set_title(title, fontsize=font_size + 4)
        for i in range(value_slice.shape[0]):
            for j in range(value_slice.shape[1]):
                if np.isfinite(value_slice[i, j]):
                    ax.text(j, i, f"{value_slice[i, j]:.1f}", size=font_size,
                            ha="center", va="center", color="k")


def _plot_slice_policy(rfig, mdp: AugmentedGridMdp, policy_slice, title: str):
    font_size = get_font_size(mdp)
    map_img = map2image(mdp.grid)
    with rfig.plot(nid=f"{title}-policy", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        ax.imshow(map_img, aspect="equal")
        ax.tick_params(axis="both", labelsize=font_size + 3)
        ax.set_title(title, fontsize=font_size + 4)
        for i in range(policy_slice.shape[0]):
            for j in range(policy_slice.shape[1]):
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


def _plot_all_slices(r: Report, mdp: AugmentedGridMdp, case_name: str,
                     value_func, policy, algo_name: str):
    labels = Z_LABELS[case_name]
    rfig = r.figure(cols=2)
    for zi in range(mdp.Z):
        _plot_slice_values(rfig, mdp, value_func[:, :, zi],
                           f"{algo_name}-{labels[zi]}")
        _plot_slice_policy(rfig, mdp, policy[:, :, zi],
                           f"{algo_name}-{labels[zi]}")


def ex4b_evaluation(ex_in: Union[TestValueEx4b, TestTransitionProbEx4b],
                    ex_out=None) -> tuple[PerformanceResults, Report]:
    if isinstance(ex_in, TestValueEx4b):
        return ex4b_evaluation_algo(ex_in, ex_out)
    elif isinstance(ex_in, TestTransitionProbEx4b):
        return ex4b_transition_prob_evaluation(ex_in, ex_out)
    raise ValueError(f"Unknown test type: {type(ex_in)}")


def ex4b_evaluation_algo(ex_in: TestValueEx4b, ex_out=None) -> tuple[PerformanceResults, Report]:
    mdp = ex_in.mdp
    solver: GridMdpSolver = ex_in.algo()
    algo_name = ex_in.str_id()
    r = Report(f"Ex4b-{algo_name}")

    t = process_time()
    value_func, policy = solver.solve(mdp)
    solve_time = process_time() - t
    value_func = np.asarray(value_func, dtype=float)
    policy = np.asarray(policy)
    _plot_all_slices(r, mdp, ex_in.case_name, value_func, policy, algo_name)

    if ex_out is not None:
        value_func_gt, policy_gt = ex_out
        mask2 = mdp.grid != Cell.CLIFF
        mask3 = np.repeat(mask2[:, :, None], mdp.Z, axis=2)

        if policy_gt.dtype == object:
            # ground truth stores ALL optimal actions per state (ex04-style),
            # so any correct tie-breaking gets full credit
            correct = 0
            for user_a, gt_list in zip(policy[mask3], policy_gt[mask3]):
                if gt_list is None:
                    correct += 1
                elif int(user_a) in gt_list:
                    correct += 1
            policy_accuracy = float(correct) / policy_gt[mask3].size
        else:
            policy_accuracy = float(
                np.sum(policy_gt[mask3] == policy[mask3]) / policy_gt[mask3].size
            )
        value_func_r2 = 1.0 - np.sum(
            np.square(value_func_gt[mask3] - value_func[mask3])
        ) / np.sum(
            np.square(value_func_gt[mask3] - np.mean(value_func_gt[mask3]))
        )
        value_func_r2 = max(0.0, float(value_func_r2))

        if policy_gt.dtype == object:
            plot_gt_policy = np.full(policy_gt.shape, -1, dtype=int)
            for idx, gt_list in np.ndenumerate(policy_gt):
                if gt_list:
                    plot_gt_policy[idx] = gt_list[0]
        else:
            plot_gt_policy = policy_gt
        _plot_all_slices(r, mdp, ex_in.case_name, value_func_gt, plot_gt_policy,
                         "GroundTruth")

        msg = f"policy_accuracy: {policy_accuracy}\n"
        msg += f"value_func_r2: {value_func_r2:.3f}\n"

        # automated VI/PI coincidence check on the converged matrices
        key = (ex_in.case_name, ex_in.testId)
        if issubclass(ex_in.algo, ValueIteration):
            _VI_CACHE[key] = (value_func.copy(), solve_time)
        elif issubclass(ex_in.algo, PolicyIteration) and key in _VI_CACHE:
            v_vi, t_vi = _VI_CACHE[key]
            d = np.abs(v_vi[mask3] - value_func[mask3])
            max_diff = float(d.max()) if d.size else 0.0
            ratio = t_vi / solve_time if solve_time > 0 else float("inf")
            if max_diff < COINCIDENCE_TOL:
                msg += (f"LABEL vi_pi_identical_suspected: the two submitted "
                        f"value functions agree to {max_diff:.1e} (bit level; "
                        f"no honest tolerance explains this). "
                        f"solve_time ratio VI/PI = {ratio:.2f}. "
                        f"Please review manually.\n")
            else:
                msg += f"vi_pi_coincidence_check: clean (max diff {max_diff:.1e})\n"

        r.text(f"{algo_name}", text=remove_escapes(msg))

    result = Ex04bPerformance(policy_accuracy=policy_accuracy,
                              value_func_r2=value_func_r2,
                              solve_time=solve_time)
    if isinstance(solver, PolicyIteration):
        perf = Ex04bPerformanceResult(policy_iteration=result)
    elif isinstance(solver, ValueIteration):
        perf = Ex04bPerformanceResult(value_iteration=result)
    else:
        raise ValueError(f"Unknown solver type: {type(solver)}")
    return perf, r


def ex4b_transition_prob_evaluation(ex_in: TestTransitionProbEx4b,
                                    ex_out=None) -> tuple[PerformanceResults, Report]:
    test_name = ex_in.str_id()
    r = Report(f"Ex4b-{test_name}")
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
    return Ex04bTransitionProbPerformance(transition_prob_accuracy=accuracy), r


def _avg(perf: Sequence[Ex04bPerformance]) -> Ex04bPerformance:
    if not perf:
        return Ex04bPerformance(policy_accuracy=0.0, value_func_r2=0.0,
                                solve_time=0.0)
    return Ex04bPerformance(
        policy_accuracy=float(round(np.mean([p.policy_accuracy for p in perf]), 3)),
        value_func_r2=float(round(np.mean([p.value_func_r2 for p in perf]), 3)),
        solve_time=float(round(np.mean([p.solve_time for p in perf]), 3)),
    )


def ex4b_perf_aggregator(
    perf: Sequence[Ex04bPerformanceResult | Ex04bTransitionProbPerformance],
) -> Ex04bPerformanceResult:
    algo_results = [p for p in perf if isinstance(p, Ex04bPerformanceResult)]
    tp_results = [p for p in perf
                  if isinstance(p, Ex04bTransitionProbPerformance)]
    tp_acc = (round(sum(p.transition_prob_accuracy for p in tp_results)
                    / len(tp_results), 3) if tp_results else 0.0)
    return Ex04bPerformanceResult(
        value_iteration=_avg([p.value_iteration for p in algo_results
                              if p.value_iteration is not None]),
        policy_iteration=_avg([p.policy_iteration for p in algo_results
                               if p.policy_iteration is not None]),
        transition_prob=Ex04bTransitionProbPerformance(
            transition_prob_accuracy=float(tp_acc)),
    )


def get_exercise4b() -> Exercise:
    algos = [ValueIteration, PolicyIteration]
    mdps = get_test_mdps_ex04b()
    test_values_algo = [
        TestValueEx4b(algo=algo, mdp=mdp, case_name=case_name, testId=mi)
        for algo in algos
        for (case_name, mi, mdp) in mdps
    ]
    expected_results_algo = get_expected_results_algo_ex04b()

    transition_test_cases = get_transition_prob_test_cases_ex04b()
    transition_expected = get_expected_results_transition_ex04b()

    all_tests: list[Union[TestTransitionProbEx4b, TestValueEx4b]] = (
        transition_test_cases + test_values_algo
    )
    all_expected = transition_expected + expected_results_algo

    return Exercise[Union[TestTransitionProbEx4b, TestValueEx4b], Any](
        desc="Dynamic programming with augmented states: momentum, forecast, glitch",
        evaluation_fun=ex4b_evaluation,
        perf_aggregator=cast(Any, ex4b_perf_aggregator),
        test_values=all_tests,
        expected_results=all_expected,
    )

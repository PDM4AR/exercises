"""Practice maps for exercise 04: extra maps with published solutions so you
can check your implementation yourself before submitting. Covers Part 1 (the
base MDP, case name "base") and the three Part-2 cases.

Usage (from anywhere in the container):

    from pdm4ar.exercises.ex04.value_iteration import ValueIteration
    from pdm4ar.exercises_def.ex04.practice import check_solution

    check_solution(ValueIteration)                       # everything
    check_solution(ValueIteration, cases=["base"])       # Part 1 only
    check_solution(ValueIteration, cases=["forecast"])   # one Part-2 case

For each practice map you get policy accuracy (against ALL optimal actions,
so your tie-breaking never costs you) and the value-function R2, computed with
exactly the formulas the graded evaluation uses. Not graded.
"""

import ast
from pathlib import Path
from typing import Optional, Sequence, Type, Union

import numpy as np

from pdm4ar.exercises.ex04.mdp import (AugmentedGridMdp, FogGridMdp,
                                       GlitchGridMdp, GridMdp, GridMdpSolver,
                                       MomentumGridMdp)
from pdm4ar.exercises.ex04.structures import Cell

_CASE_CLASSES = {
    "base": GridMdp,
    "momentum": MomentumGridMdp,
    "forecast": FogGridMdp,
    "glitch": GlitchGridMdp,
}


def _load():
    data_dir = Path(__file__).parent / "data"
    return np.load(data_dir / "practice_results.npz", allow_pickle=True)


def get_practice_data(cases: Optional[Sequence[str]] = None):
    """Yields (case_name, map_index, mdp, value_gt, policy_gt). policy_gt is
    an object array listing ALL optimal actions per state; shapes are (M, N)
    for the base case and (M, N, Z) for the augmented cases."""
    data = _load()
    meta = ast.literal_eval(str(data["meta"][0]))
    for case_name in (cases or list(_CASE_CLASSES)):
        cls = _CASE_CLASSES[case_name]
        for k in range(meta["n_maps"]):
            grid = data[f"map_{k}"]
            yield (case_name, k, cls(grid=grid, gamma=0.9),
                   data[f"{case_name}_value_{k}"],
                   data[f"{case_name}_policy_{k}"])


def score(mdp: Union[GridMdp, AugmentedGridMdp], value_func, policy,
          value_gt, policy_gt):
    """The same policy-accuracy and value-R2 the graded evaluation computes."""
    mask = mdp.grid != Cell.CLIFF
    if value_gt.ndim == 3:
        mask = np.repeat(mask[:, :, None], value_gt.shape[2], axis=2)
    value_func = np.asarray(value_func, dtype=float)
    policy = np.asarray(policy)

    correct = 0
    for user_a, gt_list in zip(policy[mask], policy_gt[mask]):
        if gt_list is None or int(user_a) in gt_list:
            correct += 1
    policy_accuracy = float(correct) / policy_gt[mask].size

    r2 = 1.0 - np.sum(np.square(value_gt[mask] - value_func[mask])) / np.sum(
        np.square(value_gt[mask] - np.mean(value_gt[mask]))
    )
    return policy_accuracy, max(0.0, float(r2))


def check_solution(solver: Type[GridMdpSolver],
                   cases: Optional[Sequence[str]] = None,
                   verbose: bool = True):
    """Run your solver on every practice map and compare with the published
    solutions. Returns {(case_name, map_index): (policy_accuracy, value_r2)}."""
    results = {}
    for case_name, k, mdp, value_gt, policy_gt in get_practice_data(cases):
        value_func, policy = solver().solve(mdp)
        acc, r2 = score(mdp, value_func, policy, value_gt, policy_gt)
        results[(case_name, k)] = (acc, r2)
        if verbose:
            shape = mdp.grid.shape
            flag = "" if acc == 1.0 and r2 > 0.999 else "   <-- check this one"
            print(f"{case_name:9s} map {k:2d} ({shape[0]}x{shape[1]}): "
                  f"policy_accuracy {acc:.3f}  value_r2 {r2:.4f}{flag}")
    if verbose and results:
        accs = [a for a, _ in results.values()]
        r2s = [r for _, r in results.values()]
        print(f"\noverall: min policy_accuracy {min(accs):.3f}, "
              f"min value_r2 {min(r2s):.4f} over {len(results)} runs")
    return results

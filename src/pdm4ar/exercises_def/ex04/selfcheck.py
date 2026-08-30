"""Make your own maps and verify your ex04 solutions without an answer key.

    from pdm4ar.exercises.ex04.mdp import FogGridMdp
    from pdm4ar.exercises.ex04.value_iteration import ValueIteration
    from pdm4ar.exercises.ex04.policy_iteration import PolicyIteration
    from pdm4ar.exercises_def.ex04.selfcheck import random_map, self_check

    grid = random_map((10, 10), seed=7)
    mdp = FogGridMdp(grid)
    v_vi, p_vi = ValueIteration.solve(mdp)
    v_pi, p_pi = PolicyIteration.solve(mdp)
    self_check(mdp, "forecast", v_vi, p_vi, v_pi)

None of these checks needs (or contains) a solver, so they work on any map you
generate. They verify NECESSARY properties: passing them all does not prove
optimality by itself; the one exact certificate, the Bellman residual, is
deliberately left for you to implement (see the handout). A wrong transition
model can self-certify, which is what the published worked examples and the
practice maps are for.
"""

from collections import deque
from typing import Optional, Sequence

import numpy as np

from pdm4ar.exercises.ex04.mdp import AugmentedGridMdp, GridMdp
from pdm4ar.exercises.ex04.structures import Action, Cell
from pdm4ar.exercises_def.ex04.map import generate_map

GAMMA_MAX_VALUE = 500.0  # 50 / (1 - 0.9)
_MOVES = {Action.NORTH: (-1, 0), Action.WEST: (0, -1),
          Action.SOUTH: (1, 0), Action.EAST: (0, 1)}


# ------------------------------------------------------------------ maps
def random_map(shape=(10, 10), n_cliff: Optional[int] = None,
               seed: Optional[int] = None) -> np.ndarray:
    """A random map. Resamples until the goal is reachable
    from the start (plain connectivity, no probabilities involved)."""
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 10**6))
    if n_cliff is None:
        n_cliff = max(2, round(0.08 * shape[0] * shape[1]))
    for s in range(seed, seed + 200):
        grid = generate_map(shape, 0.2, n_cliff=n_cliff, n_seed=s)
        if _reachable(grid):
            return grid
    raise RuntimeError("no reachable map found; try another seed/shape")


def _cells(grid):
    return [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])
            if grid[i, j] != Cell.CLIFF]


def _find(grid, cell_type):
    pos = np.argwhere(grid == cell_type)
    return tuple(pos[0]) if len(pos) else None


def _reachable(grid) -> bool:
    start, goal = _find(grid, Cell.START), _find(grid, Cell.GOAL)
    if start is None or goal is None:
        return False
    seen, frontier = {start}, deque([start])
    while frontier:
        i, j = frontier.popleft()
        if (i, j) == goal:
            return True
        for di, dj in _MOVES.values():
            n = (i + di, j + dj)
            if (0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1]
                    and grid[n] != Cell.CLIFF and n not in seen):
                seen.add(n)
                frontier.append(n)
    return False


# ------------------------------------------------------------------ checks
def _admissible(grid, cell):
    """As specified in the handout: moves that stay on the map and off
    cliffs, plus ABANDON; STAY only at the GOAL."""
    if grid[cell] == Cell.GOAL:
        return [Action.STAY]
    acts = []
    for a, (di, dj) in _MOVES.items():
        n = (cell[0] + di, cell[1] + dj)
        if (0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1]
                and grid[n] != Cell.CLIFF):
            acts.append(a)
    acts.append(Action.ABANDON)
    return acts


def _zs(mdp):
    return range(mdp.Z) if isinstance(mdp, AugmentedGridMdp) else [None]


def _support(grid, cell):
    """Cells where a correct model may put probability mass from `cell`."""
    sup = {cell, _find(grid, Cell.START)}
    for di, dj in _MOVES.values():
        n = (cell[0] + di, cell[1] + dj)
        if (0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1]
                and grid[n] != Cell.CLIFF):
            sup.add(n)
    return sup


def _prob(mdp, cell, z, a, ncell, nz):
    if z is None:
        return mdp.get_transition_prob(cell, a, ncell)
    return mdp.get_transition_prob((*cell, z), a, (*ncell, nz))


def check_model(mdp, verbose=True) -> bool:
    """Probabilities sum to 1 over the plausible support for every
    (state, action); ABANDON puts all mass on START. On small maps the sum
    runs over ALL cells, so stray mass is caught too."""
    grid = mdp.grid
    cells = _cells(grid)
    start = _find(grid, Cell.START)
    full_scan = grid.size <= 150
    worst, worst_at = 0.0, None
    abandon_ok = True
    for cell in cells:
        for z in _zs(mdp):
            for a in _admissible(grid, cell):
                targets = cells if full_scan else sorted(_support(grid, cell))
                total = sum(_prob(mdp, cell, z, a, n, nz)
                            for n in targets for nz in _zs(mdp))
                if abs(total - 1.0) > worst:
                    worst, worst_at = abs(total - 1.0), (cell, z, a)
                if a == Action.ABANDON:
                    mass = sum(_prob(mdp, cell, z, a, start, nz)
                               for nz in _zs(mdp))
                    if abs(mass - 1.0) > 1e-9:
                        abandon_ok = False
    ok = worst < 1e-9 and abandon_ok
    if verbose:
        scope = "all cells" if full_scan else "plausible support"
        print(f"  [{'ok' if worst < 1e-9 else 'FAIL'}] probabilities sum to 1 "
              f"over {scope} (worst |sum-1| = {worst:.1e} at {worst_at})")
        print(f"  [{'ok' if abandon_ok else 'FAIL'}] ABANDON sends all mass "
              f"to START")
    return ok


def check_bounds(mdp, value_func, verbose=True) -> bool:
    """V <= 500 everywhere; V = 500 exactly at the GOAL."""
    grid = mdp.grid
    goal = _find(grid, Cell.GOAL)
    mask = grid != Cell.CLIFF
    v = np.asarray(value_func, dtype=float)
    vmax = float(np.nanmax(v[mask]))
    goal_v = v[goal] if v.ndim == 2 else v[goal[0], goal[1], :]
    upper_ok = vmax <= GAMMA_MAX_VALUE + 1e-6
    goal_ok = bool(np.all(np.abs(np.asarray(goal_v) - GAMMA_MAX_VALUE) < 1e-3))
    if verbose:
        print(f"  [{'ok' if upper_ok else 'FAIL'}] V <= 500 everywhere "
              f"(max {vmax:.2f})")
        print(f"  [{'ok' if goal_ok else 'FAIL'}] V(goal) = 500")
    return upper_ok and goal_ok


def check_dominance(case_name, value_func, verbose=True) -> bool:
    """forecast: V(., CLEAR) >= V(., FOGGY); glitch: V(., OK) >= V(., GLITCHY).
    (No such simple per-cell rule holds for momentum or the base case.)"""
    if case_name not in ("forecast", "glitch"):
        return True
    v = np.asarray(value_func, dtype=float)
    bad = int(np.nansum(v[:, :, 0] < v[:, :, 1] - 1e-9))
    labels = ("CLEAR", "FOGGY") if case_name == "forecast" else ("OK", "GLITCHY")
    if verbose:
        print(f"  [{'ok' if bad == 0 else 'FAIL'}] V(., {labels[0]}) >= "
              f"V(., {labels[1]}) at every cell ({bad} violations)")
    return bad == 0


def check_vi_pi_agree(v_vi, v_pi, tol=1e-4, verbose=True) -> bool:
    """Your two algorithms must converge to the same V*."""
    d = float(np.nanmax(np.abs(np.asarray(v_vi, dtype=float)
                               - np.asarray(v_pi, dtype=float))))
    ok = d < tol
    if verbose:
        print(f"  [{'ok' if ok else 'FAIL'}] VI and PI agree "
              f"(max diff {d:.2e})")
    return ok


def simulate_policy(mdp, policy, value_func, episodes=300, horizon=150,
                    seed=0, verbose=True) -> bool:
    """Monte-Carlo consistency: roll out YOUR policy with YOUR model from
    START; the mean discounted return must be statistically compatible with
    your V(start). Catches value/policy mismatches."""
    grid = mdp.grid
    start = _find(grid, Cell.START)
    rng = np.random.default_rng(seed)
    zs = list(_zs(mdp))
    policy = np.asarray(policy)
    returns = []
    for _ in range(episodes):
        cell, z = start, (0 if zs[0] is not None else None)
        g, disc = 0.0, 1.0
        for _step in range(horizon):
            a = Action(int(policy[cell] if z is None else policy[(*cell, z)]))
            cand, probs = [], []
            for n in sorted(_support(grid, cell)):
                for nz in zs:
                    p = _prob(mdp, cell, z, a, n, nz)
                    if p > 0:
                        cand.append((n, nz))
                        probs.append(p)
            probs = np.array(probs)
            probs = probs / probs.sum()
            n, nz = cand[rng.choice(len(cand), p=probs)]
            if z is None:
                r = mdp.stage_reward(cell, a, n)
            else:
                r = mdp.stage_reward((*cell, z), a, (*n, nz))
            g += disc * r
            disc *= mdp.gamma
            cell, z = n, nz
        returns.append(g)
    mean, se = float(np.mean(returns)), float(np.std(returns) / np.sqrt(episodes))
    v0 = float(value_func[start] if zs[0] is None
               else value_func[(*start, 0)])
    ok = abs(mean - v0) <= 4 * se + 2.0
    if verbose:
        print(f"  [{'ok' if ok else 'FAIL'}] Monte-Carlo return from START = "
              f"{mean:.1f} +/- {se:.1f} vs V(start) = {v0:.1f}")
    return ok


def self_check(mdp, case_name, value_func, policy,
               v_pi=None, verbose=True, episodes=300) -> bool:
    """Run every no-answer-key check. Remember: these are necessary
    conditions; the exact certificate (the Bellman residual) is yours to
    implement, see the handout."""
    if verbose:
        print(f"self-check ({case_name}, map {mdp.grid.shape}):")
    ok = check_model(mdp, verbose)
    ok &= check_bounds(mdp, value_func, verbose)
    ok &= check_dominance(case_name, value_func, verbose)
    if v_pi is not None:
        ok &= check_vi_pi_agree(value_func, v_pi, verbose=verbose)
    ok &= simulate_policy(mdp, policy, value_func, episodes=episodes,
                          verbose=verbose)
    if verbose:
        print("  all checks passed" if ok else "  SOME CHECKS FAILED")
    return bool(ok)

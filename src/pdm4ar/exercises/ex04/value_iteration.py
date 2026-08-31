import numpy as np

from pdm4ar.exercises.ex04.mdp import AnyGridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: AnyGridMdp) -> tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        # todo implement here
        # Note: Part 1 mdps (GridMdp) expect (M, N) outputs like the arrays
        # above; Part 2 mdps (AugmentedGridMdp, with a Z attribute) expect
        # (M, N, Z), so adapt the shapes accordingly.

        return value_func, policy

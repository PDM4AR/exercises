import numpy as np

from pdm4ar.exercises.ex04.mdp import AnyGridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: AnyGridMdp) -> tuple[ValueFunc, Policy]:
        # Part 1 mdps (GridMdp) expect (M, N) outputs; Part 2 mdps
        # (AugmentedGridMdp, with a Z attribute) expect (M, N, Z)
        shape = grid_mdp.grid.shape
        if hasattr(grid_mdp, "Z"):
            shape = (*shape, grid_mdp.Z)
        value_func = np.zeros(shape, dtype=float)
        policy = np.zeros(shape, dtype=int)

        # todo implement here

        return value_func, policy

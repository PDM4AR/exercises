import numpy as np
from typing import Optional

from pdm4ar.exercises.ex04.mdp import AnyGridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: AnyGridMdp, max_iters: Optional[int] = None) -> tuple[ValueFunc, Policy]:
        # todo implement here
        # Part 1 mdps (GridMdp) expect (M, N) outputs; Part 2 mdps
        # (AugmentedGridMdp, with a Z attribute) expect (M, N, Z).
        # If max_iters is given, return the value function and greedy policy
        # after exactly max_iters synchronous sweeps starting from V = 0.
        raise NotImplementedError("implement ValueIteration.solve")

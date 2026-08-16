from typing import Optional

from pdm4ar.exercises.ex04b.mdp import AugmentedGridMdp, GridMdpSolver
from pdm4ar.exercises.ex04b.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: AugmentedGridMdp, max_iters: Optional[int] = None) -> tuple[ValueFunc, Policy]:
        # todo implement here
        # Return (M, N, Z) arrays; Z and the z ordering are fixed per case,
        # see structures.py and the handout. If max_iters is given, return the
        # value function and greedy policy after exactly max_iters synchronous
        # sweeps starting from V = 0.
        raise NotImplementedError("implement ValueIteration.solve for ex04b")

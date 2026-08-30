import numpy as np

from pdm4ar.exercises.ex04.mdp import AnyGridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: AnyGridMdp) -> tuple[ValueFunc, Policy]:
        # todo implement here
        # Part 1 mdps (GridMdp) expect (M, N) outputs; Part 2 mdps
        # (AugmentedGridMdp, with a Z attribute) expect (M, N, Z).
        raise NotImplementedError("implement ValueIteration.solve")

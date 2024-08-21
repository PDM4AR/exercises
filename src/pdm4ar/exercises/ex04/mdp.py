from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc


class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        pass

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        # todo
        pass


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        pass

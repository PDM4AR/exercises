from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, AugmentedState, Policy, State, ValueFunc


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


# Part 2: augmented states. Everything behaves as in Part 1; see the
# handout for the z of each case.
class AugmentedGridMdp:
    Z: int = 1
    """Number of values the z component takes (fixed per case)."""

    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""


class FogGridMdp(AugmentedGridMdp):
    """Case 1: each hour the base radios a fog forecast for the coming hour
    (z = Fog; forecasts are iid each hour)."""

    Z = 2

    def get_transition_prob(self, state: AugmentedState, action: Action, next_state: AugmentedState) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        pass

    def stage_reward(self, state: AugmentedState, action: Action, next_state: AugmentedState) -> float:
        # todo
        pass


class MomentumGridMdp(AugmentedGridMdp):
    """Case 2: slips lean toward the direction the robot moved last hour
    (z = Heading; see the handout table)."""

    Z = 5

    def get_transition_prob(self, state: AugmentedState, action: Action, next_state: AugmentedState) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        pass

    def stage_reward(self, state: AugmentedState, action: Action, next_state: AugmentedState) -> float:
        # todo
        pass


class GlitchGridMdp(AugmentedGridMdp):
    """Case 3: the wheels sometimes glitch and it persists (z = Gear;
    a freshly deployed robot has new wheels)."""

    Z = 2

    def get_transition_prob(self, state: AugmentedState, action: Action, next_state: AugmentedState) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        pass

    def stage_reward(self, state: AugmentedState, action: Action, next_state: AugmentedState) -> float:
        # todo
        pass


AnyGridMdp = Union[GridMdp, AugmentedGridMdp]


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: AnyGridMdp) -> tuple[ValueFunc, Policy]:
        pass

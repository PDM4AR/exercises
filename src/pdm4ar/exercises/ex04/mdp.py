from abc import ABC, abstractmethod
from typing import Optional, Union

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


# --------------------------------------------------------------------------
# Part 2: augmented states. Movement, swamps, cliffs, breakdowns, and all
# rewards behave exactly as in Part 1; see the handout for the z per case.
# --------------------------------------------------------------------------
class AugmentedGridMdp:
    Z: int = 1
    """Number of values the z component takes (fixed per case)."""

    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""


class MomentumGridMdp(AugmentedGridMdp):
    """Case 1: slips lean toward the direction the robot moved last hour
    (z = Heading; see the handout table)."""

    Z = 5

    def get_transition_prob(self, state: AugmentedState, action: Action, next_state: AugmentedState) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        pass

    def stage_reward(self, state: AugmentedState, action: Action, next_state: AugmentedState) -> float:
        # todo
        pass


class FogGridMdp(AugmentedGridMdp):
    """Case 2: each hour the base radios a fog forecast for the coming hour
    (z = Fog; forecasts iid with P_FOGGY)."""

    Z = 2

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
    def solve(grid_mdp: AnyGridMdp, max_iters: Optional[int] = None) -> tuple[ValueFunc, Policy]:
        """Part 1 mdps expect (M, N) outputs, Part 2 mdps (M, N, Z).
        If max_iters is given, return the state of your algorithm after
        exactly max_iters iterations (synchronous sweeps for VI; full
        evaluate-improve cycles for PI), starting from V = 0 and, for PI,
        from the first admissible action per state, ties broken by action
        order."""
        pass

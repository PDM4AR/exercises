from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pdm4ar.exercises.ex04b.structures import Action, Policy, State, ValueFunc


class AugmentedGridMdp:
    """Base for the three ex04b MDPs. The state is (i, j, z); see the handout
    (docs/04b-dynamicprogramming2.md) for the meaning of z per case and the
    exact probabilities. Movement, swamps, cliffs, breakdowns, and all rewards
    behave exactly as in Exercise 04; the maps contain no WONDERLAND cells."""

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

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        pass

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        # todo
        pass


class FogGridMdp(AugmentedGridMdp):
    """Case 2: each hour the base radios a fog forecast for the coming hour
    (z = Fog; forecasts iid with P_FOGGY)."""

    Z = 2

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo
        pass

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        # todo
        pass


class GlitchGridMdp(AugmentedGridMdp):
    """Case 3: the wheels sometimes glitch and it persists (z = Gear;
    a freshly deployed robot has new wheels)."""

    Z = 2

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
    def solve(grid_mdp: AugmentedGridMdp, max_iters: Optional[int] = None) -> tuple[ValueFunc, Policy]:
        """If max_iters is given, return the state of your algorithm after
        exactly max_iters iterations (synchronous sweeps for VI; full
        evaluate-improve cycles for PI), starting from V = 0 and, for PI, from
        the first admissible action per state, breaking ties by action order."""
        pass

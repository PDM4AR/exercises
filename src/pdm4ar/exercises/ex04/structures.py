from enum import IntEnum, unique

import numpy as np
from numpy.typing import NDArray


@unique
class Action(IntEnum):
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3
    STAY = 4
    ABANDON = 5


State = tuple[int, int]
"""The state on a grid is simply a tuple of two ints"""


@unique
class Cell(IntEnum):
    GOAL = 0
    START = 1
    GRASS = 2
    SWAMP = 3
    WONDERLAND  = 4
    CLIFF = 5


Policy = NDArray[np.int64]
"""Type Alias for the policy.It is the expected type of the policy that your solution should return."""
OptimalActions = NDArray[np.object_]
"""
Type Alias for the all optimal actions per state. It is a numpy array of list objects where each list contains the
optimal actions that are equally good for a given state. It is the type of the ground truth policy that your
solution will be compared against. You are not required to use this type in your solution.
"""
ValueFunc = NDArray[np.float64]
"""Type Alias for the value function. It is the expected type of the value function that your solution should return."""


# --------------------------------------------------------------------------
# Part 2: augmented states (momentum / forecast / glitch)
# --------------------------------------------------------------------------
from enum import IntEnum, unique  # noqa: E402


@unique
class Heading(IntEnum):
    """Momentum case: the direction the robot actually moved last hour."""

    NONE = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3
    EAST = 4


@unique
class Fog(IntEnum):
    """Forecast case: this hour's fog forecast."""

    CLEAR = 0
    FOGGY = 1


@unique
class Gear(IntEnum):
    """Glitch case: the state of the robot's wheels."""

    OK = 0
    GLITCHY = 1


AugmentedState = tuple[int, int, int]
"""Part-2 state: (i, j, z). The meaning and ordering of z is fixed per case
(Heading / Fog / Gear); your (M, N, Z) output arrays must use these orderings."""

P_FOGGY = 0.3
"""A-priori probability of a FOGGY forecast (forecasts are iid each hour)."""
FOGGY_EXTRA_SLIP = 0.20
"""Extra slip mass under a FOGGY forecast, taken from the intended direction."""
P_GEAR_BREAK_IN = 0.1
"""P(OK -> GLITCHY) each hour."""
P_GEAR_RECOVER = 0.3
"""P(GLITCHY -> OK) each hour."""
GLITCH_EXTRA_SLIP = 0.15
"""Extra slip mass while GLITCHY, taken from the intended direction."""
MOMENTUM_ALIGNED_BONUS = 0.10
"""Added to the intended-direction probability when commanding along h."""
MOMENTUM_OPPOSED_MALUS = 0.10
"""Removed from the intended-direction probability when commanding against h."""

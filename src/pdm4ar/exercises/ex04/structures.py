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
    CLIFF = 4


Policy = NDArray[np.int64]
"""Type Alias for the policy.It is the expected type of the policy that your solution should return."""
OptimalActions = NDArray[np.object_]
"""All optimal actions per state (object array of lists): the ground-truth
policy type your solution is compared against. You need not use it."""
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

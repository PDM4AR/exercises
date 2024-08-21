from enum import IntEnum, unique
from typing import Union

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
    WORMHOLE = 4
    CLIFF = 5


Policy = NDArray[np.int64]
"""Type Alias for the policy"""
AllOptimalActions = Union[NDArray[np.object_], Policy]
"""Type Alias for the all optimal actions per state"""
ValueFunc = NDArray[np.float64]
"""Type Alias for the value function"""

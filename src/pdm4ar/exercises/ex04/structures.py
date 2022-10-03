from enum import IntEnum, unique
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@unique
class Action(IntEnum):
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3
    STAY = 4


State = Tuple[int, int]
"""The state on a grid is simply a tuple of two ints"""


@unique
class Cell(IntEnum):
    GOAL = 0
    START = 1
    GRASS = 2
    SWAMP = 3


Policy = NDArray[np.int]
"""Type Alias for the policy"""
ValueFunc = NDArray[np.float]
"""Type Alias for the value function"""

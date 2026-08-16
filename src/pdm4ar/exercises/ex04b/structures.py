from enum import IntEnum, unique

import numpy as np
from numpy.typing import NDArray

from pdm4ar.exercises.ex04.structures import Action, Cell  # noqa: F401  (re-exported)


@unique
class Heading(IntEnum):
    """Case 1 (Momentum): the direction the robot actually moved last hour."""

    NONE = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3
    EAST = 4


@unique
class Fog(IntEnum):
    """Case 2 (Forecast): this hour's fog forecast."""

    CLEAR = 0
    FOGGY = 1


@unique
class Gear(IntEnum):
    """Case 3 (Glitch): the state of the robot's wheels."""

    OK = 0
    GLITCHY = 1


State = tuple[int, int, int]
"""The ex04b state: (i, j, z). The meaning and ordering of z is fixed per case
(Heading / Fog / Gear above); your output arrays must use these orderings."""

ValueFunc = NDArray[np.float64]
"""Expected shape (M, N, Z): one grid slice per value of z."""

Policy = NDArray[np.int64]
"""Expected shape (M, N, Z): one grid slice per value of z."""

# ------------------------------------------------------------------ constants
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

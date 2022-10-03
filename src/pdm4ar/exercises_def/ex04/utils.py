from functools import wraps
from time import process_time
from typing import Callable

from frozendict import frozendict
from matplotlib.colors import to_rgb

from pdm4ar.exercises.ex04.structures import Action, Cell
from pdm4ar.exercises_def import logger

arrow_size = 0.25
head_width = 0.15
action2arrow = frozendict(
    {
        Action.NORTH: (-arrow_size, 0),
        Action.WEST: (0, -arrow_size),
        Action.SOUTH: (arrow_size, 0),
        Action.EAST: (0, arrow_size),
        Action.STAY: (0, 0),
    }
)

cell2color = frozendict(
    {Cell.GOAL: to_rgb("red"), Cell.START: to_rgb("yellow"), Cell.GRASS: to_rgb("green"), Cell.SWAMP: to_rgb("cyan")}
)


def time_function(func: Callable):
    """Decorator to time the execution time of a function/method call"""

    @wraps(func)
    def _time_it(*args, **kwargs):
        start = process_time()
        try:
            return func(*args, **kwargs)
        finally:
            delta = process_time() - start
            msg = f'Execution time of "{func.__qualname__}" defined in "{func.__module__}": {delta} s'
            logger.info(msg)

    return _time_it

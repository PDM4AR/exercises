from dataclasses import dataclass
from typing import Any, Mapping

from dg_commons import PlayerName
from dg_commons.sim.shared_goals import SharedPolygonGoal

GOAL_VALUE_MIN: int = 10
"""Smallest admissible goal value (in dollars)."""
GOAL_VALUE_MAX: int = 100
"""Largest admissible goal value (in dollars)."""
REVENUE_FLOOR: float = 0.2
"""Minimum fraction of a goal's value that is paid for a (late) delivery."""


@dataclass
class ValuedSharedPolygonGoal(SharedPolygonGoal):
    """A shared goal carrying a value (in dollars) that decays with its delivery time. The goal can be assigned
    to a specific robot. If a robot which is not the assigned one happens to deliver that goal, it earns neither
    the delivery reward nor the revenue, and the task counts as not accomplished"""

    value: float = 0.0
    assigned_robot: PlayerName | None = None


def goal_revenue(value: float, delivery_time: float | None, max_sim_time: float) -> float:
    """
    Revenue earned for a goal delivered at `delivery_time`.
    The value decays linearly with time, down to a floor of REVENUE_FLOOR * value.
    A goal that is never delivered earns nothing.
    """
    if delivery_time is None:
        return 0.0
    return value * max(REVENUE_FLOOR, 1.0 - delivery_time / max_sim_time)


def validate_goal_value(goal_data: Mapping[str, Any]) -> float:
    """Validate a goal value; zero disables revenue for legacy comparisons.

    Generated valued scenarios still use the normal 10-100 range.
    """
    goal_id = goal_data.get("id")
    if "value" not in goal_data:
        raise ValueError(f"shared_goal {goal_id} has no 'value'.")
    value = goal_data["value"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"shared_goal {goal_id} value must be an integer, got {value!r}.")
    if value != 0 and not GOAL_VALUE_MIN <= value <= GOAL_VALUE_MAX:
        raise ValueError(
            f"shared_goal {goal_id} value must be 0 or in [{GOAL_VALUE_MIN}, {GOAL_VALUE_MAX}], got {value}."
        )
    return float(value)

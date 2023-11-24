from dataclasses import dataclass
from decimal import Decimal
from dg_commons.sim import SimTime
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.rocket import RocketState
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class RocketTarget(PlanningGoal):
    target: DynObstacleState
    pos_tol: float
    vel_tol: float

    def is_fulfilled(self, state: RocketState, at: SimTime = Decimal(0)) -> bool:
        target_now = self._compute_taget_position_at(at)
        # TODO implement state within tollerance
        # between target_now and state

        return True

    def get_plottable_geometry(self) -> BaseGeometry:
        raise NotImplementedError

    def _compute_taget_position_at(self, at: SimTime) -> DynObstacleState:
        # TODO
        return self.target

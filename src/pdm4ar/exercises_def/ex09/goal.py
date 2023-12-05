from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from math import cos, sin

import numpy as np
from dg_commons.sim import SimTime
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.rocket import RocketState
from shapely import Polygon
from shapely.geometry import Point


@dataclass(frozen=True)
class RocketTarget(PlanningGoal):
    target: DynObstacleState
    pos_tol: float
    vel_tol: float
    dir_tol: float

    def is_fulfilled(self, state: RocketState, at: SimTime = Decimal(0)) -> bool:
        return self._is_fulfilled(state, self.target, self.pos_tol, self.vel_tol, self.dir_tol)

    def get_plottable_geometry(self) -> Polygon:
        return self._plottable_geometry

    @cached_property
    def _plottable_geometry(self) -> Polygon:
        # Make sure norm is aligned with is_fulfilled function
        goal_shape = Point(self.target.x, self.target.y).buffer(self.pos_tol)
        return goal_shape

    @staticmethod
    def _is_fulfilled(state: RocketState, target: DynObstacleState, pos_tol: float, vel_tol: float, dir_tol: float) -> bool:
        is_within_position = np.linalg.norm(
                np.array([state.x, state.y]) - np.array([target.x, target.y])) < pos_tol
        is_within_orientation = np.linalg.norm(
                np.array([state.psi]) - np.array([target.psi])) < dir_tol
        is_within_velocity = np.linalg.norm(
                np.array([state.vx, state.vy]) - np.array([target.vx, target.vy])) < vel_tol

        return is_within_position and is_within_orientation and is_within_velocity


@dataclass(frozen=True)
class SatelliteTarget(RocketTarget):
    planet_x: float
    planet_y: float
    omega: float
    tau: float
    orbit_r: float
    radius: float
    offset_r: float

    def is_fulfilled(self, state: RocketState, at: SimTime = Decimal(0)) -> bool:
        target_at = self.get_target_state_at(at)
        return self._is_fulfilled(state, target_at, self.pos_tol, self.vel_tol, self.dir_tol)

    def get_plottable_geometry(self, at: SimTime | float = 0) -> Polygon:
        # Make sure norm is aligned with is_fulfilled function
        target_at = self.get_target_state_at(at)
        goal_shape = Point(target_at.x, target_at.y).buffer(self.pos_tol)
        return goal_shape

    def get_target_state_at(self, at: SimTime) -> DynObstacleState:
        at_float = float(at)
        cos_omega_t = (self.orbit_r + self.offset_r) * cos(self.omega * at_float + self.tau)
        sin_omega_t = (self.orbit_r + self.offset_r) * sin(self.omega * at_float + self.tau)
        x = cos_omega_t + self.planet_x
        y = sin_omega_t + self.planet_y
        v = self.omega * self.orbit_r

        psi = np.pi/2 + self.omega * at_float + self.tau
        vx = v * sin(psi)
        vy = v * cos(psi)

        return DynObstacleState(x=x, y=y, psi=psi, vx=vx, vy=vy, dpsi=self.omega)
    
    @property
    def is_static(self) -> bool:
        return False
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

    def is_fulfilled(self, state: RocketState, at: SimTime = Decimal(0)) -> bool:
        return self._is_fulfilled(state, self.target, self.pos_tol, self.vel_tol)

    def get_plottable_geometry(self) -> Polygon:
        return self._plottable_geometry

    @cached_property
    def _plottable_geometry(self) -> Polygon:
        # Make sure norm is aligned with is_fulfilled function
        goal_shape = Point(self.target.x, self.target.y).buffer(self.pos_tol)
        return goal_shape

    @staticmethod
    def _is_fulfilled(state: RocketState, target: DynObstacleState, pos_tol: float, vel_tol: float) -> bool:
        is_within_position = np.linalg.norm(
                np.array([state.x, state.y, state.psi]) - np.array([target.x, target.y, target.psi])) < pos_tol
        is_within_velocity = np.linalg.norm(
                np.array([state.vx, state.vy, state.dpsi]) - np.array([target.vx, target.vy, target.dpsi])) < vel_tol

        return is_within_position and is_within_velocity


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
        target_at = self._get_target_state_at(at)
        return self._is_fulfilled(state, target_at, self.pos_tol, self.vel_tol)

    def get_plottable_geometry(self) -> Polygon:
        # Make sure norm is aligned with is_fulfilled function
        goal_shape = Point(self.target.x, self.target.y).buffer(self.pos_tol)
        return goal_shape

    def _get_target_state_at(self, at: SimTime) -> DynObstacleState:
        at_float = float(at)
        cos_omega_t = (self.orbit_r + self.offset_r) * cos(self.omega * at_float + self.tau)
        sin_omega_t = (self.orbit_r + self.offset_r) * sin(self.omega * at_float + self.tau)
        x = cos_omega_t + self.planet_x
        y = sin_omega_t + self.planet_y
        v = self.omega * self.orbit_r
        psi = np.pi/2 + np.arctan2(sin_omega_t, cos_omega_t)
        # vx = v*cos(psi)
        # vy = v*sin(psi)

        return DynObstacleState(x=x, y=y, psi=psi, vx=v, vy=0, dpsi=self.omega)
    

if __name__ == '__main__':
    # verify get_target_state_at
    import matplotlib.pyplot as plt

    planet_x = 2.0
    planet_y = 1.0

    satellite_orbit_r = 7.0
    satellite_tau = 5.2359
    satellite_radius = 0.5
    satellite_omega = 0.05454
    
    target_offset_r = 0.7

    satellite_x0 = planet_x + (satellite_orbit_r) * cos(satellite_omega * planet_x + satellite_tau)
    satellite_y0 = planet_y + (satellite_orbit_r) * sin(satellite_omega * planet_y + satellite_tau)
    satellite_psi0 = np.pi/2 + np.arctan2(satellite_y0, satellite_x0)
    satellite_v0 = satellite_omega * satellite_orbit_r
    satellite_vx0 = satellite_v0*cos(satellite_psi0)
    satellite_vy0 = satellite_v0*sin(satellite_psi0)
    
    target0 = DynObstacleState(x=satellite_x0, y=satellite_y0, psi=satellite_psi0, vx=satellite_vx0, vy=satellite_vy0, dpsi=satellite_omega)

    pos_tol, vel_tol = 0.7, 0.7

    target = SatelliteTarget(target0, pos_tol, vel_tol, planet_x, planet_y, satellite_omega, satellite_tau, satellite_orbit_r, satellite_radius, target_offset_r)

    times = np.linspace(0, 200, 100)

    print(target0.__dict__)
    print(target._get_target_state_at(0).__dict__)

    # make square figure and axes
    plt.figure(figsize=(6,6))

    xs = []
    ys = []
    vxs = []
    vys = []
    for time in times:
        new_target = target._get_target_state_at(time)
        xs.append(new_target.x)
        ys.append(new_target.y)
        vxs.append(new_target.vx)
        vys.append(new_target.vy)
    
    plt.plot(planet_x, planet_y, 'o')
    plt.plot(xs, ys, 'x')
    for i,x in enumerate(xs):
        plt.plot([x, x+vxs[i]], [ys[i], ys[i]+vys[i]], 'k-')

    dist = np.sqrt((np.array(xs) - planet_x)**2 + (np.array(ys) - planet_y)**2)
    print(dist)

    plt.show()
    
    pass

# change _get_target_state_at
# increase tolerance also in private ones
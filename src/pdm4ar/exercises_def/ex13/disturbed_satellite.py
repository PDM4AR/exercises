from dataclasses import replace
from decimal import Decimal

import numpy as np
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteModel, SatelliteState
from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters

from pdm4ar.exercises_def.ex13.utils_params import DynamicsDisturbanceParams


class DisturbedSatelliteModel(SatelliteModel):
    """Satellite model with a reproducible, time-limited external acceleration.

    The disturbance is sampled once when the model is created and is held
    constant during its activation window. It affects only the simulator; the
    student's nominal dynamics and planner remain unchanged.
    """

    def __init__(
        self,
        x0: SatelliteState,
        rg: SatelliteGeometry,
        rp: SatelliteParameters,
        disturbance: DynamicsDisturbanceParams,
    ):
        super().__init__(x0=x0, rg=rg, rp=rp)
        self.disturbance = disturbance
        self._elapsed_time = Decimal("0")
        self._start_time = Decimal(str(disturbance.start_time))
        self._end_time = self._start_time + Decimal(str(disturbance.duration))
        self._disturbance_active = False

        rng = np.random.default_rng(disturbance.seed)
        self._linear_acceleration = rng.normal(0.0, disturbance.linear_acc_sigma, size=2)
        self._angular_acceleration = float(rng.normal(0.0, disturbance.angular_acc_sigma))

    @classmethod
    def default(cls, x0: SatelliteState, disturbance: DynamicsDisturbanceParams):
        return cls(
            x0=x0,
            rg=SatelliteGeometry.default(),
            rp=SatelliteParameters.default(),
            disturbance=disturbance,
        )

    @property
    def sampled_disturbance(self) -> tuple[float, float, float]:
        """Return the sampled (ax, ay, angular acceleration) for reporting/tests."""
        return (
            float(self._linear_acceleration[0]),
            float(self._linear_acceleration[1]),
            self._angular_acceleration,
        )

    def update(self, commands: SatelliteCommands, dt: Decimal):
        self._disturbance_active = self._start_time <= self._elapsed_time < self._end_time
        super().update(commands=commands, dt=dt)
        self._elapsed_time += dt

    def dynamics(self, x0: SatelliteState, u: SatelliteCommands) -> SatelliteState:
        derivative = super().dynamics(x0=x0, u=u)
        if not self._disturbance_active:
            return derivative

        return replace(
            derivative,
            vx=derivative.vx + float(self._linear_acceleration[0]),
            vy=derivative.vy + float(self._linear_acceleration[1]),
            dpsi=derivative.dpsi + self._angular_acceleration,
        )

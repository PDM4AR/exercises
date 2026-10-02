from dataclasses import replace
from decimal import Decimal

import numpy as np
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipModel, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters

from pdm4ar.exercises_def.ex15.utils_params import DynamicsDisturbanceParams


class DisturbedSpaceshipModel(SpaceshipModel):
    """Spaceship model with a reproducible, time-limited world-frame acceleration.

    The disturbance is sampled once when the model is created and is held
    constant during its activation window. It affects only the simulator; the
    student's nominal dynamics and planner remain unchanged.
    """

    def __init__(
        self,
        x0: SpaceshipState,
        rg: SpaceshipGeometry,
        sp: SpaceshipParameters,
        disturbance: DynamicsDisturbanceParams,
    ):
        super().__init__(x0=x0, rg=rg, sp=sp)
        self.disturbance = disturbance
        self._elapsed_time = Decimal("0")
        self._start_time = Decimal(str(disturbance.start_time))
        self._end_time = self._start_time + Decimal(str(disturbance.duration))
        self._disturbance_active = False

        rng = np.random.default_rng(disturbance.seed)
        self._linear_acceleration = rng.normal(0.0, disturbance.linear_acc_sigma, size=2)
        self._angular_acceleration = float(rng.normal(0.0, disturbance.angular_acc_sigma))

    @classmethod
    def default(cls, x0: SpaceshipState, disturbance: DynamicsDisturbanceParams):
        return cls(
            x0=x0,
            rg=SpaceshipGeometry.default(),
            sp=SpaceshipParameters.default(),
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

    def update(self, commands: SpaceshipCommands, dt: Decimal):
        step_end = self._elapsed_time + dt
        # Split steps crossing the activation boundaries so the impulse is
        # independent of the simulator timestep, including short bursts.
        boundaries = [t for t in (self._start_time, self._end_time) if self._elapsed_time < t < step_end]
        for end in boundaries + [step_end]:
            self._disturbance_active = self._start_time <= self._elapsed_time < self._end_time
            super().update(commands=commands, dt=end - self._elapsed_time)
            self._elapsed_time = end
        self._disturbance_active = self._start_time <= self._elapsed_time < self._end_time

    def dynamics(self, x0: SpaceshipState, u: SpaceshipCommands) -> SpaceshipState:
        derivative = super().dynamics(x0=x0, u=u)
        if not self._disturbance_active:
            return derivative

        # Spaceship velocities are expressed in the body frame. Rotate the
        # fixed world-frame acceleration at each integration evaluation.
        cospsi, sinpsi = np.cos(x0.psi), np.sin(x0.psi)
        ax, ay = self._linear_acceleration
        return replace(
            derivative,
            vx=derivative.vx + float(cospsi * ax + sinpsi * ay),
            vy=derivative.vy + float(-sinpsi * ax + cospsi * ay),
            dpsi=derivative.dpsi + self._angular_acceleration,
        )

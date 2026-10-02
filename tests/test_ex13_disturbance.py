"""Physics regressions for world-frame disturbances on the spaceship model."""

from dataclasses import replace
from decimal import Decimal
import unittest

import numpy as np
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipModel, SpaceshipState

from pdm4ar.exercises_def.ex13.disturbed_spaceship import DisturbedSpaceshipModel
from pdm4ar.exercises_def.ex13.utils_params import DynamicsDisturbanceParams


class DisturbanceTests(unittest.TestCase):
    state = SpaceshipState(0, 0, 0, 0, 0, 0, 0, 2.5)
    command = SpaceshipCommands(0, 0)

    def model(self, **kwargs):
        params = dict(start_time=0.05, duration=0.025, linear_acc_sigma=0.35, angular_acc_sigma=0.0, seed=42)
        params.update(kwargs)
        return DisturbedSpaceshipModel.default(self.state, DynamicsDisturbanceParams(**params))

    def test_seed_is_reproducible(self):
        self.assertEqual(self.model().sampled_disturbance, self.model().sampled_disturbance)
        self.assertNotEqual(self.model().sampled_disturbance, self.model(seed=43).sampled_disturbance)

    def test_world_acceleration_rotates_into_body_frame(self):
        for psi in (0, np.pi / 2, np.pi, -np.pi / 3):
            with self.subTest(psi=psi):
                model = self.model(start_time=0, duration=1, angular_acc_sigma=0.08)
                model.update(self.command, Decimal("0.01"))
                state = replace(self.state, psi=psi, vx=0.3, vy=-0.2, dpsi=0.1)
                nominal = SpaceshipModel.default(state).dynamics(state, self.command).as_ndarray()
                change = model.dynamics(state, self.command).as_ndarray() - nominal
                world_acceleration = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]) @ change[3:5]
                np.testing.assert_allclose(world_acceleration, model.sampled_disturbance[:2], atol=1e-12)
                self.assertAlmostEqual(change[5], model.sampled_disturbance[2])
                np.testing.assert_allclose(change[[0, 1, 2, 6, 7]], 0, atol=1e-12)

    def test_short_burst_inside_one_timestep_has_exact_impulse(self):
        model = self.model()
        model.update(self.command, Decimal("0.1"))
        ax, ay, _ = model.sampled_disturbance
        expected = np.array([ax * 0.0009375, ay * 0.0009375, 0, ax * 0.025, ay * 0.025, 0, 0, 2.5])
        np.testing.assert_allclose(model.get_state().as_ndarray(), expected, atol=1e-10)
        np.testing.assert_allclose(
            model.dynamics(self.state, self.command).as_ndarray(),
            SpaceshipModel.default(self.state).dynamics(self.state, self.command).as_ndarray(),
        )

    def test_timestep_does_not_change_burst_impulse(self):
        coarse, fine = self.model(), self.model()
        coarse.update(self.command, Decimal("0.1"))
        for _ in range(10):
            fine.update(self.command, Decimal("0.01"))
        np.testing.assert_allclose(coarse.get_state().as_ndarray(), fine.get_state().as_ndarray(), atol=1e-10)

    def test_zero_disturbance_matches_nominal_with_thrust(self):
        disturbed = self.model(start_time=0, duration=1, linear_acc_sigma=0, angular_acc_sigma=0)
        nominal = SpaceshipModel.default(self.state)
        command = SpaceshipCommands(0.8, 0.1)
        for _ in range(10):
            nominal.update(command, Decimal("0.01"))
            disturbed.update(command, Decimal("0.01"))
        np.testing.assert_allclose(disturbed.get_state().as_ndarray(), nominal.get_state().as_ndarray(), atol=1e-12)

    def test_invalid_disturbance_is_rejected(self):
        for kwargs in ({"start_time": -1}, {"duration": 0}, {"linear_acc_sigma": -1}, {"angular_acc_sigma": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)


if __name__ == "__main__":
    unittest.main()

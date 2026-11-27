import importlib.util
import math
import unittest

from locomocao import (
    BASELINE, BOUNDS, HORIZON, INITIAL, POLICY_DT, PD_STEPS, DT, TAU_MAX,
    Scenario, evaluation, pd_torque, rollout, scenario, targets, train,
)

HAVE_MUJOCO = all(importlib.util.find_spec(package) is not None
                  for package in ("gymnasium", "numpy", "mujoco"))
if HAVE_MUJOCO:
    from hopper import INITIAL as HOPPER_INITIAL, action_pd, joint_addresses, make_hopper
    from hopper import rollout as hopper_rollout
    import numpy as np


class LocoTest(unittest.TestCase):
    def test_pd_resists_velocity_and_saturates(self):
        self.assertEqual(pd_torque(0, 0, 0), 0)
        self.assertLess(pd_torque(0, 0, 1.0), 0)
        self.assertEqual(pd_torque(20, 0, 0), TAU_MAX)
        self.assertAlmostEqual(PD_STEPS * DT, POLICY_DT)

    def test_phase_alternates_targets(self):
        left, right = targets(INITIAL, 0.0, 0.0, 0.0, 0.4)
        self.assertLess(left, right)
        rl, rr = targets(INITIAL, math.pi, 0.0, 0.0, 0.4)
        self.assertAlmostEqual(rl, right)
        self.assertAlmostEqual(rr, left)

    def test_quiet_baseline_ends_at_budget(self):
        run = rollout(BASELINE, Scenario(0, 1, 0, 0, 0))
        self.assertEqual(run.ret, HORIZON)
        self.assertEqual(run.steps, HORIZON)
        self.assertFalse(run.fell)

    def test_fall_really_ends_early(self):
        config = Scenario(0.4, 1.0, 0.07, 0.0, 0.0)
        run = rollout(BASELINE, config)
        self.assertTrue(run.fell)
        self.assertLess(run.steps, HORIZON)
        self.assertIsNone(evaluation(BASELINE, [config])[2])

    def test_short_training_reproducible_and_bounded(self):
        first = train(seed=7, iterations=2, population=6)
        self.assertEqual(first, train(seed=7, iterations=2, population=6))
        for value, (low, high) in zip(first, BOUNDS):
            self.assertTrue(low <= value <= high)

    def test_scenario_is_reproducible(self):
        self.assertEqual(scenario(7, 1.0), scenario(7, 1.0))
        flat = scenario(7, 0.0)
        self.assertEqual((flat.traction, flat.push_speed, flat.push_pitch), (1.0, 0.0, 0.0))


@unittest.skipUnless(HAVE_MUJOCO, "instale requirements-mujoco.txt para testar Hopper")
class HopperTest(unittest.TestCase):
    def test_action_is_normalized(self):
        action = action_pd(np.array([-10, 0, 10]), np.zeros(3), np.zeros(3))
        np.testing.assert_array_equal(action, np.array([-1, 0, 1], np.float32))
        self.assertEqual(action.dtype, np.float32)

    def test_reproducible_rollout_and_time_limit(self):
        env = make_hopper(horizon=3)
        try:
            dof = joint_addresses(env.unwrapped.model)
            self.assertEqual(len(set(dof)), 3)
            first = hopper_rollout(env, HOPPER_INITIAL, 9, dof)
            self.assertEqual(first, hopper_rollout(env, HOPPER_INITIAL, 9, dof))
            self.assertEqual(first.steps, 3)
            self.assertTrue(first.truncated)
            self.assertFalse(first.terminated)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

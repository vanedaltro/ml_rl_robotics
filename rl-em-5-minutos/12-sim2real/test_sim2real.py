import unittest

from sim2real import Plant, Policy, evaluate, rollout, scenarios, train


class Sim2RealTest(unittest.TestCase):
    def test_action_is_saturated_in_both_directions(self):
        p = Policy(0.25, 3)
        self.assertEqual(p.action(-9, 1), 1)
        self.assertEqual(p.action(9, -1), -1)

    def test_delay_longer_than_rollout_blocks_motion(self):
        s = rollout(Policy(0.25, 3), Plant(delay=8), 4, steps=8)
        self.assertEqual(s.mae, 1)
        self.assertLess(s.reward, -1)

    def test_randomization_is_reproducible(self):
        a = list(scenarios(17, 10))
        self.assertEqual(a, list(scenarios(17, 10)))
        b = list(scenarios(71, 10))
        self.assertEqual(len(set(a).intersection(b)), 0)
        self.assertTrue(all(0 <= p.delay <= 4 for p, _ in a + b))

    def test_seeded_stress_is_outside_training_range(self):
        self.assertTrue(all(p.gain < 0.50 and p.delay > 4
                            for p, _ in scenarios(123, 10, stress=True)))

    def test_training_maximizes_reward_among_candidates(self):
        cases = list(scenarios(17, 3))
        trained = train(cases)
        first = Policy(0.10, 0.5)
        self.assertGreaterEqual(evaluate(trained, cases).reward, evaluate(first, cases).reward)

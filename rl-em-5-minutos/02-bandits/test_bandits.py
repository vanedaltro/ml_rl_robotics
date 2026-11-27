import unittest
from random import Random

from bandits import BEST, MU, choose, experiment, rollout, update


class BanditTest(unittest.TestCase):
    def test_means(self):
        self.assertEqual(MU, (4, 6, 3))

    def test_incremental_mean(self):
        q = 0
        for n, r in enumerate([2, 7, 10, 5], 1):
            q = update(q, n, r)
        self.assertAlmostEqual(q, 6)

    def test_every_learning_strategy_tries_unseen_arms_first(self):
        for policy in ("guloso", "eps-010", "eps-001", "ucb"):
            self.assertEqual(choose(policy, [999, 999, 0], [1, 1, 0], 3, Random(42)), 2)

    def test_ucb_can_choose_less_tested_arm(self):
        self.assertEqual(choose("ucb", [4, 3, 0], [100, 1, 100], 202, Random(42)), 1)

    def test_oracle_regret(self):
        self.assertEqual(choose("oraculo", [9, 0, 9], [1, 1, 1], 4, Random(42)), BEST)
        for _, gap, share in rollout("oraculo", 50, 42):
            self.assertEqual((gap, share), (0, 1))

    def test_initial_regret_matches_acb_gaps(self):
        for policy in ("guloso", "eps-010", "eps-001", "ucb"):
            self.assertEqual([g for _, g, _ in rollout(policy, 3, 7)], [2, 2, 5])

    def test_reproducible_and_monotone_regret(self):
        a = list(experiment(9, 5, 7))
        self.assertEqual(a, list(experiment(9, 5, 7)))
        for policy in ("guloso", "eps-010", "eps-001", "ucb"):
            gs = [g for p, _, _, g, _ in a if p == policy]
            self.assertEqual(gs, sorted(gs))


if __name__ == "__main__":
    unittest.main()

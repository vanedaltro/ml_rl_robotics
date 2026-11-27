from unittest import TestCase

from marl import (actor_gradient, best_response, counterfactual, exploitability,
                  independent_q, prob_one, rps_reward, team_reward, train_ctde)


class CooperativeTests(TestCase):
    def test_reward_matrix(self):
        self.assertEqual([[team_reward((a0, a1)) for a1 in range(2)] for a0 in range(2)],
                         [[-1, 1], [1, -1]])

    def test_credit_changes_sign_and_has_zero_conditional_mean(self):
        q = [[-1, 1], [1, -1]]
        for agent in (0, 1):
            for other in (0, 1):
                pair = (0, other) if agent == 0 else (other, 0)
                alternative = (1, other) if agent == 0 else (other, 1)
                p = 0.31
                v0 = counterfactual(q, pair, agent, p)
                v1 = counterfactual(q, alternative, agent, p)
                self.assertLess(v0 * v1, 0)
                self.assertAlmostEqual((1-p) * v0 + p * v1, 0.0)

    def test_exact_gradient_matches_central_finite_difference(self):
        q = [[-1, 1], [1, -1]]
        logits = [-0.2, 0.6]

        def joint_return(z0, z1):
            p0, p1 = prob_one(z0), prob_one(z1)
            return 2 * (p0 * (1 - p1) + (1 - p0) * p1) - 1

        eps = 1e-5
        measured = actor_gradient(q, [prob_one(z) for z in logits])
        for i in (0, 1):
            plus, minus = logits.copy(), logits.copy()
            plus[i] += eps
            minus[i] -= eps
            expected = (joint_return(*plus) - joint_return(*minus)) / (2 * eps)
            self.assertAlmostEqual(measured[i], expected, places=8)

    def test_four_warm_up_samples_include_two_collisions(self):
        result, _, _ = train_ctde(seed=7, episodes=4)
        self.assertEqual(result.collisions, 2)
        self.assertEqual(result.success, 0.5)

    def test_seeded_demo_is_repeatable_and_coordinates_locally(self):
        result = train_ctde(seed=42, episodes=1200)
        self.assertEqual(result, train_ctde(seed=42, episodes=1200))
        _, actions, (p0, p1) = result
        self.assertNotEqual(*actions)
        self.assertGreater(p0, 0.95)
        self.assertLess(p1, 0.05)
        self.assertEqual(independent_q(42, 1200), independent_q(42, 1200))


class CompetitiveTests(TestCase):
    def test_zero_sum_and_best_responses(self):
        for a in range(3):
            for b in range(3):
                self.assertEqual(rps_reward(a, b), -rps_reward(b, a))
            self.assertEqual(rps_reward(best_response(a), a), 1)

    def test_one_sided_exploitability_is_zero_for_uniform(self):
        self.assertEqual(exploitability((1 / 3, 1 / 3, 1 / 3)), 0)
        self.assertEqual(exploitability((1, 0, 0)), 1)

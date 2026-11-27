import unittest
from collections import defaultdict
from random import Random

from model_based import ACTIONS, GOAL, Model, action_values, dataset, dyna, policy_mpc, q_update, step, train


class ModelBasedTest(unittest.TestCase):
    def test_empirical_probabilities_and_rewards(self):
        model = Model()
        for nxt in (0, 2, 2, 2):
            model.observe(0, "salta", nxt, -1)
        self.assertEqual(model.predict(0, "salta"), ((0.25, 0, -1), (0.75, 2, -1)))

    def test_unobserved_action_is_rejected(self):
        with self.assertRaises(ValueError):
            Model().predict(0, "anda")

    def test_terminal_transition_and_no_bootstrap(self):
        self.assertEqual(step(5, "anda", Random(0)), (GOAL, 9))
        with self.assertRaises(ValueError):
            step(GOAL, "anda", Random(0))
        model = train(20, seed=18)
        self.assertEqual(action_values(model, 5, 5)["anda"], 9)
        q = defaultdict(float, {(GOAL, a): 999 for a in ACTIONS})
        q_update(q, 5, "anda", GOAL, 9, alpha=1)
        self.assertEqual(q[5, "anda"], 9)

    def test_five_steps_can_prefer_jump_over_horizon_one(self):
        model = train(80, seed=18)
        self.assertEqual(policy_mpc(model, 1)(0), "anda")
        self.assertEqual(policy_mpc(model, 5)(0), "salta")

    def test_dyna_with_no_planning_is_q_learning(self):
        transitions = list(dataset(20, seed=3))
        q = defaultdict(float)
        for s, a, nxt, reward in transitions:
            q_update(q, s, a, nxt, reward)
        self.assertEqual(dyna(transitions, 0, 17), q)

    def test_dyna_planning_is_seed_reproducible(self):
        self.assertEqual(dyna(dataset(20, 5), 7, 17), dyna(dataset(20, 5), 7, 17))

import unittest
from functools import lru_cache

from demo import (
    ACTIONS, DRY, GOAL, GRIP, HORIZON, LEFT, STEP, WET, episode,
    experiment, explore, planner, reward, transition,
)


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = experiment()

    def test_dynamics_and_terminal(self):
        self.assertEqual(transition(0, LEFT, DRY), 0)
        self.assertEqual(transition(3, STEP, DRY), 4)
        self.assertEqual(transition(3, STEP, WET), 3)
        self.assertEqual(transition(3, GRIP, WET), 4)
        self.assertEqual(transition(GOAL, GRIP, WET), GOAL)
        self.assertEqual(reward(7, STEP, GOAL), 11)
        self.assertEqual(reward(7, GRIP, GOAL), 9)
        self.assertEqual(reward(GOAL, GRIP, GOAL), 0)

    def test_exploration_covers_training_state_action_pairs(self):
        observed = {(s, a) for s, a, _ in explore(2101, 80, DRY)}
        self.assertEqual(observed, {(s, a) for s in range(GOAL) for a in ACTIONS})

    def test_oracle_uses_only_two_grip_steps_in_wet(self):
        policy = planner(lambda s, a: transition(s, a, WET))
        run = episode(0, policy, WET)
        self.assertEqual(run["positions"], list(range(GOAL + 1)))
        self.assertEqual([s for s, a in zip(run["positions"], run["actions"]) if a == GRIP], [3, 5])
        self.assertEqual(run["total"], 0)

    def test_finite_horizon_against_exhaustive_recurrence(self):
        def next_state(s, a):
            return transition(s, a, WET)

        @lru_cache
        def best_score(s, h):
            if not h or s == GOAL:
                return 0
            return max(
                reward(s, a, next_state(s, a)) + best_score(next_state(s, a), h - 1)
                for a in ACTIONS
            )

        policy = planner(next_state)
        for s in range(GOAL):
            for h in range(1, HORIZON + 1):
                a = policy(s, h)
                self.assertEqual(reward(s, a, next_state(s, a)) + best_score(next_state(s, a), h - 1),
                                 best_score(s, h))

    def test_aggregate_accuracy_hides_critical_failures(self):
        dry = self.data["results"]["seco"]
        wet = self.data["results"]["molhado"]
        self.assertEqual(dry["preditor"]["correct"], dry["preditor"]["n"])
        self.assertGreater(wet["preditor"]["accuracy"], 0.8)
        self.assertGreater(wet["preditor_pares_criticos"]["n"], 0)
        self.assertEqual(wet["preditor_pares_criticos"]["correct"], 0)

    def test_closed_loop_shift_isolated_by_model_swap(self):
        dry = self.data["results"]["seco"]
        wet = self.data["results"]["molhado"]
        for policy in ("clonagem", "planejador_modelo_aprendido", "planejador_modelo_exato"):
            self.assertEqual(dry[policy]["successes"], 32)
        self.assertEqual(wet["clonagem"]["successes"], 0)
        self.assertEqual(wet["planejador_modelo_aprendido"]["successes"], 0)
        self.assertEqual(wet["planejador_modelo_exato"]["successes"], 32)


if __name__ == "__main__":
    unittest.main()

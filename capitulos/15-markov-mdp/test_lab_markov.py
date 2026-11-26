"""Testes do laboratório de cadeias de Markov e MDP."""
import unittest

from lab_markov import (
    MDP, SECOND_ORDER_RAIN, STATES, WEATHER, pair_matrix,
    policy_matrix, simulate_weather, step, value_iteration,
)


class MarkovLabTest(unittest.TestCase):
    def assert_stochastic(self, matrix):
        for row in matrix:
            self.assertTrue(all(0 <= probability <= 1 for probability in row))
            self.assertAlmostEqual(sum(row), 1.0)

    def test_weather(self):
        self.assert_stochastic(WEATHER)
        self.assertAlmostEqual(step((1, 0), WEATHER)[1], 0.20)
        self.assertAlmostEqual(step(step((1, 0), WEATHER), WEATHER)[1], 0.28)
        self.assertAlmostEqual(step((2 / 3, 1 / 3), WEATHER)[1], 1 / 3)

    def test_order_two_pairs(self):
        self.assertEqual(SECOND_ORDER_RAIN[("CHUVA", "CHUVA")], 0.8)
        self.assertEqual(SECOND_ORDER_RAIN[("SOL", "CHUVA")], 0.4)
        pairs, matrix = pair_matrix()
        self.assert_stochastic(matrix)
        for (a, b), row in zip(pairs, matrix):
            for (c, _), probability in zip(pairs, row):
                if c != b:
                    self.assertEqual(probability, 0)
            rain = sum(p for (_, d), p in zip(pairs, row) if d == "CHUVA")
            self.assertAlmostEqual(rain, SECOND_ORDER_RAIN[(a, b)])

    def test_optimal_actions_and_terminal_values(self):
        for choices in MDP.values():
            for outcomes in choices.values():
                self.assertAlmostEqual(sum(p for p, _, _ in outcomes), 1)
        v, policy, scores, _ = value_iteration(0.9)
        self.assertEqual(policy, {"partida": "desvio", "corredor": "avançar"})
        self.assertAlmostEqual(scores["partida"]["atalho"], 1.0)
        self.assertAlmostEqual(scores["partida"]["desvio"], 6.2)
        self.assertAlmostEqual(v["meta"], 0)
        self.assertAlmostEqual(v["falha"], 0)
        v, policy, scores, _ = value_iteration(0)
        self.assertEqual(policy["partida"], "atalho")
        self.assertAlmostEqual(v["partida"], 1)
        self.assertAlmostEqual(scores["partida"]["desvio"], -1)

    def test_induced_chain(self):
        no_shortcut = policy_matrix(value_iteration(0.9)[1])
        shortcut = policy_matrix(value_iteration(0)[1])
        self.assertEqual(STATES, ("partida", "corredor", "meta", "falha"))
        self.assert_stochastic(no_shortcut)
        self.assert_stochastic(shortcut)
        self.assertEqual(no_shortcut, ((0, 1, 0, 0), (0, 0, 1, 0),
                                       (0, 0, 1, 0), (0, 0, 0, 1)))
        self.assertEqual(shortcut[0], (0, 0, 0.7, 0.3))

    def test_simulation(self):
        self.assertEqual(simulate_weather(500, 3), simulate_weather(500, 3))
        sim1, sim2 = simulate_weather(20_000, 42)
        self.assertLess(abs(sim1 - 0.2), 0.015)
        self.assertLess(abs(sim2 - 0.28), 0.015)

    def test_parameter_validation(self):
        for invalid in (-0.2, 1.0):
            with self.assertRaises(ValueError):
                value_iteration(invalid)
        with self.assertRaises(RuntimeError):
            value_iteration(0.9, max_iterations=1)


if __name__ == "__main__":
    unittest.main()

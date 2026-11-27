import unittest
from random import Random

from cliff import INICIO, META, atualizar, avaliar, escolher, step, treinar


class TestCliff(unittest.TestCase):
    def test_transicoes(self):
        self.assertEqual(step(INICIO, 1), (INICIO, -100, False, True))
        self.assertEqual(step(INICIO, 3), (INICIO, -1, False, False))
        self.assertEqual(step((2, 11), 2), (META, -1, True, False))

    def test_conta_do_video(self):
        self.assertAlmostEqual(atualizar(2, -1, 6, False, 0.5, 0.9), 3.2)

    def test_sarsa_e_q_usam_alvos_distintos(self):
        valores = [1, 6, -1, 0]
        proxima_acao = 0
        self.assertAlmostEqual(atualizar(2, -1, max(valores), False, .5, .9), 3.2)
        self.assertAlmostEqual(
            atualizar(2, -1, valores[proxima_acao], False, .5, .9), .95
        )

    def test_terminal_sem_bootstrap(self):
        self.assertEqual(atualizar(0, -1, 999, True, 1, 1), -1)

    def test_limite_de_coleta_nao_e_terminal(self):
        self.assertEqual(atualizar(0, -1, 6, False, 1, 1), 5)

    def test_epsilon_e_seed(self):
        self.assertEqual(escolher([1, 5, 3, 2], 0, Random(42)), 1)
        a = treinar("sarsa", episodios=5, seed=7)
        self.assertEqual(a, treinar("sarsa", episodios=5, seed=7))

    def test_politicas_chegam_a_meta(self):
        for nome in ("q-learning", "sarsa"):
            q, _, _ = treinar(nome, episodios=500, seed=42)
            caminho, terminou = avaliar(q)
            self.assertTrue(terminou)
            self.assertEqual(caminho[-1], META)


if __name__ == "__main__":
    unittest.main()

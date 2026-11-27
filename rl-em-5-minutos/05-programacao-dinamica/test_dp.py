import unittest
from math import inf

from programacao_dinamica import (ACOES, BANDEIRA, ESTADOS, GAMMA, INICIO, backup,
                                  moedas_memoizadas, moedas_minimas,
                                  policy_iteration, roteiro, value_iteration)


class TestDP(unittest.TestCase):
    def test_coin_change(self):
        tabela = moedas_minimas(6)
        self.assertEqual(tabela, [0, 1, 2, 1, 1, 2, 2])
        self.assertEqual([moedas_memoizadas(i) for i in range(7)], tabela)
        self.assertEqual(moedas_minimas(5, (2, 4))[-1], inf)
        self.assertEqual(moedas_memoizadas(5, (2, 4)), inf)

    def test_value_iteration_ponto_fixo(self):
        v, pi, _ = value_iteration()
        for s in ESTADOS:
            self.assertAlmostEqual(v[s], max(backup(s, a, v, GAMMA) for a in ACOES))
            self.assertAlmostEqual(v[s], backup(s, pi[s], v, GAMMA))

    def test_value_e_policy_iteration_concordam(self):
        v, _, _ = value_iteration()
        vp, _, _ = policy_iteration()
        for s in ESTADOS:
            self.assertAlmostEqual(v[s], vp[s], places=8)

    def test_rota_otima_e_recompensas(self):
        for solver in (value_iteration, policy_iteration):
            v, pi, _ = solver()
            caminho = roteiro(pi)
            self.assertAlmostEqual(v[INICIO], 1.8098)
            self.assertEqual([r for _, _, r in caminho], [-1] * 5 + [10])
            self.assertEqual(caminho[-1][1], BANDEIRA)

    def test_limites(self):
        with self.assertRaises(ValueError):
            value_iteration(gamma=1.0)
        with self.assertRaises(RuntimeError):
            value_iteration(max_iter=1)

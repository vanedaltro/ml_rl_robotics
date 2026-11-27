import unittest
from random import Random

from previsao import caminhada, comparar, mc_atualizar, td_atualizar


class TestPrevisao(unittest.TestCase):
    def test_mc_retorno_descontado(self):
        v = {1: 0., 2: 0.}
        mc_atualizar(v, [(1, 0, 2, False), (2, 1, 3, True)], 1., 0.9)
        self.assertEqual(v, {1: 0.9, 2: 1.})

    def test_mc_recusa_truncamento(self):
        with self.assertRaises(ValueError):
            mc_atualizar({1: 0.}, [(1, 0., 2, False)], 0.1)

    def test_alvo_td_e_terminal(self):
        v = {1: 4., 2: 6., 3: 999.}
        self.assertAlmostEqual(td_atualizar(v, (1, 0, 2, False), 0.5, 0.9), 1.4)
        self.assertAlmostEqual(v[1], 4.7)
        td_atualizar(v, (2, 1, 3, True), 1., 0.9)
        self.assertEqual(v[2], 1.)

    def test_caminhada_acaba_em_terminal_no_seed(self):
        passos = list(caminhada(Random(42)))
        self.assertTrue(passos[-1][3])
        self.assertIn(passos[-1][2], {0, 6})
        self.assertTrue(all(not passo[3] for passo in passos[:-1]))

    def test_comparacao_reproduzivel_e_aprende(self):
        curvas = comparar(50, 30)
        self.assertEqual(curvas, comparar(50, 30))
        self.assertLess(curvas[-1][0], curvas[0][0])
        self.assertLess(curvas[-1][1], curvas[0][1])

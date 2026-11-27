import unittest
from collections import Counter
from random import Random

from mdp import ACOES, BANDEIRA, BURACO, ESTADOS, INICIO, TERMINAIS, episodio, modelo, passo, retorno


class TestMDP(unittest.TestCase):
    def test_distribuicao_valida(self):
        for estado in set(ESTADOS) - TERMINAIS:
            for acao in ACOES:
                resultados = modelo(estado, acao)
                self.assertAlmostEqual(sum(p for p, *_ in resultados), 1)
                self.assertEqual(len(resultados), len({s for _, s, *_ in resultados}))
                self.assertTrue(all(p > 0 and s in ESTADOS for p, s, *_ in resultados))

    def test_colisoes_agregadas(self):
        resultados = dict((s, p) for p, s, *_ in modelo(INICIO, "B"))
        self.assertAlmostEqual(resultados[INICIO], 0.9)
        self.assertAlmostEqual(resultados[(2, 1)], 0.1)

    def test_terminais(self):
        self.assertEqual(modelo(INICIO, "C", 1), ((1.0, BURACO, -10, True),))
        self.assertEqual(modelo((0, 3), "D", 1), ((1.0, BANDEIRA, 10, True),))
        with self.assertRaises(ValueError):
            modelo(BURACO, "D")

    def test_amostras_respeitam_modelo(self):
        e = (1, 3)
        rng = Random(42)
        amostras = Counter(passo(e, "D", rng)[0] for _ in range(10_000))
        for p, novo, *_ in modelo(e, "D"):
            self.assertLess(abs(amostras[novo] / 10_000 - p), 0.02)

    def test_retorno(self):
        rs = [item[2] for item in episodio("DDDDCC", Random(42), aderencia=1)]
        self.assertAlmostEqual(retorno(rs), 1.8098)
        self.assertEqual(retorno(rs, 1), 5)
        self.assertGreater(retorno([2], 0.5), retorno([0] * 5 + [10], 0.5))
        self.assertLess(retorno([2], 0.9), retorno([0] * 5 + [10], 0.9))

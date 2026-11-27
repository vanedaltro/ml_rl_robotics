import unittest
from random import Random
from statistics import mean

from valores import (BANDEIRA, CORREDOR, DIRETA, INICIO, MISTA,
                      amostra_retorno, avaliar, valor_acao)


class TestValores(unittest.TestCase):
    def test_exemplo_62(self):
        v = avaliar(DIRETA)
        self.assertAlmostEqual(v[CORREDOR], 10)
        self.assertAlmostEqual(v[INICIO], 6.2)
        self.assertEqual(v[BANDEIRA], 0)

    def test_mista_por_formula_fechada(self):
        v = avaliar(MISTA)
        self.assertAlmostEqual(v[INICIO], (0.7 * 6.2 - 0.3) / (1 - 0.3 * 0.9))
        self.assertAlmostEqual(v[INICIO], sum(
            p * valor_acao(INICIO, a, v) for a, p in MISTA[INICIO].items()
        ))

    def test_q_condicionado_a_primeira_acao(self):
        rng = Random(42)
        estimativa = mean(amostra_retorno(MISTA, rng, acao_inicial="esperar") for _ in range(5000))
        self.assertLess(abs(estimativa - valor_acao(INICIO, "esperar", avaliar(MISTA))), 0.15)

    def test_limites(self):
        with self.assertRaises(ValueError):
            avaliar(DIRETA, gamma=1.0)
        with self.assertRaises(ValueError):
            avaliar({**DIRETA, INICIO: {"direita": 0.7}})
        with self.assertRaises(RuntimeError):
            amostra_retorno(DIRETA, Random(42), max_passos=0)

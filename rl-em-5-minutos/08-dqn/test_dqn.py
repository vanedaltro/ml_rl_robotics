import unittest
from random import Random

import torch

from dqn import (BURACO, INICIO, META, Replay, Transicao, alvos, atualizar,
                 rede_q, sincronizar, step, tensor_de_estados, treinar)


class TestDQN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_grid(self):
        self.assertEqual(step(INICIO, 0), (BURACO, -10, True))
        self.assertEqual(step((0, 3), 1), (META, 10, True))
        self.assertEqual(step((2, 2), 0), ((2, 2), -1, False))

    def test_um_valor_por_acao(self):
        self.assertEqual(rede_q()(tensor_de_estados([INICIO, META])).shape, (2, 4))

    def test_alvo_terminal_e_truncado(self):
        # O segundo caso representa um corte externo: o estado ainda tem futuro.
        y = alvos(torch.tensor([10., -1.]), torch.tensor([True, False]),
                  torch.tensor([[7., 100.], [2., 6.]]), gamma=.9)
        torch.testing.assert_close(y, torch.tensor([10., 4.4]))
        self.assertFalse(y.requires_grad)

    def test_replay_limitado(self):
        replay = Replay(2)
        ts = [Transicao(INICIO, i, -1, INICIO, False) for i in range(3)]
        for t in ts:
            replay.guardar(t)
        self.assertCountEqual(replay.amostra(2, Random(7)), ts[1:])

    def test_rede_alvo_so_muda_na_copia(self):
        torch.manual_seed(7)
        online, alvo = rede_q(), rede_q()
        sincronizar(alvo, online)
        antes = [p.clone() for p in alvo.parameters()]
        opt = torch.optim.SGD(online.parameters(), lr=.1)
        perda = atualizar(online, alvo, opt, [Transicao((0, 3), 1, 10, META, True)])
        self.assertGreater(perda, 0)
        self.assertTrue(any(not torch.equal(a, p) for a, p in zip(antes, online.parameters())))
        self.assertTrue(all(torch.equal(a, p) for a, p in zip(antes, alvo.parameters())))
        self.assertTrue(all(p.grad is None for p in alvo.parameters()))
        sincronizar(alvo, online)
        self.assertTrue(all(torch.equal(a, p) for a, p in zip(online.parameters(), alvo.parameters())))

    def test_treino_chega_a_meta(self):
        antes, depois, _, _ = treinar(episodios=250, seed=42)
        self.assertFalse(antes[1])
        self.assertTrue(depois[1])
        self.assertEqual(depois[0], 5.0)


if __name__ == "__main__":
    unittest.main()

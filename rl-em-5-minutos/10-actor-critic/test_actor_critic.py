import unittest

import torch

from actor_critic import neural_demo, perda_neural, rota_gulosa, softmax, td, train


class TestActorCritic(unittest.TestCase):
    def test_td_video(self):
        self.assertEqual(td(1, 4, 5, False), (5.5, 1.5))
        self.assertAlmostEqual(td(0, 4, 3, False)[1], -1.3)

    def test_terminal_alvo(self):
        self.assertEqual(td(1, 4, 99, True), (1, -3))

    def test_softmax_stable(self):
        self.assertAlmostEqual(sum(softmax([1000, 999])), 1)

    def test_actor_has_no_gradient_through_baseline(self):
        x = torch.zeros(1, 2, requires_grad=True)
        v = torch.ones(1, requires_grad=True)
        y = torch.tensor([2.], requires_grad=True)
        perda_neural(x, v, torch.tensor([0]), y, beta=0).backward()
        torch.testing.assert_close(v.grad, torch.tensor([-1.]))
        torch.testing.assert_close(x.grad, torch.tensor([[-.5, .5]]))
        self.assertIsNone(y.grad)

    def test_neural_gradients(self):
        self.assertEqual(neural_demo(), [True, True, True])

    def test_greedy_flag(self):
        logits, _, _ = train(300, 7)
        self.assertTrue(rota_gulosa(logits)[1])

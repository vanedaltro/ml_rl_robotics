import unittest

import torch
from torch.distributions import Categorical

from reinforce import exemplo_gaussiano, perda, retornos, train


class TestREINFORCE(unittest.TestCase):
    def test_return_to_go(self):
        self.assertEqual(retornos([-1, 10], 0.9), [8, 10])

    def test_positive_and_negative_advantage(self):
        for sign in (1.0, -1.0):
            x = torch.zeros(2, requires_grad=True)
            logp = Categorical(logits=x).log_prob(torch.tensor(0)).reshape(1)
            l = perda(logp, torch.tensor([sign]), torch.tensor([0.0]), .9)
            l.backward()
            self.assertAlmostEqual(x.grad[0].item(), -.5 * sign)
            self.assertAlmostEqual(x.grad[1].item(), .5 * sign)

    def test_baseline_does_not_receive_actor_gradient(self):
        x = torch.tensor([-.7], requires_grad=True)
        b = torch.ones(1, requires_grad=True)
        perda(x, torch.ones(1), b, .9).backward()
        self.assertIsNone(b.grad)
        self.assertEqual(x.grad.item(), 0)

    def test_discount_from_start(self):
        logps = torch.tensor([-.1, -.2], requires_grad=True)
        perda(logps, torch.ones(2), torch.zeros(2), .5).backward()
        torch.testing.assert_close(logps.grad, torch.tensor([-1., -.5]))

    def test_gaussian_score(self):
        self.assertEqual(exemplo_gaussiano(), .75)

    def test_small_run_reproducible(self):
        a, g = train(3, 5, True)
        b, h = train(3, 5, True)
        torch.testing.assert_close(a, b)
        self.assertEqual(g, h)

import unittest

import torch
from torch import nn

from lab_moe import SparseMoE, TinyCausalLM, routing


class TestSparseMoE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_top_k_counts_and_weights(self):
        logits = torch.tensor([[3., 2., -1., 0.], [0., 1., 8., 9.], [3., -1., 1., 2.]])
        route = routing(logits, 2)
        self.assertEqual(route.top.tolist(), [[0, 1], [3, 2], [0, 3]])
        self.assertEqual(route.counts.tolist(), [2, 1, 1, 2])
        torch.testing.assert_close(route.gates.sum(-1), torch.ones(3))
        self.assertEqual(int(route.counts.sum()), 6)

    def test_sparse_dispatch_matches_dense_reference_only_selected_tokens(self):
        class CountingExpert(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value, self.seen = value, 0

            def forward(self, x):
                self.seen += len(x)
                return x * self.value

        layer = SparseMoE(2, 4, n_experts=4, k=2)
        layer.experts = nn.ModuleList(CountingExpert(i + 1) for i in range(4))
        x = torch.tensor([[[1., 0.], [0., 1.]], [[2., 0.], [0., 2.]]])
        with torch.no_grad():
            layer.router.bias.zero_()
            layer.router.weight.copy_(torch.tensor([[8., 1.], [2., 6.], [1., 8.], [-1., -2.]]))
        sparse, scores = layer(x)
        self.assertEqual([e.seen for e in layer.experts], scores.counts.tolist())
        self.assertEqual(scores.counts.tolist(), [2, 4, 2, 0])
        flat = x.reshape(-1, 2)
        weights = torch.zeros((len(flat), 4)).scatter(1, scores.top, scores.gates)
        dense = (torch.stack([flat * i for i in range(1, 5)], dim=1) * weights[:, :, None]).sum(1)
        torch.testing.assert_close(sparse.reshape_as(flat), dense)

    def test_router_and_selected_expert_gradients(self):
        torch.manual_seed(7)
        layer = SparseMoE(4, 8, 4, 2)
        with torch.no_grad():
            layer.router.weight.zero_()
            layer.router.bias.copy_(torch.tensor([1.0, 0.3, -2.0, -3.0]))
        x = torch.randn(5, 4, requires_grad=True)
        out, route = layer(x)
        out.square().mean().backward()
        self.assertGreater(float(layer.router.weight.grad.abs().sum()), 0)
        for i, e in enumerate(layer.experts):
            for p in e.parameters():
                if i < 2:
                    self.assertGreater(float(p.grad.abs().sum()), 0)
                else:
                    self.assertIsNone(p.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertEqual(route.counts.tolist(), [5, 5, 0, 0])

    def test_balancing_penalizes_deliberate_collapse_and_has_gradients(self):
        a = torch.tensor([[8., 7., -8., -8.], [8., 7., -8., -8.]], requires_grad=True)
        b = torch.tensor([[8., 7., -8., -8.], [-8., -8., 8., 7.]])
        collapse, uniform = routing(a, 2), routing(b, 2)
        torch.testing.assert_close(uniform.balance, torch.tensor(1.))
        self.assertGreater(float(collapse.balance.detach()), 1.99)
        collapse.balance.backward()
        self.assertGreater(float(a.grad.abs().sum()), 0)

    def test_causal_prefix_does_not_see_suffix(self):
        a = torch.tensor([[0, 1, 5, 13]])
        b = torch.tensor([[0, 1, 5, 20]])
        for mode in ("dense", "moe_com_aux"):
            torch.manual_seed(7)
            model = TinyCausalLM(mode).eval()
            with torch.no_grad():
                ya, ra = model(a)
                yb, rb = model(b)
            torch.testing.assert_close(ya[:, :-1], yb[:, :-1], atol=1e-6, rtol=0)
            if ra is not None:
                self.assertTrue(torch.equal(ra.top[:-1], rb.top[:-1]))
            self.assertGreater(float((ya[:, -1] - yb[:, -1]).abs().mean()), 1e-5)

    def test_renormalized_top1_main_gate_constant(self):
        logit = torch.tensor([[2., 1., 0.]], requires_grad=True)
        route = routing(logit, 1)
        torch.testing.assert_close(route.gates, torch.ones_like(route.gates))
        route.gates.sum().backward()
        torch.testing.assert_close(logit.grad, torch.zeros_like(logit))


if __name__ == "__main__":
    unittest.main()

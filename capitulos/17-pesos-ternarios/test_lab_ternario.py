import unittest

import torch

from lab_ternario import ARCHES, Linear, Conv, build, dataset, forward_weight, ternary_codes


class TestTernary(unittest.TestCase):
    def test_exact_strict_forward_and_surrogate_gradient(self):
        w = torch.tensor([[-0.8, 0.01, 0.7, -0.02]], requires_grad=True)
        q = forward_weight(w, "strict")
        self.assertTrue(torch.equal(q, torch.tensor([[-1., 0., 1., 0.]])))
        q.sum().backward()
        self.assertTrue(torch.equal(w.grad, torch.ones_like(w)))

    def test_scaled_codes_share_nonzero_magnitude_per_row(self):
        w = torch.tensor([[-1., 0.01, 0.6], [3., -5., 0.]])
        q, a = ternary_codes(w)
        self.assertTrue(torch.allclose(a, torch.tensor([[0.8], [4.]])))
        self.assertTrue(torch.equal(forward_weight(w, "scaled"), a * q))

    def test_zero_rows_do_not_produce_nans(self):
        for mode in ("strict", "scaled"):
            z = torch.zeros(2, 4, requires_grad=True)
            self.assertTrue(torch.equal(forward_weight(z, mode), z))
            forward_weight(z, mode).sum().backward()
            self.assertTrue(torch.all(torch.isfinite(z.grad)))

    def test_same_master_initialization_and_all_projection_weights_ternary(self):
        torch.set_num_threads(1)
        for arch in ARCHES:
            torch.manual_seed(7)
            f = build(arch, "fp32")
            torch.manual_seed(7)
            q = build(arch, "strict")
            self.assertTrue(all(torch.equal(f.state_dict()[k], v) for k, v in q.state_dict().items()))
            for mod in q.modules():
                if isinstance(mod, (Linear, Conv)):
                    w = forward_weight(mod.weight, mod.mode)
                    self.assertTrue(((w == -1) | (w == 0) | (w == 1)).all().item())
            self.assertEqual(q(dataset(4, 42)[0]).shape, (4, 4))

    def test_balanced_reproducible_dataset(self):
        x, y = dataset(40, 2026)
        self.assertEqual(y.bincount().tolist(), [10, 10, 10, 10])
        self.assertTrue(torch.equal(x, dataset(40, 2026)[0]))
        self.assertFalse(torch.equal(x, dataset(40, 2027)[0]))


if __name__ == "__main__":
    unittest.main()

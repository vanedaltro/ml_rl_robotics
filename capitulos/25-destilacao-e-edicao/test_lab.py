import unittest
import torch
from torch.nn import functional as F
from lab import data, kd_loss, edit_demo, net

class LabTests(unittest.TestCase):
    def test_identical_distributions_and_detach(self):
        z = torch.tensor([[2., 1., 0.]], requires_grad=True)
        t = z.detach().clone().requires_grad_(True)
        loss = kd_loss(z, torch.tensor([0]), t, alpha=1.)
        self.assertAlmostEqual(loss.item(), 0., places=5)
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_hard_only(self):
        z = torch.tensor([[1., 2.]])
        y = torch.tensor([1])
        torch.testing.assert_close(kd_loss(z, y, z, alpha=0.),
                                   F.cross_entropy(z, y))

    def test_data_and_parameter_counts(self):
        x, y = data(48, 207)
        self.assertEqual(x.shape, (48, 20))
        self.assertEqual(y.shape, (48,))
        self.assertFalse(torch.equal(x, data(48, 307)[0]))
        self.assertEqual(sum(p.numel() for p in net(16).parameters()), 710)
        self.assertEqual(sum(p.numel() for p in net(64).parameters()), 5894)

    def test_edit(self):
        out = edit_demo()
        self.assertEqual(out["editado"], out["esperado"])
        self.assertEqual(sum(a == b for a, b in zip(out["original"], out["esperado"])), 2)

if __name__ == "__main__":
    unittest.main()

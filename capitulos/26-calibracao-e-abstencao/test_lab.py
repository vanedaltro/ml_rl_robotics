import unittest
import torch
import torch.nn.functional as F
from lab import images, expose, fit_temperature, outcome, policy_stats, select_tau, reliability

class LabTests(unittest.TestCase):
    def test_exposure_preserves_geometry_and_labels(self):
        x, y = images(16, 101)
        x2, y2 = images(16, 101)
        self.assertTrue(torch.equal(x, x2) and torch.equal(y, y2))
        self.assertEqual(x.shape, (16, 1, 20, 20))
        self.assertFalse(torch.equal(expose(x, "claro", 1), expose(x, "escuro", 1)))
        self.assertTrue(bool(y.ge(0).all()))
        self.assertTrue(bool(images(16, 101, unknown=True)[1].eq(-1).all()))

    def test_temperature_not_change_argmax(self):
        z = torch.tensor([[3., 1., -2.], [1., 4., 0.]])
        y = torch.tensor([0, 0])
        t = fit_temperature(z, y)
        self.assertTrue(.25 <= t <= 4.)
        self.assertLessEqual(F.cross_entropy(z/t, y).item(), F.cross_entropy(z, y).item() + 1e-6)
        self.assertTrue(torch.equal(outcome(z, y, t)[1], outcome(z, y)[1]))

    def test_abstention_accounting(self):
        c = torch.tensor([.99, .90, .60, .50])
        ok = torch.tensor([True, False, True, False])
        s = policy_stats(c, ok, .80)
        self.assertEqual((s["accepted_errors"], s["rejected_correct"], s["rejected_errors"]), (1, 1, 1))
        self.assertEqual(s["risk"], .50)
        self.assertIsNone(policy_stats(c, ok, 1.)["risk"])

    def test_selects_feasible_policy_or_none(self):
        c = torch.tensor([.95, .80, .70, .40])
        ok = torch.tensor([True, True, False, False])
        tau = select_tau(c, ok, cap=0., min_accept=2)
        self.assertEqual(policy_stats(c, ok, tau)["accepted"], 2)
        self.assertIsNone(select_tau(c, ok, cap=0., min_accept=3))

    def test_reliability_includes_confidence_one(self):
        b = reliability(torch.tensor([.27, .95, 1.]), torch.tensor([True, False, True]))
        self.assertEqual(sum(i["n"] for i in b), 3)
        self.assertEqual(b[-1]["n"], 2)

if __name__ == "__main__":
    unittest.main()

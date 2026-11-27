import unittest
import torch
from lab_finetuning import Adapter, base_model, dataset, fit, run


class TestLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identity_and_frozen_base(self):
        model = base_model().eval()
        adapter = Adapter(model).eval()
        x, _, y = dataset(64, 100)
        torch.testing.assert_close(model(x), adapter(x), rtol=0, atol=0)
        fit(adapter, x, y, 99, 8, .01)
        self.assertTrue(all(not p.requires_grad for p in adapter.base.parameters()))
        torch.testing.assert_close(model(x), adapter(x, active=False), rtol=0, atol=0)
        self.assertFalse(torch.equal(model(x), adapter(x)))

    def test_retention_and_adaptation(self):
        r = run(7)
        self.assertGreater(r["base"][0], .95)
        self.assertLess(r["full"][0], r["base"][0] - .35)
        self.assertGreater(r["replay"][0], r["full"][0])
        self.assertTrue(r["off_logits_exact"])
        self.assertEqual(r["adapter_routed"][0], r["base"][0])
        self.assertGreater(r["adapter_routed"][1], .9)

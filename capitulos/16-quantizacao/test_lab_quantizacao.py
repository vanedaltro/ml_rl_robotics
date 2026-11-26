"""Testes do laboratório de quantização."""
import unittest
import torch
from lab_quantizacao import make_data, make_model, quantize_weight, set_mode, symmetric_codes, ternary_codes, weight_stats


class QuantizacaoTest(unittest.TestCase):
    def test_int_ranges_and_scales(self):
        x = torch.tensor([[-1., 0., 1.], [-0.05, 0., 0.05]])
        for bits, bound in ((8, 127), (4, 7)):
            q, scale = symmetric_codes(x, bits)
            self.assertEqual(q.tolist(), [[-bound, 0, bound]] * 2)
            torch.testing.assert_close(scale, torch.tensor([[1 / bound], [.05 / bound]]))
            torch.testing.assert_close(q * scale, x)

    def test_zeros_and_ternary_symbols(self):
        zero = torch.zeros(4, 3, 2, 2)
        for mode in ("int8", "int4", "ternary"):
            q = quantize_weight(zero, mode)
            self.assertTrue(torch.isfinite(q).all())
            self.assertEqual(torch.count_nonzero(q).item(), 0)
        w = torch.tensor([[-2., .01, 1.]])
        q, scale = ternary_codes(w)
        self.assertEqual(q.tolist(), [[-1., 0., 1.]])
        self.assertEqual(scale.item(), 1.5)

    def test_ste_backward(self):
        w = torch.tensor([[-2., .01, 1.]], requires_grad=True)
        upstream = torch.tensor([[1., 2., -3.]])
        z = quantize_weight(w, "ternary")
        torch.testing.assert_close(z, torch.tensor([[-1.5, 0., 1.5]]))
        (z * upstream).sum().backward()
        torch.testing.assert_close(w.grad, upstream)

    def test_shapes_counts_and_fp32_activations(self):
        x, y = make_data(12, 1).tensors
        self.assertEqual(x.shape, (12, 1, 16, 16))
        self.assertEqual(y.bincount().tolist(), [3] * 4)
        for name in ("MLP", "CNN", "Transformer"):
            model = make_model(name)
            for mode in ("fp32", "int8", "int4", "ternary"):
                logits = set_mode(model, mode)(x)
                self.assertEqual(logits.shape, (12, 4))
                self.assertEqual(logits.dtype, torch.float32)
                self.assertTrue(torch.isfinite(logits).all())
                stats = weight_stats(model, mode)
                self.assertEqual(sum(stats["symbol_histogram"].values()), stats["quantizable_weights"])
                if mode == "ternary":
                    self.assertTrue(set(stats["symbol_histogram"]) <= {-1, 0, 1})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

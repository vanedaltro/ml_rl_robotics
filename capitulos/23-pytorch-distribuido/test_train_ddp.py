"""Testes locais. DDP_CHECKPOINT ativa a comparação com o treino distribuído."""
import math
import os
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset

from train_ddp import N_SAMPLES, SEED, dataset, model


class DDPTests(unittest.TestCase):
    def test_dataset_reproduzivel(self):
        for a, b in zip(dataset().tensors, dataset().tensors):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_shards_sem_repeticao_e_com_shuffle_por_epoca(self):
        data = dataset()
        a, b = (DistributedSampler(data, num_replicas=2, rank=r, seed=SEED) for r in (0, 1))
        first, other = list(a), list(b)
        self.assertEqual(len(first), N_SAMPLES // 2)
        self.assertEqual(len(other), N_SAMPLES // 2)
        self.assertEqual(set(first) | set(other), set(range(N_SAMPLES)))
        self.assertFalse(set(first) & set(other))
        a.set_epoch(1)
        self.assertNotEqual(first, list(a))

    @unittest.skipUnless(os.getenv("DDP_CHECKPOINT"), "forneça o checkpoint de dois ranks")
    def test_checkpoint_equivale_a_batch_global(self):
        # Vale para o padrão: dois ranks, 32 exemplos locais, SGD.
        torch.set_num_threads(1)
        device = "cpu"
        checkpoint = torch.load(os.environ["DDP_CHECKPOINT"], map_location=device, weights_only=True)
        data = dataset()
        torch.manual_seed(SEED)
        reference = model()
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for epoch in range(checkpoint["epoch"]):
            # DistributedSampler divide esta permutação em índices pares/ímpares.
            # Cada par de lotes de 32 representa o mesmo conjunto do lote de 64.
            rng = torch.Generator().manual_seed(SEED + epoch)
            order = torch.randperm(N_SAMPLES, generator=rng).tolist()
            for x, y in DataLoader(Subset(data, order), batch_size=64):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(reference(x), y)
                self.assertTrue(math.isfinite(loss.item()))
                loss.backward()
                optimizer.step()

        gap = 0.0
        for key, value in reference.state_dict().items():
            actual = checkpoint["model_state"][key]
            torch.testing.assert_close(actual, value, rtol=2e-5, atol=2e-6)
            gap = max(gap, (actual - value).abs().max().item())
        print(f"max_gap_vs_serial_global_batch_64={gap:.3e}", flush=True)


if __name__ == "__main__":
    unittest.main()

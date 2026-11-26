"""Supplement: no complete (rule, a, b) prompt overlaps train/test."""
import json
from pathlib import Path

import torch

from lab_moe import MODES, fit, mean_sd


def split(seed=2028):
    g = torch.Generator().manual_seed(seed)
    train_x, train_y, test_x, test_y = [], [], [], []
    for d in range(4):
        p = torch.randperm(64, generator=g)
        a, b = p // 8, p % 8
        inputs = torch.stack([torch.zeros_like(p), torch.full_like(p, d + 1), 5 + a, 13 + b], dim=1)
        target = (a+b, a-b, a^b, 2*a+b)[d] % 8
        train_ids = torch.randint(48, (512,), generator=g)
        train_x.append(inputs[train_ids])
        train_y.append(target[train_ids])
        test_x.append(inputs[48:])
        test_y.append(target[48:])
    train, test = (torch.cat(train_x), torch.cat(train_y)), (torch.cat(test_x), torch.cat(test_y))
    overlap = set(map(tuple, train[0].tolist())) & set(map(tuple, test[0].tolist()))
    assert not overlap
    return train, test


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train, test = split()
    rs = []
    for mode, coeff in MODES:
        for seed in (7, 23, 42):
            rs.append(fit(mode, seed, train, test, 12, coeff))
    summary = mean_sd(rs)
    print(json.dumps(summary, indent=2))
    Path("resultados-combinacoes-novas.json").write_text(
        json.dumps({"train_n": len(train[0]), "test_n": len(test[0]), "split_seed": 2028,
                    "summary": summary, "runs": rs}, indent=2) + "\n")

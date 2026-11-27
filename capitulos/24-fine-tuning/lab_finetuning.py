"""CPU-only controlled example of sequential adaptation and forgetting."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def dataset(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    a = (x[:, 0] + 0.2 * x[:, 2] > 0).long()
    b = (-0.8 * x[:, 0] + x[:, 1] + 0.2 * x[:, 2] > 0).long()
    return x, a, b


def base_model():
    return nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))


class Adapter(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.base = copy.deepcopy(base).requires_grad_(False)
        self.down = nn.Linear(32, rank, bias=False)
        self.up = nn.Linear(rank, 32, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, active=True):
        h = self.base[1](self.base[0](x))
        if active:
            h = h + self.up(F.gelu(self.down(h)))
        return self.base[2](h)


def fit(model, x, y, seed, steps, lr, replay=None):
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=0,
    )
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        i = torch.randint(len(x), (64,), generator=g)
        loss = F.cross_entropy(model(x[i]), y[i])
        if replay is not None:
            rx, ry, weight = replay
            j = torch.randint(len(rx), (64,), generator=g)
            loss = (1 - weight) * loss + weight * F.cross_entropy(model(rx[j]), ry[j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


@torch.inference_mode()
def score(model, x, y, **kwargs):
    return (model(x, **kwargs).argmax(dim=-1) == y).float().mean().item()


def run(seed):
    torch.manual_seed(seed)
    xa, ya, _ = dataset(1536, 10_000 + seed)
    xb, _, yb = dataset(1536, 20_000 + seed)
    xt, ya_test, yb_test = dataset(2048, 30_000 + seed)
    base = fit(base_model(), xa, ya, seed + 6, 160, 0.01)
    full = fit(copy.deepcopy(base), xb, yb, seed + 7, 160, 0.008)
    replay = fit(copy.deepcopy(base), xb, yb, seed + 7, 160, 0.008,
                 replay=(xa[:256], ya[:256], 0.5))
    adapter = fit(Adapter(base), xb, yb, seed + 7, 160, 0.008)
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    with torch.inference_mode():
        same = torch.equal(base(xt), adapter(xt, active=False))
    def scores(m):
        return [score(m, xt, ya_test), score(m, xt, yb_test)]
    return {
        "seed": seed, "base": scores(base), "full": scores(full),
        "replay": scores(replay), "adapter_on": scores(adapter),
        "adapter_routed": [score(adapter, xt, ya_test, active=False), score(adapter, xt, yb_test)],
        "off_logits_exact": same, "full_params": count(full), "adapter_params": count(adapter),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 23, 42])
    p.add_argument("--output", type=Path, default=Path("resultados.json"))
    args = p.parse_args()
    torch.set_num_threads(1)
    result = {"torch": torch.__version__, "device": "cpu",
              "runs": [run(seed) for seed in args.seeds]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Método            A (%)   B (%)")
    for method in ("base", "full", "replay", "adapter_on", "adapter_routed"):
        a, b = (100 * sum(r[method][k] for r in result["runs"]) / len(args.seeds)
                for k in range(2))
        print(f"{method:16} {a:6.2f}  {b:6.2f}")


if __name__ == "__main__":
    main()

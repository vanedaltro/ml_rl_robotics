"""Ablation of exact vs scaled ternary weights in PyTorch, CPU only."""
import argparse
import json
import platform
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODES = ("fp32", "strict", "scaled")
ARCHES = ("mlp", "cnn", "transformer")


@torch.no_grad()
def ternary_codes(weight):
    """A separate threshold per output row or convolutional filter."""
    dims = tuple(range(1, weight.ndim))
    mag = weight.abs()
    keep = mag > 0.7 * mag.mean(dim=dims, keepdim=True)
    code = weight.sign() * keep
    scale = (mag * keep).sum(dim=dims, keepdim=True) / keep.sum(dim=dims, keepdim=True).clamp_min(1)
    return code, scale


def forward_weight(weight, mode):
    if mode == "fp32":
        return weight
    code, scale = ternary_codes(weight)
    if mode == "scaled":
        code = code * scale
    elif mode != "strict":
        raise ValueError(mode)
    # Forward exactly equals code (or scale*code), not up to cancellation.
    # Backward approximates dQ/dW with the identity (STE).
    return code + (weight - weight.detach())


class Linear(nn.Linear):
    def __init__(self, n_in, n_out, mode):
        super().__init__(n_in, n_out, bias=False)
        self.mode = mode

    def forward(self, x):
        return F.linear(x, forward_weight(self.weight, self.mode))


class Conv(nn.Conv2d):
    def __init__(self, n_in, n_out, mode):
        super().__init__(n_in, n_out, 3, padding=1, bias=False)
        self.mode = mode

    def forward(self, x):
        return F.conv2d(x, forward_weight(self.weight, self.mode), padding=1)


class Transformer(nn.Module):
    """One pre-norm encoder block. Position embeddings and LayerNorm stay FP32."""
    def __init__(self, mode):
        super().__init__()
        self.patch = Linear(16, 24, mode)
        self.pos = nn.Parameter(torch.empty(1, 16, 24))
        nn.init.normal_(self.pos, std=0.02)
        self.norm1, self.norm2 = nn.LayerNorm(24), nn.LayerNorm(24)
        self.qkv, self.proj = Linear(24, 72, mode), Linear(24, 24, mode)
        self.ff = nn.Sequential(Linear(24, 48, mode), nn.GELU(), Linear(48, 24, mode))
        self.out = Linear(24, 4, mode)

    def forward(self, images):
        x = self.patch(F.unfold(images, kernel_size=4, stride=4).transpose(1, 2)) + self.pos
        b, n, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, 4, 6).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(att.transpose(1, 2).reshape(b, n, 24))
        x = x + self.ff(self.norm2(x))
        return self.out(x.mean(dim=1))


def build(arch, mode):
    if arch == "mlp":
        return nn.Sequential(nn.Flatten(), Linear(256, 64, mode), nn.ReLU(),
                             Linear(64, 32, mode), nn.ReLU(), Linear(32, 4, mode))
    if arch == "cnn":
        return nn.Sequential(Conv(1, 8, mode), nn.ReLU(), nn.MaxPool2d(2),
                             Conv(8, 16, mode), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), Linear(256, 4, mode))
    if arch == "transformer":
        return Transformer(mode)
    raise ValueError(arch)


def dataset(n, seed):
    """Four equally frequent geometric patterns, translated and corrupted."""
    if n % 4:
        raise ValueError("n must be a multiple of 4")
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 4
    y = y[torch.randperm(n, generator=g)]
    off = torch.randint(-4, 5, (n, 2), generator=g)
    r = torch.arange(16)[None, :, None] - 7 - off[:, 0, None, None]
    c = torch.arange(16)[None, None, :] - 7 - off[:, 1, None, None]
    masks = torch.stack([(r.abs() <= 1) & (c.abs() <= 5),
                         (c.abs() <= 1) & (r.abs() <= 5),
                         ((r - c).abs() <= 1) & (r.abs() <= 5) & (c.abs() <= 5),
                         ((r + c).abs() <= 1) & (r.abs() <= 5) & (c.abs() <= 5)], dim=1)
    signal = masks[torch.arange(n), y].unsqueeze(1)
    x = signal * (0.45 + 0.65 * torch.rand(n, 1, 1, 1, generator=g))
    x = x + 0.45 * torch.randn(n, 1, 16, 16, generator=g)
    return x, y


@torch.no_grad()
def evaluate(model, split):
    model.eval()
    x, y = split
    preds = torch.cat([model(b) for b in x.split(128)])
    return {"accuracy_pct": float(100 * (preds.argmax(1) == y).double().mean()),
            "cross_entropy": float(F.cross_entropy(preds, y)),
            "mean_abs_logit": float(preds.abs().mean())}


@torch.no_grad()
def weights_stats(model, mode):
    projected = [m for m in model.modules() if isinstance(m, (Linear, Conv))]
    c = torch.cat([ternary_codes(m.weight)[0].flatten() for m in projected])
    w = torch.cat([forward_weight(m.weight, mode).flatten() for m in projected])
    return {"rms_forward_weight": float(w.square().mean().sqrt()),
            "quantized_weights": int(len(w)) if mode != "fp32" else 0,
            "pct_zero_codes": float(100 * (c == 0).double().mean()) if mode != "fp32" else None}


def fit(arch, mode, seed, train, test, epochs):
    torch.manual_seed(seed)
    model = build(arch, mode)
    initial = evaluate(model, test)
    initial.update(weights_stats(model, mode))
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    x, y = train
    g = torch.Generator().manual_seed(seed + 10000)
    for _ in range(epochs):
        model.train()
        for ids in torch.randperm(len(y), generator=g).split(128):
            opt.zero_grad()
            loss = F.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            opt.step()
    out = evaluate(model, test)
    out.update(weights_stats(model, mode))
    return {"arch": arch, "mode": mode, "seed": seed, "epochs": epochs,
            "initial": initial, "final": out}


def mean_sd(results):
    summary = []
    for arch in ARCHES:
        for mode in MODES:
            rows = [r for r in results if (r["arch"], r["mode"]) == (arch, mode)]
            if not rows:
                continue
            row = {"arch": arch, "mode": mode, "seeds": len(rows)}
            for name, path in (("accuracy_pct", "final"), ("cross_entropy", "final"),
                               ("mean_abs_logit", "initial"), ("rms_forward_weight", "initial"),
                               ("pct_zero_codes", "final")):
                xs = [r[path][name] for r in rows if r[path][name] is not None]
                if xs:
                    row[f"{path}_{name}_mean"] = statistics.mean(xs)
                    row[f"{path}_{name}_sd"] = statistics.stdev(xs) if len(xs) > 1 else None
            summary.append(row)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 42])
    p.add_argument("--arch", nargs="+", choices=ARCHES, default=ARCHES)
    p.add_argument("--out", type=Path, default=Path("resultados.json"))
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train, test = dataset(1536, 2026), dataset(512, 2027)
    results = []
    for arch in args.arch:
        for mode in MODES:
            for seed in args.seeds:
                r = fit(arch, mode, seed, train, test, args.epochs)
                print(f"{arch:12} {mode:6} seed={seed:2} accuracy={r['final']['accuracy_pct']:6.2f}%", flush=True)
                results.append(r)
    meta = {"pytorch": torch.__version__, "platform": platform.platform(), "threads": 1,
            "train_seed": 2026, "test_seed": 2027, "train_n": 1536, "test_n": 512,
            "epoch_count": args.epochs, "batch_size": 128, "learning_rate": 0.003, "optimizer": "AdamW"}
    args.out.write_text(json.dumps({"metadata": meta, "summary": mean_sd(results),
                                    "runs": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

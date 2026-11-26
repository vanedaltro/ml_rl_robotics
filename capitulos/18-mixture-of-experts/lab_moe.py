"""Sparse top-k token routing integrated into a tiny causal Transformer."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class Route(NamedTuple):
    top: torch.Tensor       # [T, k]: hard expert IDs
    gates: torch.Tensor     # [T, k]: weights that sum to one
    counts: torch.Tensor    # [E]: number of token/expert assignments
    balance: torch.Tensor  # scalar: differentiable through mean router probabilities


def routing(logits: torch.Tensor, k: int) -> Route:
    """Router scores [T,E] -> top-k dispatch; expects a nonempty token batch."""
    n, n_experts = logits.shape
    if n == 0 or not 1 <= k <= n_experts:
        raise ValueError("Routing requires T > 0 and 1 <= k <= number of experts")
    scores, top = logits.topk(k, dim=-1)
    gates = scores.softmax(dim=-1, dtype=torch.float32).to(logits.dtype)
    counts = torch.bincount(top.reshape(-1), minlength=n_experts)
    fraction = counts.float() / (n * k)
    probs = logits.softmax(dim=-1, dtype=torch.float32).mean(dim=0)
    return Route(top, gates, counts, n_experts * (fraction * probs).sum())


def ffn(d_model, width):
    # Bias-free FFNs simplify the analytic count 2*d_model*width.
    return nn.Sequential(nn.Linear(d_model, width, bias=False), nn.GELU(),
                         nn.Linear(width, d_model, bias=False))


class SparseMoE(nn.Module):
    def __init__(self, d_model, width, n_experts=4, k=2, initial_bias=0.0):
        super().__init__()
        if not 1 <= k <= n_experts:
            raise ValueError("Expected 1 <= k <= n_experts")
        self.k = k
        self.router = nn.Linear(d_model, n_experts, bias=True)
        self.experts = nn.ModuleList(ffn(d_model, width) for _ in range(n_experts))
        if initial_bias:
            with torch.no_grad():
                self.router.bias.zero_()
                self.router.bias[:k].fill_(initial_bias)

    def forward(self, x):
        flat = x.reshape(-1, x.size(-1))
        route = routing(self.router(flat), self.k)
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token, slot = torch.where(route.top == expert_id)
            if token.numel() == 0:
                continue
            picked = flat.index_select(0, token)
            weighted = expert(picked) * route.gates[token, slot, None]
            # The same token occurs for k experts, so accumulate, don't overwrite.
            out = out.index_add(0, token, weighted.to(out.dtype))
        return out.reshape_as(x), route


class TinyCausalLM(nn.Module):
    """One attention block; next-token loss is applied at the answer position."""
    def __init__(self, mode, d_model=32, width=64):
        super().__init__()
        self.d_model = d_model
        self.tok = nn.Embedding(21, d_model)
        self.pos = nn.Embedding(4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 8, bias=False)
        self.ff = ffn(d_model, width) if mode == "dense" else SparseMoE(
            d_model, width, n_experts=4, k=2, initial_bias=2.0)

    def forward(self, ids):
        b, n = ids.shape
        h = self.tok(ids) + self.pos(torch.arange(n, device=ids.device))
        q, k, v = self.qkv(self.norm1(h)).reshape(b, n, 3, 4, self.d_model // 4).permute(2, 0, 3, 1, 4)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.out_proj(z.transpose(1, 2).reshape(b, n, self.d_model))
        z = self.norm2(h)
        if isinstance(self.ff, SparseMoE):
            z, route = self.ff(z)
        else:
            z, route = self.ff(z), None
        return self.head(self.norm3(h + z)), route


def dataset(n, seed):
    """Prompt [BOS, domain, operand a, operand b] -> response token (0..7)."""
    g = torch.Generator().manual_seed(seed)
    domain = torch.randint(4, (n,), generator=g)
    a = torch.randint(8, (n,), generator=g)
    b = torch.randint(8, (n,), generator=g)
    ids = torch.stack([torch.zeros_like(a), 1 + domain, 5 + a, 13 + b], dim=-1)
    rules = torch.stack([a + b, a - b, a ^ b, 2 * a + b], dim=-1)
    targets = rules.gather(1, domain[:, None]).squeeze(1) % 8
    return ids, targets


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    xs, ys = data
    correct, loss = 0, 0.0
    counts = torch.zeros(4, dtype=torch.long)
    for x, y in zip(xs.split(128), ys.split(128)):
        logits, route = model(x)
        last = logits[:, -1, :]
        correct += int((last.argmax(-1) == y).sum())
        loss += float(F.cross_entropy(last, y, reduction="sum"))
        if route is not None:
            counts += route.counts
    total = int(counts.sum())
    return {"accuracy_pct": 100 * correct / len(ys), "cross_entropy": loss / len(ys),
            "routing_counts": counts.tolist() if total else None,
            "routing_pct": (100 * counts.double() / total).tolist() if total else None,
            "routed_token_expert_pairs": total if total else None}


def fit(mode, seed, train, test, epochs, balance_coef):
    torch.manual_seed(seed)
    model = TinyCausalLM(mode)
    initial = evaluate(model, test)
    opt = torch.optim.AdamW(model.parameters(), lr=0.004)
    xs, ys = train
    g = torch.Generator().manual_seed(seed + 10000)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(ys), generator=g).split(128):
            opt.zero_grad(set_to_none=True)
            logits, route = model(xs[idx])
            loss = F.cross_entropy(logits[:, -1, :], ys[idx])
            if route is not None:
                loss = loss + balance_coef * route.balance
            loss.backward()
            opt.step()
    return {"mode": mode, "seed": seed, "balance_coefficient": balance_coef,
            "parameters": sum(p.numel() for p in model.parameters()),
            "ffn_parameters_total": sum(p.numel() for p in model.ff.parameters()) if mode == "dense" else
                sum(p.numel() for e in model.ff.experts for p in e.parameters()),
            "ffn_parameters_active_per_token": 2 * 32 * 64 * (1 if mode == "dense" else 2),
            "initial": initial, "final": evaluate(model, test)}


MODES = (("dense", 0.0), ("moe_sem_aux", 0.0), ("moe_com_aux", 0.2))


def mean_sd(results):
    out = []
    for mode, _ in MODES:
        runs = [r for r in results if r["mode"] == mode]
        if not runs:
            continue
        acc = [r["final"]["accuracy_pct"] for r in runs]
        others = [r["final"]["routing_pct"] for r in runs if r["final"]["routing_pct"]]
        out.append({"mode": mode, "acc_mean": statistics.mean(acc),
                    "acc_sd": statistics.stdev(acc) if len(acc) > 1 else None,
                    "nll_mean": statistics.mean(r["final"]["cross_entropy"] for r in runs),
                    "routing_pct_mean": [statistics.mean(e) for e in zip(*others)] if others else None})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 42])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("resultados.json"))
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train, test = dataset(2048, 2026), dataset(512, 2027)
    rs = []
    for mode, alpha in MODES:
        for seed in args.seeds:
            row = fit(mode, seed, train, test, args.epochs, alpha)
            print(mode, seed, row["final"], flush=True)
            rs.append(row)
    info = {"pytorch": torch.__version__, "platform": platform.platform(),
            "epochs": args.epochs, "learning_rate": 0.004, "optimizer": "AdamW",
            "train": [2048, 2026], "test": [512, 2027], "batch_size": 128, "threads": 1}
    args.out.write_text(json.dumps({"metadata": info, "summary": mean_sd(rs), "runs": rs},
                                  indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

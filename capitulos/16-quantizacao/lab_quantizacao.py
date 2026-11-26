"""Experimento CPU: FP32, PTQ simulada e treinamento ternário com PyTorch."""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def symmetric_codes(w, bits):
    """Quantização simétrica por canal de saída (pesos somente)."""
    if bits not in (4, 8):
        raise ValueError("bits deve ser 4 ou 8")
    flat = w.detach().flatten(1)
    qmax = (1 << (bits - 1)) - 1
    peak = flat.abs().amax(1, keepdim=True)
    scale = torch.where(peak > 0, peak / qmax, 1.0)
    # O dtype dos códigos é float: este laboratório não armazena low-bit.
    q = (flat / scale).round().clamp(-qmax, qmax)
    return q.reshape_as(w), scale.reshape((-1,) + (1,) * (w.ndim - 1))


def ternary_codes(w, threshold_factor=0.7):
    """Sinal ou zero, com escala positiva aprendida indiretamente via w."""
    if threshold_factor < 0:
        raise ValueError("threshold_factor deve ser não negativo")
    flat = w.detach().flatten(1)
    keep = flat.abs() > threshold_factor * flat.abs().mean(1, keepdim=True)
    code = flat.sign() * keep
    scale = (flat.abs() * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True).clamp_min(1)
    scale = torch.where(scale > 0, scale, 1.0)
    return code.reshape_as(w), scale.reshape((-1,) + (1,) * (w.ndim - 1))


def quantize_weight(w, mode):
    if mode in ("int8", "int4"):
        q, scale = symmetric_codes(w, int(mode[3:]))
        return q * scale
    if mode == "ternary":
        q, scale = ternary_codes(w)
        quantized = q * scale
        # Forward com valores discretizados e reescalados; backward identidade.
        return quantized + (w - w.detach())
    if mode == "fp32":
        return w
    raise ValueError(f"modo desconhecido: {mode}")


class QLinear(nn.Linear):
    mode = "fp32"

    def forward(self, x):
        return F.linear(x, quantize_weight(self.weight, self.mode), self.bias)


class QConv2d(nn.Conv2d):
    mode = "fp32"

    def forward(self, x):
        return self._conv_forward(x, quantize_weight(self.weight, self.mode), self.bias)


def set_mode(model, mode):
    for layer in model.modules():
        if isinstance(layer, (QLinear, QConv2d)):
            layer.mode = mode
    return model


class PatchTransformer(nn.Module):
    """Um bloco encoder, sem máscara, com média dos tokens para classificação."""
    def __init__(self):
        super().__init__()
        d = 24
        self.patch = QConv2d(1, d, kernel_size=4, stride=4)
        self.pos = nn.Parameter(torch.zeros(1, 16, d))
        nn.init.normal_(self.pos, std=0.02)
        self.norm1 = nn.LayerNorm(d)
        self.qkv = QLinear(d, 3 * d)
        self.attn_proj = QLinear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(QLinear(d, 2 * d), nn.GELU(), QLinear(2 * d, d))
        self.head = nn.Sequential(nn.LayerNorm(d), QLinear(d, 4))

    def forward(self, x):
        z = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        b, n, d = z.shape
        # Quatro cabeças; atenção e softmax permanecem em FP32.
        q, k, v = self.qkv(self.norm1(z)).reshape(b, n, 3, 4, d // 4).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v)
        z = z + self.attn_proj(attn.transpose(1, 2).reshape(b, n, d))
        z = z + self.mlp(self.norm2(z))
        return self.head(z.mean(dim=1))


def make_model(name):
    if name == "MLP":
        return nn.Sequential(nn.Flatten(), QLinear(256, 48), nn.GELU(), QLinear(48, 4))
    if name == "CNN":
        return nn.Sequential(
            QConv2d(1, 12, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            QConv2d(12, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), QLinear(16 * 4 * 4, 4),
        )
    if name == "Transformer":
        return PatchTransformer()
    raise ValueError(f"arquitetura desconhecida: {name}")


def make_data(n, seed):
    """Classes: faixa horizontal, vertical, diagonal principal, antidiagonal."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 4
    y = y[torch.randperm(n, generator=g)]
    i = torch.arange(16).view(1, 16, 1).float()
    j = torch.arange(16).view(1, 1, 16).float()
    di = torch.randint(-3, 4, (n, 1, 1), generator=g)
    dj = torch.randint(-3, 4, (n, 1, 1), generator=g)
    u, v = i - (7.5 + di), j - (7.5 + dj)
    length = torch.randint(5, 8, (n, 1, 1), generator=g)
    boxes = (u.abs() < length) & (v.abs() < length)
    masks = torch.stack(tuple(expr.expand(-1, 16, 16) for expr in (
        u.abs() < 1.25, v.abs() < 1.25, (u - v).abs() < 1.8, (u + v).abs() < 1.8,
    )), dim=1)
    signal = masks[torch.arange(n), y] & boxes
    amplitude = 0.45 + 0.55 * torch.rand((n, 1, 1), generator=g)
    noise = 0.43 * torch.randn((n, 16, 16), generator=g)
    x = (signal * amplitude + noise).clamp(-1.0, 1.5).float().unsqueeze(1)
    return TensorDataset(x.contiguous(), y)


def fit(model, train, epochs, seed):
    batches = DataLoader(train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    model.train()
    for _ in range(epochs):
        for x, y in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def accuracy(model, data):
    model.eval()
    total = sum(int((model(x).argmax(1) == y).sum())
                for x, y in DataLoader(data, batch_size=128))
    return total / len(data)


@torch.no_grad()
def weight_stats(model, mode):
    histogram = Counter()
    error_sq = weight_sq = 0.0
    for layer in model.modules():
        if not isinstance(layer, (QLinear, QConv2d)):
            continue
        w = layer.weight
        if mode == "fp32":
            # FP32 não usa um alfabeto discreto no experimento.
            histogram["fp32"] += w.numel()
        else:
            q, scale = (ternary_codes(w) if mode == "ternary"
                        else symmetric_codes(w, int(mode[3:])))
            symbols, counts = q.to(torch.int16).unique(return_counts=True)
            histogram.update({int(k): int(v) for k, v in zip(symbols, counts)})
            error_sq += ((q * scale - w).double().square().sum().item())
        weight_sq += w.double().square().sum().item()
    return {
        "quantizable_weights": sum(histogram.values()),
        "symbol_histogram": dict(sorted(histogram.items())),
        "relative_l2_error": (error_sq / weight_sq) ** .5 if weight_sq else 0.0,
        "zero_symbol_fraction": histogram.get(0, 0) / sum(histogram.values()) if mode != "fp32" else None,
    }


def run(seeds, epochs, n_train=1536, n_test=512):
    train, test = make_data(n_train, 2026), make_data(n_test, 2027)
    result = []
    for seed in seeds:
        for architecture in ("MLP", "CNN", "Transformer"):
            torch.manual_seed(seed)
            fp32 = make_model(architecture)
            torch.manual_seed(seed)
            ternary = set_mode(make_model(architecture), "ternary")
            fit(fp32, train, epochs, seed)
            fit(ternary, train, epochs, seed)
            for mode, model in (("fp32", fp32), ("int8", fp32), ("int4", fp32), ("ternary", ternary)):
                set_mode(model, mode)
                metrics = {"seed": seed, "architecture": architecture, "mode": mode,
                           "held_out_accuracy": accuracy(model, test), **weight_stats(model, mode)}
                result.append(metrics)
                print(f"{seed:3} {architecture:11} {mode:7} "
                      f"teste={metrics['held_out_accuracy']:.2%} "
                      f"erro L2 relativo={metrics['relative_l2_error']:.4f}", flush=True)
    return {
        "torch": torch.__version__, "device": "cpu", "epochs": epochs, "seeds": seeds,
        "n_train": n_train, "n_test": n_test, "train_data_seed": 2026, "test_data_seed": 2027,
        "weights_only": True, "real_low_bit_kernels": False,
        "relative_l2_reference": "pesos FP32 treinados originais em PTQ; pesos-sombra treinados no ternário",
        "runs": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 42])
    parser.add_argument("--out", type=Path, default=Path("resultados.json"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs deve ser pelo menos 1")
    torch.set_num_threads(1)
    args.out.write_text(json.dumps(run(args.seeds, args.epochs), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

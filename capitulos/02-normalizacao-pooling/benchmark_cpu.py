"""Compara blocos da mesma CNN, na CPU, com três repetições pareadas."""
import argparse
import csv
import random
import statistics
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval
from torch.utils.benchmark import Timer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CONFIGS = {
    "sem_norm+max": ("none", "max"),
    "batch+max": ("batch", "max"),
    "layer_c+max": ("layer", "max"),
    "group+max": ("group", "max"),
    "batch+avg": ("batch", "avg"),
    "batch+stride": ("batch", "stride"),
}

class LayerNormC(nn.Module):
    """Normaliza pelos canais de cada posição, por amostra, em NCHW."""
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)
    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

def bloco(in_c, out_c, norm, pool):
    normaliza = {
        "none": nn.Identity,
        "batch": lambda: nn.BatchNorm2d(out_c),
        "layer": lambda: LayerNormC(out_c),
        "group": lambda: nn.GroupNorm(4, out_c),
    }[norm]()
    reduz = {"max": lambda: nn.MaxPool2d(2),
             "avg": lambda: nn.AvgPool2d(2),
             "stride": nn.Identity}[pool]()
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=2 if pool == "stride" else 1, padding=1),
        normaliza, nn.ReLU(), reduz,
    )

def modelo(nome):
    norm, pool = CONFIGS[nome]
    return nn.Sequential(bloco(1, 16, norm, pool), bloco(16, 32, norm, pool),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

def loader(base, ixs, shuffle, batch, seed):
    return DataLoader(Subset(base, ixs), batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)

def treino(net, data, epochs):
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0)
    net.train()
    t0 = time.perf_counter()
    for _ in range(epochs):
        for x, y in data:
            optim.zero_grad(set_to_none=True)
            criterion(net(x), y).backward()
            optim.step()
    return time.perf_counter() - t0

@torch.inference_mode()
def validar(net, data):
    net.eval()
    certos = sum(int((net(x).argmax(1) == y).sum()) for x, y in data)
    return certos / len(data.dataset)

def latencia(net, x, threads, min_time):
    net.eval()
    @torch.inference_mode()
    def forward():
        return net(x)
    for _ in range(12):
        forward()
    return 1000 * Timer(stmt="forward()", globals={"forward": forward},
                        num_threads=threads).blocked_autorange(
                            min_run_time=min_time).median

def fundir(net):
    out = deepcopy(net).eval()
    for block in (out[0], out[1]):
        block[0] = fuse_conv_bn_eval(block[0], block[1])
        block[1] = nn.Identity()
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dados", type=Path, default=Path("dados"))
    p.add_argument("--saida", type=Path, default=Path("benchmark_cpu.csv"))
    p.add_argument("--epocas", type=int, default=2)
    p.add_argument("--treino", type=int, default=6000)
    p.add_argument("--val", type=int, default=2000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--min-time", type=float, default=0.18)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    args = p.parse_args()
    if min(args.epocas, args.treino, args.val, args.threads) < 1 or args.treino + args.val > 60000 or args.min_time <= 0:
        p.error("Parâmetros de benchmark inválidos.")
    torch.set_num_threads(args.threads)
    base = datasets.FashionMNIST(args.dados, train=True, download=True,
                                transform=transforms.ToTensor())
    ixs = torch.randperm(len(base), generator=torch.Generator().manual_seed(2026)).tolist()
    i_tr = ixs[:args.treino]
    i_va = ixs[args.treino : args.treino + args.val]
    x1 = torch.randn(1, 1, 28, 28, generator=torch.Generator().manual_seed(876))
    x32 = torch.randn(32, 1, 28, 28, generator=torch.Generator().manual_seed(876))
    rows = []
    for seed in args.seeds:
        order = list(CONFIGS)
        random.Random(seed).shuffle(order)
        for nome in order:
            torch.manual_seed(seed)
            net = modelo(nome)
            duration = treino(net, loader(base, i_tr, True, 128, seed), args.epocas)
            accuracy = validar(net, loader(base, i_va, False, 128, seed))
            small, bulk = [latencia(net, x, args.threads, args.min_time) for x in (x1, x32)]
            rows.append([seed,nome,accuracy,duration,small,bulk,None])
            print(f"{seed:3} {nome:12} val={accuracy:.2%} treino={duration:.2f}s B1={small:.4f}ms B32={bulk:.4f}ms", flush=True)
            if nome == "batch+max":
                fused = fundir(net)
                with torch.inference_mode():
                    diff = (net(x32) - fused(x32)).abs().max().item()
                    torch.testing.assert_close(net(x32), fused(x32), atol=4e-5, rtol=1e-5)
                fused_acc = validar(fused, loader(base, i_va, False, 128, seed))
                tf1, tf32 = [latencia(fused, x, args.threads, args.min_time) for x in (x1, x32)]
                rows.append([seed,"batch_fundida",fused_acc,None,tf1,tf32,diff])
                print(f"{seed:3} {'batch_fundida':12} val={fused_acc:.2%} B1={tf1:.4f}ms B32={tf32:.4f}ms diff={diff:.2g}", flush=True)
    with args.saida.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed","modelo","val_acc","treino_s","b1_ms","b32_ms","delta_logits"]); w.writerows(rows)
    print(f"CPU | Torch {torch.__version__} | threads {args.threads} | epochs {args.epocas} | treino/val {args.treino}/{args.val}")
    for k in [*CONFIGS, "batch_fundida"]:
        group = [r for r in rows if r[1] == k]
        md = [statistics.median(z) if (z := [r[c] for r in group if r[c] is not None]) else None for c in range(2, 7)]
        print(k, *[None if x is None else round(x, 6) for x in md])

if __name__ == "__main__":
    main()

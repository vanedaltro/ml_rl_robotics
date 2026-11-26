"""Laboratório: gradientes, treino e latência de uma mini-ResNet na CPU."""
import argparse
import csv
import platform
from pathlib import Path

import torch
from torch import nn
from torch.utils.benchmark import Timer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class Bloco(nn.Module):
    def __init__(self, c, residual):
        super().__init__()
        self.residual = residual
        self.ramo = nn.Sequential(
            nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1, bias=False),
        )

    def forward(self, x):
        f = self.ramo(x)
        return x + f if self.residual else f


class Rede(nn.Module):
    def __init__(self, residual, profundidade=8):
        super().__init__()
        self.stem = nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False)
        self.blocos = nn.Sequential(*(Bloco(16, residual) for _ in range(profundidade)))
        self.head = nn.Sequential(nn.BatchNorm2d(16), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))

    def forward(self, x):
        return self.head(self.blocos(self.stem(x)))


def escalar():
    for a, residual in ((.5, False), (.01, False), (.01, True), (-1., True)):
        x = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        z = x
        for _ in range(20):
            # Mesmo ramo F(z) = a * z, com e sem atalho.
            z = z + a * z if residual else a * z
        g, = torch.autograd.grad(z, x)
        print(f"autograd: a={a} residual={residual} | dy/dx={g.item():.12g}")


def perfis(net, x, y):
    net.train()
    net.zero_grad(set_to_none=True)
    ativacoes = [net.stem(x)]
    for bloco in net.blocos:
        ativacoes.append(bloco(ativacoes[-1]))
    loss = nn.functional.cross_entropy(net.head(ativacoes[-1]), y)
    grads = torch.autograd.grad(loss, ativacoes)
    return [g.square().mean().sqrt().item() for g in grads]


def loader(base, idx, batch, shuffle, seed):
    return DataLoader(Subset(base, idx), batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dados", type=Path, default=Path("dados"))
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--treino", type=int, default=6000)
    p.add_argument("--val", type=int, default=2000)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if min(args.epocas, args.treino, args.val, args.threads) < 1 or args.treino + args.val > 60000 or args.val < 32:
        p.error("Use parâmetros positivos, pelo menos 32 exemplos de validação e treino + val <= 60000.")
    torch.set_num_threads(args.threads)
    print(f"PyTorch {torch.__version__} | {platform.machine()} | {args.threads} threads", flush=True)
    escalar()
    base = datasets.FashionMNIST(args.dados, train=True, download=True, transform=transforms.ToTensor())
    idx = torch.randperm(len(base), generator=torch.Generator().manual_seed(2026)).tolist()
    tr, va = idx[:args.treino], idx[-args.val:]
    valid = loader(base, va, 128, False, 0)
    probe_x, probe_y = next(iter(valid))
    probe_x, probe_y = probe_x[:32], probe_y[:32]
    results, curves, grad_rows = [], [], []
    for seed in (7, 42, 123):
        for residual in ([False, True] if seed != 42 else [True, False]):
            nome = "residual" if residual else "plain"
            torch.manual_seed(seed)
            net = Rede(residual)
            # Mede no mesmo lote, com as mesmas ponderações iniciais.
            # Descarta o modelo de diagnóstico para não contaminar as estatísticas.
            for stage, rms in enumerate(perfis(net, probe_x, probe_y)):
                grad_rows.append((seed, nome, stage, rms))
            torch.manual_seed(seed)
            net = Rede(residual)
            opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=0)
            train = loader(base, tr, 64, True, seed + 600)
            for ep in range(1, args.epocas+1):
                net.train()
                tr_loss, tr_count = 0., 0
                for x, y in train:
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(net(x), y)
                    loss.backward()
                    opt.step()
                    tr_loss += loss.item() * len(y)
                    tr_count += len(y)
                net.eval()
                with torch.inference_mode():
                    correct, vl = 0, 0.
                    for x, y in valid:
                        logits = net(x)
                        correct += int((logits.argmax(1) == y).sum())
                        vl += nn.functional.cross_entropy(logits, y, reduction="sum").item()
                acc = correct / len(va)
                curves.append((seed,nome,ep,tr_loss/tr_count,vl/len(va),acc))
                print(seed, nome, ep, "loss_train", round(tr_loss/tr_count,4),
                      "loss_val", round(vl/len(va),4), "acc", round(acc,4), flush=True)
            @torch.inference_mode()
            def forward(x):
                return net(x)
            def tempo(x):
                return Timer("forward(x)", globals={"forward":forward,"x":x},
                             num_threads=args.threads).blocked_autorange(min_run_time=.35).median*1000
            results.append((seed,nome,sum(p.numel() for p in net.parameters()),acc,
                            tempo(probe_x[:1]),tempo(probe_x)))
            if seed == 42:
                torch.save(net.state_dict(), f"{nome}.pt")
    for file, header, rows in (
        ("treino.csv", ["seed","modelo","epoca","loss_treino","loss_val","acc_val"],curves),
        ("gradientes.csv", ["seed","modelo","estagio","grad_rms"],grad_rows),
        ("benchmark.csv", ["seed","modelo","parametros","acc_val","forward_b1_ms","forward_b32_ms"],results),
    ):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    main()

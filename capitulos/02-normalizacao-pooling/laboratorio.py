"""Comparação exploratória CPU: FashionMNIST, sem MLP nem teste oficial."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def rede(batchnorm):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1, bias=True),
        nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
        nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1, bias=True),
        nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
        nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )

def rodar(base, idx_treino, idx_val, nome, bn, epocas, seed):
    torch.manual_seed(seed)
    modelo = rede(bn)
    treino = DataLoader(Subset(base, idx_treino), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val = DataLoader(Subset(base, idx_val), batch_size=128)
    # Sem weight decay para esta comparação curta: aqui Adam e AdamW
    # NÃO são comparados diretamente. O AdamW é a referência adaptativa.
    if nome == "AdamW":
        opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=0.0)
    elif nome == "SGD":
        opt = torch.optim.SGD(modelo.parameters(), lr=0.05, momentum=0.9)
    else:
        opt = torch.optim.RMSprop(modelo.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()
    for epoca in range(1, epocas + 1):
        modelo.train()
        for x, y in treino:
            opt.zero_grad(set_to_none=True)
            perda = loss(modelo(x), y)
            perda.backward()
            opt.step()
        modelo.eval()
        total = acertos = soma = 0
        with torch.inference_mode():
            for x, y in val:
                logits = modelo(x)
                soma += float(loss(logits, y)) * len(y)
                acertos += int((logits.argmax(1) == y).sum())
                total += len(y)
        print(f"{nome:7} BN={bn:5} epoca={epoca} val_loss={soma/total:.4f} val_acc={acertos/total:.2%}", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dados", type=Path, default=Path("dados"))
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epocas < 1:
        p.error("--epocas precisa ser positiva")
    torch.set_num_threads(2)
    base = datasets.FashionMNIST(args.dados, train=True, download=True,
                                transform=transforms.ToTensor())
    ordem = torch.randperm(len(base), generator=torch.Generator().manual_seed(args.seed)).tolist()
    print(f"Torch: {torch.__version__}. CPU. 12000 treino / 3000 val. Sem teste oficial.", flush=True)
    for nome, bn in [("AdamW", True), ("AdamW", False), ("SGD", True), ("RMSprop", True)]:
        rodar(base, ordem[:12_000], ordem[12_000:15_000], nome, bn, args.epocas, args.seed)

if __name__ == "__main__":
    main()

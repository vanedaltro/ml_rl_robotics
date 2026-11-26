"""MiniViT: classificação de relações entre formas. Só requer PyTorch."""
import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MiniViT(nn.Module):
    def __init__(
        self, size=32, patch=8, channels=1, dim=48, heads=4,
        depth=2, classes=2,
    ):
        super().__init__()
        if size % patch != 0 or dim % heads != 0:
            raise ValueError("size deve ser divisível por patch; dim, por heads.")
        self.size, self.channels = size, channels
        n = (size // patch) ** 2
        self.project = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=0.1, activation="gelu", batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def patch_tokens(self, x):
        return self.project(x).flatten(2).transpose(1, 2)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (self.channels, self.size, self.size):
            raise ValueError("Formato esperado: [B, canais, size, size].")
        tokens = self.patch_tokens(x)                           # [B, 16, 48]
        cls = self.cls.expand(x.size(0), -1, -1)                # [B, 1, 48]
        z = torch.cat((cls, tokens), dim=1) + self.pos            # [B, 17, 48]
        for block in self.blocks:
            z = block(z)                                        # sem máscara causal
        return self.head(self.norm(z[:, 0]))                      # [B, 2]


def make_data(n, seed):
    """Classe 0: barras com orientação igual. Classe 1: diferente."""
    g = torch.Generator().manual_seed(seed)
    x = 0.12 * torch.rand(n, 1, 32, 32, generator=g)
    y = torch.arange(n) % 2
    y = y[torch.randperm(n, generator=g)]
    orientation = torch.randint(2, (n,), generator=g)
    rows = torch.randint(4, (n, 2), generator=g) * 8
    cols = (torch.randint(2, (n, 2), generator=g) + torch.tensor([0, 2])) * 8
    shifts = torch.randint(-1, 2, (n, 2), generator=g)
    for i in range(n):
        for side, ori in enumerate((int(orientation[i]), int(orientation[i] ^ y[i]))):
            r, c = int(rows[i, side]), int(cols[i, side])
            d = int(shifts[i, side])
            if ori == 0:
                x[i, 0, r+3+d:r+5+d, c+1:c+7] += 0.85
            else:
                x[i, 0, r+1:r+7, c+3+d:c+5+d] += 0.85
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        correct += int((model(x).argmax(dim=1) == y).sum())
        total += y.numel()
    return correct / total


@torch.no_grad()
def check_shapes(model, sample):
    model.eval()
    tokens = model.patch_tokens(sample)
    patches = nn.functional.unfold(sample, kernel_size=8, stride=8).transpose(1, 2)
    explicit = nn.functional.linear(patches, model.project.weight.flatten(1),
                                    model.project.bias)
    torch.testing.assert_close(tokens, explicit)
    logits = model(sample)
    assert logits.shape == (sample.size(0), 2)
    assert tokens.shape[1:] == (16, 48)
    print(f"patches={tuple(tokens.shape)} logits={tuple(logits.shape)} Conv2d=Unfold+Linear OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train = make_data(1536, 101)
    test = make_data(512, 202)
    loader = DataLoader(train, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    test_loader = DataLoader(test, batch_size=64)
    model = MiniViT()
    print(f"PyTorch {torch.__version__} | CPU | {sum(p.numel() for p in model.parameters()):,} parâmetros")
    check_shapes(model, train.tensors[0][:4])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    t = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * y.numel()
        print(f"época {epoch+1:02d}: loss treino={total/len(train):.4f}", flush=True)
    print(f"acurácia teste={evaluate(model, test_loader):.2%} | {time.perf_counter()-t:.1f}s")


if __name__ == "__main__":
    main()

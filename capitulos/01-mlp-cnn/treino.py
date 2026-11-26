"""Treine MLP e CNN no FashionMNIST com o mesmo protocolo."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH = 128

def construir(nome):
    if nome == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    if nome == "cnn":
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )
    raise ValueError(nome)

def treinar_epoca(modelo, dados, loss_fn, opt, device):
    modelo.train()
    for x, y in dados:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        perda = loss_fn(modelo(x), y)
        perda.backward()
        opt.step()

@torch.inference_mode()
def avaliar(modelo, dados, loss_fn, device):
    modelo.eval()
    perda_total, acertos, total = 0.0, 0, 0
    for x, y in dados:
        x, y = x.to(device), y.to(device)
        logits = modelo(x)
        perda_total += loss_fn(logits, y).item() * y.size(0)
        acertos += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return perda_total / total, acertos / total

def escolher_dispositivo(nome):
    if nome != "auto":
        return torch.device(nome)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=["mlp", "cnn", "ambos"], default="ambos")
    parser.add_argument("--epocas", type=int, default=4)
    parser.add_argument("--dispositivo", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dados", type=Path, default=Path("dados"))
    parser.add_argument("--saida", type=Path, default=Path("modelos"))
    args = parser.parse_args()
    if args.epocas < 1:
        parser.error("--epocas precisa ser >= 1")

    device = escolher_dispositivo(args.dispositivo)
    transform = transforms.ToTensor()
    base = datasets.FashionMNIST(args.dados, train=True, download=True, transform=transform)
    teste = datasets.FashionMNIST(args.dados, train=False, download=True, transform=transform)
    treino, val = random_split(
        base, [55_000, 5_000], generator=torch.Generator().manual_seed(SEED)
    )
    val_dl = DataLoader(val, batch_size=BATCH)
    teste_dl = DataLoader(teste, batch_size=BATCH)
    loss_fn = nn.CrossEntropyLoss()
    args.saida.mkdir(exist_ok=True, parents=True)

    nomes = ["mlp", "cnn"] if args.modelo == "ambos" else [args.modelo]
    print(f"PyTorch {torch.__version__} | {device} | treino/val/teste: {len(treino)}/{len(val)}/{len(teste)}", flush=True)
    for nome in nomes:
        torch.manual_seed(SEED)
        treino_dl = DataLoader(treino, batch_size=BATCH, shuffle=True,
                               generator=torch.Generator().manual_seed(SEED))
        modelo = construir(nome).to(device)
        opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
        melhor_loss = float("inf")
        caminho = args.saida / f"{nome}.pt"
        n_params = sum(p.numel() for p in modelo.parameters())
        print(f"{nome}: {n_params:,} parâmetros", flush=True)
        for epoca in range(1, args.epocas + 1):
            treinar_epoca(modelo, treino_dl, loss_fn, opt, device)
            loss_val, acc_val = avaliar(modelo, val_dl, loss_fn, device)
            print(f"{nome} epoca={epoca} val_loss={loss_val:.4f} val_acc={acc_val:.2%}", flush=True)
            if loss_val < melhor_loss:
                melhor_loss = loss_val
                torch.save(modelo.state_dict(), caminho)

        modelo.load_state_dict(torch.load(caminho, map_location=device, weights_only=True))
        loss_t, acc_t = avaliar(modelo, teste_dl, loss_fn, device)
        print(f"{nome} teste_loss={loss_t:.4f} teste_acc={acc_t:.2%}", flush=True)

        modelo.eval()
        with torch.inference_mode():
            imagem, rotulo = teste[0]
            classe = modelo(imagem.unsqueeze(0).to(device)).argmax(dim=1).item()
        print(f"Exemplo: {teste.classes[classe]} (real: {teste.classes[rotulo]})", flush=True)

if __name__ == "__main__":
    main()

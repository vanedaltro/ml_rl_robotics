"""AE versus VAE 2D em FashionMNIST. As figuras são produzidas pelo script figuras.py."""
import argparse
import csv
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

class Codec(nn.Module):
    def __init__(self, variational=False, latent=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU())
        self.mu = nn.Linear(128, latent)
        self.logvar = nn.Linear(128, latent) if variational else None
        self.decoder = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 256),
                                     nn.ReLU(), nn.Linear(256, 784))
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), (self.logvar(h) if self.logvar is not None else None)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu if logvar is None else mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

def objective(logits, x, mu, logvar, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(logits, x.flatten(1), reduction='none').sum(1)
    kl = (torch.zeros_like(recon) if logvar is None else
          -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1))
    return (recon + beta * kl).mean()

def main(a):
    torch.set_num_threads(2)
    ds = FashionMNIST(a.data, train=True, download=a.download)
    test = FashionMNIST(a.data, train=False, download=a.download)
    ix = torch.randperm(len(ds), generator=torch.Generator().manual_seed(2026))[:a.n]
    train_x = ds.data[ix].float() / 255
    x = test.data.float() / 255
    reps, latents, stats, random_out, grids = [], [], [], [], []
    out = Path(a.out)
    out.mkdir(exist_ok=True, parents=True)
    samples = torch.randn(32, 2, generator=torch.Generator().manual_seed(471))
    axis = torch.linspace(-2.2, 2.2, 9)
    grid = torch.cartesian_prod(-axis, axis)[:, [1, 0]]
    shown = torch.stack([(test.targets == k).nonzero().flatten()[0] for k in range(10)])
    for v in (False, True):
        torch.manual_seed(42)
        mod = Codec(v)
        opt = torch.optim.Adam(mod.parameters(), lr=0.001)
        batches = DataLoader(TensorDataset(train_x), batch_size=128, shuffle=True,
                             generator=torch.Generator().manual_seed(43))
        for epoch in range(a.epochs):
            mod.train()
            for (xb,) in batches:
                opt.zero_grad(set_to_none=True)
                loss = objective(*_objective_order(mod(xb), xb))
                loss.backward()
                opt.step()
            print('VAE' if v else 'AE', epoch+1, round(loss.item(), 2), flush=True)
        mod.eval()
        with torch.inference_mode():
            mu = torch.cat([mod.encode(b)[0] for b in x.split(256)])
            logits = torch.cat([mod.decoder(b) for b in mu.split(256)])
            bce = F.binary_cross_entropy_with_logits(logits,x.flatten(1)).item()
            mse = (logits.sigmoid() - x.flatten(1)).square().mean().item()
            stats.append(dict(model='VAE' if v else 'AE', params=sum(p.numel() for p in mod.parameters()),
                              mse_pixel=mse, bce_pixel=bce, latent_mean=mu.mean(0).tolist(),
                              latent_std=mu.std(0).tolist()))
            reps.append(logits[shown].sigmoid().reshape(-1,28,28))
            latents.append(mu)
            random_out.append(mod.decoder(samples).sigmoid().reshape(-1,28,28))
            grids.append(mod.decoder(grid).sigmoid().reshape(-1,28,28))
        torch.save(mod.state_dict(),out/('vae.pt' if v else 'ae.pt'))
    torch.save(dict(original=x[shown], recon=reps, latents=latents, labels=test.targets,
                    random=random_out, grid=grids),out/'visual.pt')
    with (out/'resultados.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=stats[0]);w.writeheader();w.writerows(stats)
    print(stats)

def _objective_order(encoded, x):
    logits, mu, logvar = encoded
    return logits, x, mu, logvar

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='./dados')
    p.add_argument('--download',action='store_true')
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--n',type=int,default=24000)
    p.add_argument('--out',default='.')
    main(p.parse_args())

"""Um MLP contrativo permite observar o diagnóstico em pesos reais."""
import csv
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, skip, d=16, profundidade=20):
        super().__init__()
        self.skip = skip
        self.camadas = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(profundidade))
        for camada in self.camadas:
            nn.init.orthogonal_(camada.weight, gain=0.2)

    def forward(self, x):
        for camada in self.camadas:
            f = camada(x).tanh()
            x = x + f if self.skip else f
        return x


def main():
    torch.set_num_threads(2)
    gerador = torch.Generator().manual_seed(2026)
    x = torch.randn(512, 16, generator=gerador)
    # Tarefa sintética perto da identidade, escolhida para ilustrar o viés do atalho.
    y = x + 0.2 * x.roll(shifts=1, dims=1).tanh()
    rows = []
    for seed in (7, 42, 123):
        for skip in (False, True):
            torch.manual_seed(seed)
            net = MLP(skip)
            opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0)
            for step in range(151):
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(net(x[:384]), y[:384])
                loss.backward()
                if step in (0, 150):
                    with torch.no_grad():
                        valid = nn.functional.mse_loss(net(x[384:]), y[384:]).item()
                    rows.append([seed, 'residual' if skip else 'plain', step,
                                 loss.item(), valid,
                                 net.camadas[0].weight.grad.norm().item(),
                                 net.camadas[-1].weight.grad.norm().item()])
                if step < 150:
                    opt.step()
    with open('mlp_gradiente.csv','w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed','modelo','passo','mse_treino','mse_val','norma_grad_peso_1','norma_grad_peso_20'])
        w.writerows(rows)


if __name__ == '__main__':
    main()

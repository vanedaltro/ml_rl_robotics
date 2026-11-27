"""REINFORCE no GridWorld do episódio 1."""
import argparse
from statistics import mean

import torch
from torch.distributions import Categorical, Normal

from grid import COLUNAS, INICIO, LINHAS, MOVIMENTOS, step

ACOES = tuple(MOVIMENTOS)
GAMMA = 0.9
LIMITE = 25


def retornos(recompensas, gamma):
    g, saida = 0.0, []
    for r in reversed(recompensas):
        g = r + gamma * g
        saida.append(g)
    return list(reversed(saida))


def indice(estado):
    return estado[0] * COLUNAS + estado[1]


def coleta(logits, guloso=False):
    estados, logps, recompensas = [], [], []
    estado = INICIO
    for _ in range(LIMITE):
        i = indice(estado)
        pi = Categorical(logits=logits[i])
        acao = logits[i].argmax() if guloso else pi.sample()
        proximo, r, terminou = step(estado, ACOES[int(acao)])
        estados.append(i)
        logps.append(pi.log_prob(acao))
        recompensas.append(r)
        estado = proximo
        if terminou:
            break
    return estados, torch.stack(logps), recompensas


def perda(logps, retorno, baseline, gamma):
    # Objetivo: retorno descontado a partir do começo, daí o gamma**t.
    pesos = logps.new_tensor([gamma ** t for t in range(logps.numel())])
    return -(pesos * logps * (retorno - baseline).detach()).sum()


def train(episodios, seed, usar_baseline):
    torch.manual_seed(seed)
    logits = torch.zeros(LINHAS * COLUNAS, len(ACOES), requires_grad=True)
    counts, medias = [0] * len(logits), [0.0] * len(logits)
    opt = torch.optim.Adam([logits], lr=0.04)
    historico = []
    for _ in range(episodios):
        estados, logps, recompensas = coleta(logits)
        g = retornos(recompensas, GAMMA)
        b = [medias[i] if usar_baseline else 0.0 for i in estados]
        opt.zero_grad()
        perda(logps, logps.new_tensor(g), logps.new_tensor(b), GAMMA).backward()
        opt.step()
        # Dados do episódio atual só entram na baseline após o update.
        for i, valor in zip(estados, g):
            counts[i] += 1
            medias[i] += (valor - medias[i]) / counts[i]
        historico.append(g[0])
    return logits.detach(), historico


def exemplo_gaussiano():
    mu = torch.tensor(0.0, requires_grad=True)
    pi = Normal(mu, torch.tensor(1.0))
    acao = torch.tensor(0.75)
    # A amostra é tratada como constante no estimador score-function.
    grad, = torch.autograd.grad(pi.log_prob(acao), mu)
    return grad.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodios', type=int, default=400)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.episodios < 1:
        parser.error('use ao menos um episódio')
    torch.set_num_threads(1)
    for baseline in (False, True):
        logits, hist = train(args.episodios, args.seed, baseline)
        with torch.no_grad():
            _, _, greedy = coleta(logits, guloso=True)
        prob = logits[indice(INICIO)].softmax(0)[ACOES.index('D')].item()
        print(f'baseline={baseline}: G início={mean(hist[:min(50, len(hist))]):+.3f}, '
              f'G fim={mean(hist[-50:]):+.3f}, P(D|início)={prob:.3f}, '
              f'rota gulosa G={retornos(greedy, GAMMA)[0]:+.3f}')
    print(f'Gaussiana: d log π(a=0,75) / d μ = {exemplo_gaussiano():+.2f}')


if __name__ == '__main__':
    main()

"""Ator-crítico tabular com TD(0) e uma demonstração de duas cabeças."""
import argparse
import math
from random import Random
from statistics import mean

from grid_ac import ACOES, BANDEIRA, COLUNAS, INICIO, LINHAS, indice, step

GAMMA = .9
LIMITE = 60


def td(recompensa, valor, valor_proximo, terminou, gamma=GAMMA):
    alvo = recompensa if terminou else recompensa + gamma * valor_proximo
    return alvo, alvo - valor


def softmax(x):
    m = max(x)
    exp = [math.exp(y - m) for y in x]
    z = sum(exp)
    return [y / z for y in exp]


def train(episodios, seed):
    rng = Random(seed)
    logits = [[0.0] * len(ACOES) for _ in range(LINHAS * COLUNAS)]
    valor = [0.0] * len(logits)
    retornos = []
    for _ in range(episodios):
        estado, retorno = INICIO, 0.0
        for t in range(LIMITE):
            i = indice(estado)
            p = softmax(logits[i])
            a = rng.choices(range(len(ACOES)), p)[0]
            proximo, r, terminou = step(estado, a)
            _, delta = td(r, valor[i], valor[indice(proximo)], terminou)
            valor[i] += .15 * delta
            for j, pj in enumerate(p):
                logits[i][j] += .15 * GAMMA ** t * delta * ((a == j) - pj)
            retorno += GAMMA ** t * r
            estado = proximo
            if terminou:
                break
        retornos.append(retorno)
    return logits, valor, retornos


def rota_gulosa(logits):
    estado, rota = INICIO, []
    for _ in range(LIMITE):
        x = logits[indice(estado)]
        acao = x.index(max(x))
        estado, _, terminou = step(estado, acao)
        rota.append(ACOES[acao])
        if terminou:
            break
    return ''.join(rota), estado == BANDEIRA


def perda_neural(logits, valores, acoes, alvos, beta=.01):
    from torch.distributions import Categorical

    pi = Categorical(logits=logits)
    delta = alvos.detach() - valores
    ator = -(pi.log_prob(acoes) * delta.detach()).mean()
    critico = delta.square().mean()
    return ator + .5 * critico - beta * pi.entropy().mean()


def neural_demo():
    import torch
    from torch import nn

    torch.manual_seed(7)
    corpo = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
    ator, critico = nn.Linear(16, 4), nn.Linear(16, 1)
    h = corpo(torch.tensor([[.5, .5], [1., 0.]]))
    loss = perda_neural(ator(h), critico(h).squeeze(-1),
                        torch.tensor([1, 0]), torch.tensor([1., -.2]))
    loss.backward()
    return [any(p.grad is not None and p.grad.abs().sum() > 0
                for p in trecho.parameters()) for trecho in (corpo, ator, critico)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodios', type=int, default=300)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()
    if args.episodios < 1:
        p.error('use ao menos um episódio')
    logits, _, g = train(args.episodios, args.seed)
    for vp, r in [(5, 1), (3, 0)]:
        alvo, delta = td(r, 4, vp, False)
        print(f'alvo={alvo:.2f}, delta={delta:+.2f}')
    rota, bandeira = rota_gulosa(logits)
    print(f'G início={mean(g[:30]):+.3f} / G fim={mean(g[-30:]):+.3f}')
    print(f'rota inicial da política final={rota[:12]}, bandeira={bandeira}')
    print('Gradientes (corpo, ator, crítico):', neural_demo())


if __name__ == '__main__':
    main()

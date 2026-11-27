"""Cliff Walking: comparação de Q-learning e SARSA sem dependências."""
import argparse
from collections import defaultdict
from random import Random
from statistics import mean

ALTURA, LARGURA = 4, 12
INICIO, META = (3, 0), (3, 11)
MOVIMENTOS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SETAS = "^>v<"


def step(estado, acao):
    """Devolve (novo, recompensa, terminou, caiu). Cair reinicia só a posição."""
    if estado == META:
        raise ValueError("Comece outro episódio antes de agir na meta.")
    dl, dc = MOVIMENTOS[acao]
    l = min(ALTURA - 1, max(0, estado[0] + dl))
    c = min(LARGURA - 1, max(0, estado[1] + dc))
    if l == ALTURA - 1 and 0 < c < LARGURA - 1:
        return INICIO, -100, False, True
    novo = (l, c)
    return novo, -1, novo == META, False


def gulosa(valores, rng):
    maior = max(valores)
    return rng.choice([i for i, valor in enumerate(valores) if valor == maior])


def escolher(valores, epsilon, rng):
    if epsilon > 0 and rng.random() < epsilon:
        return rng.randrange(len(valores))
    return gulosa(valores, rng)


def atualizar(valor, recompensa, futuro, terminou, alpha, gamma):
    """Uma transição do MDP: o alvo zera o bootstrap só na meta."""
    alvo = recompensa + (0.0 if terminou else gamma * futuro)
    return valor + alpha * (alvo - valor)


def treinar(algoritmo, episodios=500, seed=42, epsilon=0.1, alpha=0.5,
            gamma=1.0, max_passos=500):
    if algoritmo not in ("q-learning", "sarsa"):
        raise ValueError(algoritmo)
    rng = Random(seed)
    q = defaultdict(lambda: [0.0] * len(MOVIMENTOS))
    retornos, quedas = [], []
    for _ in range(episodios):
        estado = INICIO
        acao = escolher(q[estado], epsilon, rng)
        retorno = caiu_total = 0
        for _ in range(max_passos):
            novo, recompensa, terminou, caiu = step(estado, acao)
            retorno += recompensa
            caiu_total += caiu
            if terminou:
                futuro = 0.0
                proxima = None
            else:
                # SARSA usa a ação sorteada aqui, inclusive quando foi exploratória.
                proxima = escolher(q[novo], epsilon, rng)
                futuro = max(q[novo]) if algoritmo == "q-learning" else q[novo][proxima]
            q[estado][acao] = atualizar(q[estado][acao], recompensa, futuro,
                                        terminou, alpha, gamma)
            if terminou:
                break
            estado, acao = novo, proxima
        # O limite interrompe a coleta, não vira estado terminal na equação.
        retornos.append(retorno)
        quedas.append(caiu_total)
    return q, retornos, quedas


def avaliar(q, seed=123, max_passos=100):
    rng = Random(seed)
    estado, caminho = INICIO, [INICIO]
    for _ in range(max_passos):
        acao = gulosa(q.get(estado, [0.0] * len(MOVIMENTOS)), rng)
        estado, _, terminou, _ = step(estado, acao)
        caminho.append(estado)
        if terminou:
            return caminho, True
    return caminho, False


def mapa(caminho):
    linhas = [["."] * LARGURA for _ in range(ALTURA)]
    linhas[-1] = list("S" + "#" * (LARGURA - 2) + "G")
    deltas = dict(zip(MOVIMENTOS, SETAS))
    for (l, c), (nl, nc) in zip(caminho, caminho[1:]):
        if (l, c) != INICIO:
            linhas[l][c] = deltas.get((nl - l, nc - c), "+")
    return "\n".join("".join(linha) for linha in linhas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodios", type=int, default=500)
    parser.add_argument("--repeticoes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodios < 1 or args.repeticoes < 1:
        parser.error("--episodios e --repeticoes devem ser positivos")
    janela = min(args.episodios, 100)
    for nome in ("q-learning", "sarsa"):
        medias, quedas_por_ep = [], []
        for seed in range(args.seed, args.seed + args.repeticoes):
            q, retornos, quedas = treinar(nome, args.episodios, seed)
            medias.append(mean(retornos[-janela:]))
            quedas_por_ep.append(mean(quedas[-janela:]))
            if seed == args.seed:
                caminho, meta = avaliar(q)
        print(f"{nome}: retorno/episódio {mean(medias):.2f}; "
              f"quedas/episódio {mean(quedas_por_ep):.3f} (últimos {janela})")
        print(f"Avaliação gulosa, seed {args.seed}: meta={meta}, passos={len(caminho) - 1}")
        print(mapa(caminho))


if __name__ == "__main__":
    main()

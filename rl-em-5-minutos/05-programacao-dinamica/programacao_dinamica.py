"""Coin Change, Value Iteration e Policy Iteration com Python puro."""
from functools import lru_cache
from math import inf

LINHAS, COLUNAS = 3, 5
INICIO, BANDEIRA, PAREDE, BURACO = (2, 0), (0, 4), (1, 2), (1, 0)
TERMINAIS = frozenset((BANDEIRA, BURACO))
ACOES = {"C": (-1, 0), "D": (0, 1), "B": (1, 0), "E": (0, -1)}
ESTADOS = tuple((l, c) for l in range(LINHAS) for c in range(COLUNAS)
                if (l, c) not in TERMINAIS and (l, c) != PAREDE)
GAMMA = 0.9


def moedas_minimas(alvo, moedas=(1, 3, 4)):
    """Tabela bottom-up: infinito marca um valor inalcançável."""
    if alvo < 0 or not moedas or any(c <= 0 for c in moedas):
        raise ValueError("Alvo não negativo e denominações positivas são necessários.")
    dp = [0] + [inf] * alvo
    for i in range(1, alvo + 1):
        dp[i] = min((1 + dp[i-c] for c in moedas if c <= i), default=inf)
    return dp


def moedas_memoizadas(alvo, moedas=(1, 3, 4)):
    """Mesma recorrência top-down com cache; use para alvos pequenos."""
    if alvo < 0 or not moedas or any(c <= 0 for c in moedas):
        raise ValueError("Alvo não negativo e denominações positivas são necessários.")

    @lru_cache(None)
    def f(i):
        return 0 if not i else min((1 + f(i-c) for c in moedas if c <= i), default=inf)

    return f(alvo)


def modelo(estado, acao):
    """Tabuleiro conhecido do episódio 1, usado agora no planejamento."""
    dl, dc = ACOES[acao]
    candidato = (estado[0] + dl, estado[1] + dc)
    if candidato not in ESTADOS and candidato not in TERMINAIS:
        candidato = estado
    r = 10 if candidato == BANDEIRA else -10 if candidato == BURACO else -1
    return ((1.0, candidato, r),)


def backup(estado, acao, valores, gamma):
    return sum(p * (r + gamma * (0 if destino in TERMINAIS else valores[destino]))
               for p, destino, r in modelo(estado, acao))


def escolha_gulosa(s, v, gamma):
    return max(ACOES, key=lambda a: backup(s, a, v, gamma))


def _iterar(regra, gamma, tolerancia, max_iter):
    if not 0 <= gamma < 1 or tolerancia <= 0 or max_iter < 1:
        raise ValueError("Parâmetros inválidos de convergência.")
    v = dict.fromkeys(ESTADOS, 0.0)
    for k in range(1, max_iter + 1):
        novo = {s: regra(s, v) for s in ESTADOS}
        if max(abs(novo[s] - v[s]) for s in ESTADOS) < tolerancia:
            return novo, k
        v = novo
    raise RuntimeError("Limite de varreduras atingido.")


def value_iteration(gamma=GAMMA, tolerancia=1e-10, max_iter=10_000):
    v, k = _iterar(lambda s, v: max(backup(s, a, v, gamma) for a in ACOES),
                   gamma, tolerancia, max_iter)
    return v, {s: escolha_gulosa(s, v, gamma) for s in ESTADOS}, k


def policy_iteration(gamma=GAMMA, tolerancia=1e-10, max_politicas=100):
    politica = dict.fromkeys(ESTADOS, "D")
    for k in range(1, max_politicas + 1):
        v, _ = _iterar(lambda s, v: backup(s, politica[s], v, gamma),
                       gamma, tolerancia, 10_000)
        nova = {}
        for s in ESTADOS:
            melhor = escolha_gulosa(s, v, gamma)
            # Em empates, conserve a ação para evitar alternância numérica.
            nova[s] = (melhor if backup(s, melhor, v, gamma) >
                       backup(s, politica[s], v, gamma) + tolerancia else politica[s])
        if nova == politica:
            return v, politica, k
        politica = nova
    raise RuntimeError("Limite de melhorias atingido.")


def roteiro(politica, limite=30):
    estado, passos = INICIO, []
    for _ in range(limite):
        acao = politica[estado]
        _, estado, recompensa = modelo(estado, acao)[0]
        passos.append((acao, estado, recompensa))
        if estado in TERMINAIS:
            return passos
    raise RuntimeError("A política não alcançou um terminal no limite.")


def main():
    print("Coin Change, moedas 1, 3, 4:", moedas_minimas(6))
    print("Mínimo para 6, com memoização:", moedas_memoizadas(6))
    v, pi, k = value_iteration()
    vp, _, kp = policy_iteration()
    print(f"Value Iteration: V(início)={v[INICIO]:.4f}, {k} varreduras.")
    print(f"Policy Iteration: V(início)={vp[INICIO]:.4f}, {kp} políticas avaliadas.")
    marcas = {BANDEIRA: "+", BURACO: "X", PAREDE: "#"}
    for l in range(LINHAS):
        print("  ", " ".join(pi.get((l, c), marcas.get((l, c), "?")) for c in range(COLUNAS)))
    passos = roteiro(pi)
    print("Rota ótima:", "".join(a for a, _, _ in passos), "recompensas:", [r for _, _, r in passos])


if __name__ == "__main__":
    main()

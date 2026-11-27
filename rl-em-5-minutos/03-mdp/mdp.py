"""MDP de um robô em piso escorregadio: Python sem dependências externas."""

from collections import Counter, defaultdict
from math import isclose
from random import Random

LINHAS, COLUNAS = 3, 5
INICIO, BANDEIRA = (2, 0), (0, 4)
PAREDE, BURACO = (1, 2), (1, 0)
ACOES = ("C", "D", "B", "E")  # sentido horário
MOVIMENTOS = dict(zip(ACOES, ((-1, 0), (0, 1), (1, 0), (0, -1))))
ESTADOS = tuple(
    (l, c) for l in range(LINHAS) for c in range(COLUNAS) if (l, c) != PAREDE
)
TERMINAIS = frozenset((BANDEIRA, BURACO))
GAMMA = 0.9


def destino(estado, movimento):
    dl, dc = MOVIMENTOS[movimento]
    candidato = estado[0] + dl, estado[1] + dc
    return candidato if candidato in ESTADOS else estado


def modelo(estado, acao, aderencia=0.8):
    """P(s',r|s,a) como tuplas (probabilidade, destino, recompensa, terminou)."""
    if estado not in ESTADOS or estado in TERMINAIS:
        raise ValueError("Escolha um estado válido não terminal.")
    if not 0 <= aderencia <= 1:
        raise ValueError("Aderência fora de [0, 1].")
    indice = ACOES.index(acao)
    lateral = (1 - aderencia) / 2
    probabilidades = defaultdict(float)
    for movimento, p in ((acao, aderencia), (ACOES[(indice-1) % 4], lateral),
                          (ACOES[(indice+1) % 4], lateral)):
        if p:
            probabilidades[destino(estado, movimento)] += p

    return tuple(
        (p, novo, 10 if novo == BANDEIRA else -10 if novo == BURACO else -1,
         novo in TERMINAIS)
        for novo, p in probabilidades.items()
    )


def passo(estado, acao, rng, aderencia=0.8):
    """Amostra do modelo; o agente escolhe a ação, não o resultado."""
    resultados = modelo(estado, acao, aderencia)
    sorteio = rng.random()
    acumulada = 0.0
    for p, *resultado in resultados:
        acumulada += p
        if sorteio < acumulada:
            return tuple(resultado)
    assert isclose(acumulada, 1.0)  # arredondamento na última fronteira
    return tuple(resultados[-1][1:])


def episodio(acoes, rng, aderencia=0.8):
    estado, trilha = INICIO, []
    for acao in acoes:
        novo, recompensa, terminou = passo(estado, acao, rng, aderencia)
        trilha.append((estado, acao, recompensa, novo, terminou))
        estado = novo
        if terminou:
            break
    return trilha


def retorno(recompensas, gamma=GAMMA):
    if not 0 <= gamma <= 1:
        raise ValueError("Gamma fora de [0, 1].")
    g = 0.0
    for r in reversed(tuple(recompensas)):
        g = r + gamma * g
    return g


def main():
    # Posição central sem paredes: os três movimentos terminam em casas distintas.
    e = (1, 3)
    print("Modelo exato para direita em", e)
    for p, novo, r, terminou in modelo(e, "D"):
        print(f"  P={p:.2f} -> {novo}, r={r:+d}, terminal={terminou}")
    rng = Random(42)
    n = 10_000
    contagem = Counter(passo(e, "D", rng)[0] for _ in range(n))
    print(f"Frequências em {n} amostras:", {s: round(v/n, 4) for s, v in sorted(contagem.items())})

    mesma_rota = episodio("DDDDCC", Random(42), aderencia=1)
    rs = [item[2] for item in mesma_rota]
    print(f"Sem gelo: {rs}; G(0.9)={retorno(rs):.4f}; G(1)={retorno(rs, 1):.1f}")
    for gamma in (0.5, 0.9):
        moeda = retorno([2], gamma)
        tesouro = retorno([0] * 5 + [10], gamma)
        print(f"γ={gamma}: moeda={moeda:.4f}, tesouro={tesouro:.4f}")


if __name__ == "__main__":
    main()

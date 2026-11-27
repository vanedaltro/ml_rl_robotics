"""Avaliação de políticas e cálculo de V e Q em um MDP tabular."""

from math import isclose
from random import Random
from statistics import mean

GAMMA = 0.9
INICIO, CORREDOR, BANDEIRA, BURACO = "início", "corredor", "bandeira", "buraco"
ESTADOS = (INICIO, CORREDOR, BANDEIRA, BURACO)
TERMINAIS = (BANDEIRA, BURACO)
# Cada ramo contém (probabilidade, próximo estado, recompensa deste passo).
MODELO = {
    INICIO: {
        "direita": ((0.8, CORREDOR, -1), (0.2, BURACO, -1)),
        "esperar": ((1.0, INICIO, -1),),
    },
    CORREDOR: {
        "avançar": ((1.0, BANDEIRA, 10),),
        "voltar": ((1.0, INICIO, -1),),
    },
}
DIRETA = {INICIO: {"direita": 1.0}, CORREDOR: {"avançar": 1.0}}
MISTA = {INICIO: {"direita": 0.7, "esperar": 0.3}, CORREDOR: {"avançar": 1.0}}


def valor_acao(estado, acao, v, gamma=GAMMA):
    """Qπ(s,a) por expectativa de r + γVπ(s')."""
    return sum(p * (r + gamma * (0.0 if novo in TERMINAIS else v[novo]))
               for p, novo, r in MODELO[estado][acao])


def avaliar(politica, gamma=GAMMA, tolerancia=1e-12, max_iter=1000):
    """Avaliação síncrona: V novo é calculado só a partir de V anterior."""
    if not 0 <= gamma < 1 or tolerancia <= 0 or max_iter < 1:
        raise ValueError("Use 0 ≤ γ < 1, tolerância positiva e pelo menos uma iteração.")
    if set(politica) != set(MODELO):
        raise ValueError("A política precisa cobrir todos os estados não terminais.")
    for estado, probs in politica.items():
        if (not set(probs) <= set(MODELO[estado]) or
                any(not 0 <= p <= 1 for p in probs.values()) or
                not isclose(sum(probs.values()), 1)):
            raise ValueError("Política inválida.")
    v = dict.fromkeys(ESTADOS, 0.0)
    for _ in range(max_iter):
        novo = dict.fromkeys(ESTADOS, 0.0)
        for s, probs in politica.items():
            novo[s] = sum(p * valor_acao(s, a, v, gamma) for a, p in probs.items())
        if max(abs(novo[s] - v[s]) for s in ESTADOS) < tolerancia:
            return novo
        v = novo
    raise RuntimeError("A avaliação não convergiu no limite de iterações.")


def amostra_retorno(politica, rng, gamma=GAMMA, max_passos=1000,
                    acao_inicial=None):
    """Retorno amostrado; opcionalmente força a primeira ação para estimar Q."""
    estado, desconto, g = INICIO, 1.0, 0.0
    for t in range(max_passos):
        probs = politica[estado]
        acao = (acao_inicial if t == 0 and acao_inicial is not None else
                rng.choices(list(probs), weights=list(probs.values()))[0])
        ramos = MODELO[estado][acao]
        _, estado, recompensa = rng.choices(ramos, weights=[p for p, *_ in ramos])[0]
        g += desconto * recompensa
        desconto *= gamma
        if estado in TERMINAIS:
            return g
    raise RuntimeError("Episódio truncado, o retorno ainda não está completo.")


def main():
    for nome, politica in (("direta", DIRETA), ("mista", MISTA)):
        v = avaliar(politica)
        q = {a: valor_acao(INICIO, a, v) for a in MODELO[INICIO]}
        rng = Random(42)
        mc = mean(amostra_retorno(politica, rng) for _ in range(10_000))
        print(f"Política {nome}: V(início)={v[INICIO]:.4f}; V(corredor)={v[CORREDOR]:.4f}")
        print(f"  Q(direita)={q['direita']:.4f}; Q(esperar)={q['esperar']:.4f}")
        print(f"  Média de 10000 retornos a partir do início: {mc:.4f}")


if __name__ == "__main__":
    main()

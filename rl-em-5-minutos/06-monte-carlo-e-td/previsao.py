"""Predição de valor com Monte Carlo e TD(0), sem o modelo nos agentes."""
from math import sqrt
from random import Random


def caminhada(rng, n=5):
    """Política fixa: cada passo vai para esquerda ou direita com 50%."""
    estado = (n + 1) // 2
    while 0 < estado < n + 1:
        novo = estado + rng.choice((-1, 1))
        terminal = novo in (0, n + 1)
        yield estado, float(novo == n + 1), novo, terminal
        estado = novo


def mc_atualizar(valores, trilha, alpha, gamma=1.0):
    """Every-visit, com passo fixo. Aguarde o fim do episódio."""
    if not trilha or not trilha[-1][3]:
        raise ValueError("Monte Carlo precisa de um episódio completo.")
    g = 0.0
    for estado, r, _, _ in reversed(trilha):
        g = r + gamma * g
        valores[estado] += alpha * (g - valores[estado])


def td_atualizar(valores, transicao, alpha, gamma=1.0):
    """Bootstrap do próximo estado apenas quando ele não é terminal."""
    estado, r, novo, terminal = transicao
    alvo = r if terminal else r + gamma * valores[novo]
    delta = alvo - valores[estado]
    valores[estado] += alpha * delta
    return delta


def erro(valores, referencia):
    return sqrt(sum((v - referencia[s]) ** 2 for s, v in valores.items()) / len(valores))


def comparar(episodios=100, sementes=100, n=5, alpha_mc=0.03, alpha_td=0.1):
    """RMSE médio em seeds independentes, com trilhas pareadas."""
    referencia = {s: s / (n + 1) for s in range(1, n + 1)}
    total_mc, total_td = [0.] * (episodios + 1), [0.] * (episodios + 1)
    for seed in range(sementes):
        rng = Random(seed)
        m = dict.fromkeys(referencia, 0.5)
        t = m.copy()
        for ep in range(episodios + 1):
            total_mc[ep] += erro(m, referencia)
            total_td[ep] += erro(t, referencia)
            if ep < episodios:
                trilha = []
                for passo in caminhada(rng, n):
                    td_atualizar(t, passo, alpha_td)
                    trilha.append(passo)
                mc_atualizar(m, trilha, alpha_mc)
    return [(m / sementes, t / sementes) for m, t in zip(total_mc, total_td)]


def main():
    corredor = [(1, 0, 2, False), (2, 0, 3, False), (3, 1, 4, True)]
    m, t = dict.fromkeys((1, 2, 3), 0.0), dict.fromkeys((1, 2, 3), 0.0)
    mc_atualizar(m, corredor, 1.0, 0.9)
    for passo in corredor:
        td_atualizar(t, passo, 1.0, 0.9)
    print("Após um episódio de corredor, γ=0,9 e α=1:", "MC", m, "| TD", t)
    print("Caminhada aleatória: RMSE médio em 100 sementes")
    curvas = comparar()
    for ep in (0, 1, 10, 50, 100):
        mc, td = curvas[ep]
        print(f"  após {ep:3d} episódios: MC={mc:.4f}; TD(0)={td:.4f}")


if __name__ == "__main__":
    main()

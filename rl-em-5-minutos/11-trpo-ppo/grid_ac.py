"""Grid 4×3 do vídeo de Actor-Critic (coordenadas linha, coluna)."""

LINHAS, COLUNAS = 3, 4
INICIO, BANDEIRA, BURACO, PAREDE = (2, 0), (0, 3), (1, 3), (1, 1)
ACOES = "CDBE"
MOVIMENTOS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def step(estado, acao):
    dl, dc = MOVIMENTOS[acao]
    l, c = estado[0] + dl, estado[1] + dc
    proximo = (l, c)
    if not (0 <= l < LINHAS and 0 <= c < COLUNAS) or proximo == PAREDE:
        proximo = estado
    if proximo == BANDEIRA:
        return proximo, 1.0, True
    if proximo == BURACO:
        return proximo, -1.0, True
    return proximo, -0.04, False


def indice(estado):
    l, c = estado
    return l * COLUNAS + c

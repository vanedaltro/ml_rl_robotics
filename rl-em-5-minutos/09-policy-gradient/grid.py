"""RL em 5 minutos, ep. 1: ambiente e trajetórias, sem treinamento."""
LINHAS, COLUNAS = 3, 5
INICIO, BANDEIRA = (2, 0), (0, 4)
PAREDE, BURACO = (1, 2), (1, 0)
MOVIMENTOS = {"C": (-1, 0), "D": (0, 1), "B": (1, 0), "E": (0, -1)}

def step(estado, acao):
    """Devolve (novo estado, recompensa, terminou), sem política embutida."""
    if estado in (BANDEIRA, BURACO):
        raise ValueError("Episódio encerrado; comece outro antes de agir.")
    dl, dc = MOVIMENTOS[acao]
    nl, nc = estado[0] + dl, estado[1] + dc
    destino = (nl, nc)
    if not (0 <= nl < LINHAS and 0 <= nc < COLUNAS) or destino == PAREDE:
        destino = estado
    if destino == BANDEIRA:
        return destino, 10, True
    if destino == BURACO:
        return destino, -10, True
    return destino, -1, False

def episodio(acoes):
    estado, trilha = INICIO, []
    for acao in acoes:
        novo, recompensa, terminou = step(estado, acao)
        trilha.append((estado, acao, recompensa, novo, terminou))
        estado = novo
        if terminou:
            break
    return trilha

def retorno(trilha, gamma=1.0):
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma fora do intervalo [0, 1]")
    g = 0.0
    for _, _, recompensa, _, _ in reversed(trilha):
        g = recompensa + gamma * g
    return g

if __name__ == "__main__":
    sucesso = episodio("DDDDCC")
    falha = episodio("EC")
    incompleto = episodio("DD")
    assert [x[2] for x in sucesso] == [-1] * 5 + [10]
    assert sucesso[-1][3:] == (BANDEIRA, True) and retorno(sucesso) == 5
    assert falha[0][0] == falha[0][3] and falha[0][2] == -1
    assert falha[-1][3:] == (BURACO, True) and retorno(falha) == -11
    assert step((2, 2), "C") == ((2, 2), -1, False)
    assert len(episodio("ECDD")) == 2
    assert abs(retorno(sucesso, .9) - 1.8098) < 1e-9
    assert not incompleto[-1][-1]
    print("8 verificações passaram.")
    print("Bandeira:", [x[2] for x in sucesso], "G =", retorno(sucesso))
    print("Buraco:  ", [x[2] for x in falha], "G =", retorno(falha))
    print("Bandeira com γ = 0,9: G =", round(retorno(sucesso, .9), 4))

"""DQN no GridWorld do episódio 01: replay, rede-alvo e entradas em coordenadas."""
import argparse
from collections import deque
from random import Random
from typing import NamedTuple

import torch
from torch import nn

INICIO, META, BURACO, PAREDE = (2, 0), (0, 4), (1, 0), (1, 2)
MOVIMENTOS = ((-1, 0), (0, 1), (1, 0), (0, -1))
State = tuple[int, int]


def step(estado, acao):
    if estado in (META, BURACO):
        raise ValueError("Comece outro episódio antes de agir em um estado terminal.")
    dl, dc = MOVIMENTOS[acao]
    l, c = estado[0] + dl, estado[1] + dc
    novo = (l, c) if 0 <= l < 3 and 0 <= c < 5 and (l, c) != PAREDE else estado
    if novo == META:
        return novo, 10.0, True
    if novo == BURACO:
        return novo, -10.0, True
    return novo, -1.0, False


class Transicao(NamedTuple):
    estado: State
    acao: int
    recompensa: float
    novo: State
    terminou: bool


class Replay:
    def __init__(self, capacidade):
        self.dados = deque(maxlen=capacidade)

    def __len__(self):
        return len(self.dados)

    def guardar(self, transicao):
        self.dados.append(transicao)

    def amostra(self, tamanho, rng):
        return rng.sample(list(self.dados), tamanho)


def tensor_de_estados(estados):
    return torch.tensor([(linha / 2, coluna / 4) for linha, coluna in estados],
                        dtype=torch.float32)


def rede_q():
    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32),
                         nn.ReLU(), nn.Linear(32, len(MOVIMENTOS)))


def sincronizar(alvo, online):
    alvo.load_state_dict(online.state_dict())
    alvo.eval()
    alvo.requires_grad_(False)


@torch.no_grad()
def alvos(recompensas, terminou, novos_q, gamma):
    maximos = novos_q.max(dim=1).values
    return recompensas + gamma * maximos.masked_fill(terminou, 0.0)


def atualizar(online, alvo, otimizador, lote, gamma=0.99):
    estados, acoes, recompensas, novos, terminou = zip(*lote)
    previstos = online(tensor_de_estados(estados)).gather(
        1, torch.tensor(acoes, dtype=torch.long).reshape(-1, 1)
    ).squeeze(1)
    with torch.no_grad():
        valores_alvo = alvos(
            torch.tensor(recompensas, dtype=torch.float32),
            torch.tensor(terminou, dtype=torch.bool),
            alvo(tensor_de_estados(novos)),
            gamma,
        )
    perda = nn.functional.mse_loss(previstos, valores_alvo)
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()
    return perda.item()


@torch.no_grad()
def escolher(online, estado, epsilon, rng):
    if epsilon and rng.random() < epsilon:
        return rng.randrange(len(MOVIMENTOS))
    return online(tensor_de_estados([estado])).argmax(dim=1).item()


def avaliar(online, max_passos=40):
    estado, retorno, trajeto = INICIO, 0.0, []
    rng = Random(0)
    for _ in range(max_passos):
        acao = escolher(online, estado, 0, rng)
        trajeto.append("CDBE"[acao])
        estado, recompensa, terminou = step(estado, acao)
        retorno += recompensa
        if terminou:
            break
    return retorno, estado == META, "".join(trajeto)


def treinar(episodios=250, seed=42, max_passos=40, tamanho_lote=32,
            intervalo_alvo=50):
    torch.manual_seed(seed)
    rng = Random(seed)
    rng_replay = Random(seed + 1)
    online, alvo = rede_q(), rede_q()
    sincronizar(alvo, online)
    otimizador = torch.optim.Adam(online.parameters(), lr=0.003)
    replay = Replay(2048)
    passos = sucessos = atualizacoes = 0
    inicial = avaliar(online, max_passos)
    for ep in range(episodios):
        epsilon = max(0.05, 1.0 - ep / max(1, 0.7 * episodios))
        estado = INICIO
        for _ in range(max_passos):
            acao = escolher(online, estado, epsilon, rng)
            novo, recompensa, terminou = step(estado, acao)
            # A expiração do limite externo não é um terminal do GridWorld.
            replay.guardar(Transicao(estado, acao, recompensa, novo, terminou))
            passos += 1
            if len(replay) >= tamanho_lote:
                atualizar(online, alvo, otimizador,
                          replay.amostra(tamanho_lote, rng_replay))
                atualizacoes += 1
                if atualizacoes % intervalo_alvo == 0:
                    sincronizar(alvo, online)
            estado = novo
            if terminou:
                sucessos += estado == META
                break
    return inicial, avaliar(online, max_passos), passos, sucessos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodios", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodios < 1:
        parser.error("--episodios deve ser positivo")
    torch.set_num_threads(1)
    inicial, final, passos, sucessos = treinar(args.episodios, args.seed)
    print(f"PyTorch {torch.__version__} (CPU); seed {args.seed}")
    print(f"Treino: {args.episodios} episódios, {passos} passos, {sucessos} metas")
    for etapa, (g, meta, trilha) in (("Antes", inicial), ("Depois", final)):
        print(f"{etapa}: retorno={g:g}; meta={meta}; trajeto={trilha}")


if __name__ == "__main__":
    main()

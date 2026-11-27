"""Capítulo 14: coordenação cooperativa e equilíbrio competitivo."""

import argparse
from dataclasses import dataclass
from math import exp
from random import Random
from statistics import mean
from typing import Sequence, Tuple


ACTIONS = (0, 1)
JointAction = Tuple[int, int]
Matrix = Sequence[Sequence[float]]


def team_reward(actions: JointAction) -> float:
    """Cada robô escolhe um corredor; só avançam sem colisão se se separarem."""
    return 1.0 if actions[0] != actions[1] else -1.0


def prob_one(logit: float) -> float:
    if logit >= 0:
        return 1 / (1 + exp(-logit))
    e = exp(logit)
    return e / (1 + e)


def greedy(values: Sequence[float], rng: Random) -> int:
    return rng.choice([a for a in ACTIONS if values[a] == max(values)])


@dataclass
class Trial:
    collisions: int
    episodes: int

    @property
    def success(self) -> float:
        return 1 - self.collisions / self.episodes


def independent_q(seed: int, episodes: int) -> Tuple[Trial, JointAction]:
    """Bandits independentes: cada robô estima recompensa dada só sua ação."""
    rng = Random(seed)
    q = [[0.0, 0.0] for _ in ACTIONS]
    counts = [[0, 0] for _ in ACTIONS]
    collisions = 0
    for _ in range(episodes):
        actions = tuple(rng.choice(ACTIONS) if rng.random() < 0.15 else greedy(q[i], rng)
                        for i in ACTIONS)
        reward = team_reward(actions)
        collisions += reward < 0
        for i, a in enumerate(actions):
            counts[i][a] += 1
            q[i][a] += (reward - q[i][a]) / counts[i][a]
    return Trial(collisions, episodes), tuple(greedy(values, rng) for values in q)


def counterfactual(q: Matrix, actions: JointAction, agent: int, p_one: float) -> float:
    """Marginaliza a ação própria e mantém a ação do outro robô fixa."""
    candidate = list(actions)
    candidate[agent] = 0
    value_zero = q[candidate[0]][candidate[1]]
    candidate[agent] = 1
    value_one = q[candidate[0]][candidate[1]]
    return q[actions[0]][actions[1]] - ((1 - p_one) * value_zero + p_one * value_one)



def actor_gradient(q: Matrix, p: Sequence[float]) -> Tuple[float, float]:
    """Gradiente do retorno esperado na tabela central, em relação aos logits."""
    gradients = [0.0, 0.0]
    for a0 in ACTIONS:
        for a1 in ACTIONS:
            actions = (a0, a1)
            joint_probability = (p[0] if a0 else 1 - p[0]) * (p[1] if a1 else 1 - p[1])
            for i, a in enumerate(actions):
                # Derivada do log de Bernoulli em relação ao logit: a - p.
                gradients[i] += joint_probability * counterfactual(q, actions, i, p[i]) * (a - p[i])
    return gradients[0], gradients[1]


def train_ctde(seed: int, episodes: int,
               batch_size: int = 20) -> Tuple[Trial, JointAction, Tuple[float, float]]:
    """Crítico tabular compartilhado e dois atores Bernoulli locais.

    Cada lote atualiza as médias do crítico com recompensas observadas.
    Atualizamos os atores com a expectativa exata sob as políticas que
    coletaram o lote, usando a estimativa corrente de Q centralizada.
    Essa enumeração é possível aqui porque há só quatro ações conjuntas.
    """
    if batch_size < 1:
        raise ValueError("batch_size deve ser positivo")
    if episodes < 4:
        raise ValueError("CTDE precisa de quatro episódios iniciais")
    rng = Random(seed)
    logits = [rng.uniform(-0.25, 0.25) for _ in ACTIONS]
    q = [[0.0, 0.0] for _ in ACTIONS]
    counts = [[0, 0] for _ in ACTIONS]
    collisions = 0

    def observe(a0, a1):
        reward = team_reward((a0, a1))
        counts[a0][a1] += 1
        q[a0][a1] += (reward - q[a0][a1]) / counts[a0][a1]
        return reward < 0

    # Exploração central durante o treino: observamos os quatro pares uma vez.
    for a0 in ACTIONS:
        for a1 in ACTIONS:
            collisions += observe(a0, a1)
    for start in range(4, episodes, batch_size):
        p = [prob_one(z) for z in logits]
        for _ in range(start, min(episodes, start + batch_size)):
            a0, a1 = (int(rng.random() < pi) for pi in p)
            collisions += observe(a0, a1)
        gradients = actor_gradient(q, p)
        for i in ACTIONS:
            logits[i] += 1.5 * gradients[i]

    # Cada ator agora pode executar sem crítico e sem observar a ação do outro.
    p0, p1 = (prob_one(logit) for logit in logits)
    return Trial(collisions, episodes), (int(p0 >= 0.5), int(p1 >= 0.5)), (p0, p1)


# 0 = pedra; 1 = papel; 2 = tesoura.
def rps_reward(first: int, second: int) -> int:
    outcome = (first - second) % 3
    return (0, 1, -1)[outcome]


def best_response(pure_opponent: int) -> int:
    return (pure_opponent + 1) % 3


def exploitability(opponent_probs: Sequence[float]) -> float:
    """Valor da melhor resposta pura (valor minimax do jogo = 0)."""
    return max(sum(rps_reward(a, b) * p for b, p in enumerate(opponent_probs))
               for a in range(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sementes", type=int, default=32)
    parser.add_argument("--episodios", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42, help="primeira semente")
    args = parser.parse_args()
    if args.sementes < 1 or args.episodios < 24:
        parser.error("use ao menos uma semente e 24 episódios")
    independent = [independent_q(s, args.episodios)
                   for s in range(args.seed, args.seed + args.sementes)]
    central = [train_ctde(s, args.episodios)
               for s in range(args.seed, args.seed + args.sementes)]
    print(f"Coordenação | {args.sementes} sementes desde {args.seed}, "
          f"{args.episodios} episódios/treino")
    print("método               sucesso/treino   sucesso/execução gulosa")
    for label, runs in [("Independente", independent), ("CTDE", central)]:
        train_success = mean(run[0].success for run in runs)
        greedy_success = mean(team_reward(run[1]) == 1 for run in runs)
        print(f"{label:19s} {train_success:14.2%} {greedy_success:23.2%}")
    p0, p1 = central[0][2]
    print(f"CTDE semente {args.seed}: P(corredor 1) = ({p0:.4f}, {p1:.4f}), "
          f"decisão local = {central[0][1]}")
    print("Pedra/papel/tesoura | melhor resposta à última jogada pura:")
    action = 0
    names = ["pedra", "papel", "tesoura"]
    cycle = [names[action]]
    for _ in range(3):
        action = best_response(action)
        cycle.append(names[action])
    print(" → ".join(cycle))
    print(f"Explorabilidade (pura / uniforme): {exploitability((1, 0, 0)):.2f} / "
          f"{exploitability((1 / 3, 1 / 3, 1 / 3)):.2f}")


if __name__ == "__main__":
    main()

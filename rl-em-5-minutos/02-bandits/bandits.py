"""Bandit estacionário com três alavancas, sem dependências externas."""
import argparse
import csv
from math import log, sqrt
from pathlib import Path
from random import Random

ARMS = ((3, 4, 5), (2, 5, 7, 10), (1, 3, 5))
MU = tuple(sum(arm) / len(arm) for arm in ARMS)
BEST = max(range(len(MU)), key=MU.__getitem__)
POLICIES = {"guloso": 0.0, "eps-010": 0.10, "eps-001": 0.01,
            "ucb": None, "oraculo": None}


def update(q, n, reward):
    """n já inclui a observação atual."""
    return q + (reward - q) / n


def choose(policy, q, n, t, rng):
    if policy == "oraculo":
        return BEST  # referência que conhece as médias verdadeiras
    if 0 in n:
        return n.index(0)
    if policy == "ucb":
        scores = [v + 3.0 * sqrt(log(t) / count) for v, count in zip(q, n)]
    else:
        epsilon = POLICIES[policy]
        if epsilon and rng.random() < epsilon:
            return rng.randrange(len(q))
        scores = q
    best = max(scores)
    return rng.choice([i for i, score in enumerate(scores) if score == best])


def rollout(policy, rounds, seed):
    # Fluxos independentes por alavanca, compartilhados entre políticas:
    # a k-ésima puxada da mesma alavanca rende a mesma amostra na mesma réplica.
    seeds = Random(seed)
    streams = [Random(seeds.getrandbits(64)) for _ in ARMS]
    rng = Random(seeds.getrandbits(64))
    n = [0] * len(ARMS)
    q = [0.0] * len(ARMS)
    reward_sum = gap_sum = best_pulls = 0
    for t in range(1, rounds + 1):
        a = choose(policy, q, n, t, rng)
        reward = streams[a].choice(ARMS[a])
        n[a] += 1
        q[a] = update(q[a], n[a], reward)
        reward_sum += reward
        gap_sum += MU[BEST] - MU[a]
        best_pulls += a == BEST
        yield reward_sum / t, gap_sum, best_pulls / t


def experiment(rounds, runs, seed):
    seeds = Random(seed)
    replicas = [seeds.getrandbits(64) for _ in range(runs)]
    for policy in POLICIES:
        totals = [[0.0, 0.0, 0.0] for _ in range(rounds)]
        for replica in replicas:
            for total, (reward, gap, share) in zip(totals, rollout(policy, rounds, replica)):
                total[0] += reward
                total[1] += gap
                total[2] += share
        for t, (reward, gap, share) in enumerate(totals, 1):
            yield policy, t, reward / runs, gap / runs, share / runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rodadas", type=int, default=1000)
    parser.add_argument("--repeticoes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, default=Path("resultados.csv"))
    args = parser.parse_args()
    if args.rodadas < len(ARMS) or args.repeticoes < 1:
        parser.error("use pelo menos três rodadas e uma repetição")
    with args.csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["politica", "rodada", "recompensa_media", "pseudo_regret", "fracao_B"])
        for row in experiment(args.rodadas, args.repeticoes, args.seed):
            writer.writerow(row)
            policy, t, reward, gap, share = row
            if t in {25, 100, args.rodadas}:
                print(f"{policy:8s} t={t:4d} recompensa/rodada={reward:.4f} "
                      f"pseudo-regret={gap:.3f} B={share:.2%}")


if __name__ == "__main__":
    main()

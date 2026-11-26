"""Capítulo 21: ambiente discreto de brinquedo. Não controla hardware."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

GOAL, HORIZON = 8, 12
LEFT, STEP, GRIP = range(3)
ACTIONS = (LEFT, STEP, GRIP)
ACTION_NAMES = ("esquerda", "passo", "tração")
DRY, WET = frozenset(), frozenset({3, 5})

Policy = Callable[[int, int], int]  # posição, passos restantes -> ação
Dynamics = Callable[[int, int], int]


def transition(s: int, a: int, wet: frozenset[int]) -> int:
    """O passo comum patina quando parte de uma casa molhada."""
    if s == GOAL:
        return GOAL
    if a == LEFT:
        return max(0, s - 1)
    if a == STEP and s in wet:
        return s
    return s + 1


def reward(s: int, a: int, next_s: int) -> int:
    if s == GOAL:
        return 0
    return (12 if next_s == GOAL else 0) - (3 if a == GRIP else 1)


def episode(start: int, policy: Policy, wet: frozenset[int]) -> dict:
    s = start
    positions, actions = [s], []
    total = 0
    for remaining in range(HORIZON, 0, -1):
        if s == GOAL:
            break
        a = policy(s, remaining)
        next_s = transition(s, a, wet)
        total += reward(s, a, next_s)
        s = next_s
        positions.append(s)
        actions.append(a)
    return dict(success=s == GOAL, total=total, positions=positions, actions=actions)


def explore(seed: int, count: int, wet: frozenset[int]) -> list[tuple[int, int, int]]:
    """Cada episódio inteiro pertence ao treino OU ao teste."""
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        run = episode(rng.randrange(GOAL), lambda s, h: rng.choice(ACTIONS), wet)
        samples.extend(zip(run["positions"], run["actions"], run["positions"][1:]))
    return samples


def demonstrations(seed: int, count: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        run = episode(rng.randrange(4), lambda s, h: STEP, DRY)
        samples.extend(zip(run["positions"], run["actions"]))
    return samples


def inputs(states, actions=None) -> torch.Tensor:
    x = F.one_hot(torch.tensor(states), GOAL + 1).float()
    if actions is None:
        return x
    return torch.cat((x, F.one_hot(torch.tensor(actions), len(ACTIONS)).float()), dim=-1)


def fit(x: torch.Tensor, y, classes: int) -> nn.Module:
    net = nn.Sequential(nn.Linear(x.shape[1], 48), nn.Tanh(), nn.Linear(48, classes))
    opt = torch.optim.Adam(net.parameters(), lr=0.025)
    target = torch.tensor(y)
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(x), target)
        loss.backward()
        opt.step()
    return net.eval()


@torch.inference_mode()
def predict(net: nn.Module, s, a=None) -> list[int]:
    return net(inputs(s, a)).argmax(dim=-1).tolist()


def planner(dynamics: Dynamics) -> Policy:
    """Bellman com horizonte finito; a função dynamics pode ser trocada."""
    values = [0] * (GOAL + 1)
    choices = {}
    for h in range(1, HORIZON + 1):
        updated = [0] * (GOAL + 1)
        for s in range(GOAL):
            scores = []
            for a in ACTIONS:
                next_s = dynamics(s, a)
                scores.append(reward(s, a, next_s) + values[next_s])
            best = max(ACTIONS, key=scores.__getitem__)
            updated[s] = scores[best]
            choices[h, s] = best
        values = updated
    return lambda s, h: choices[h, s]


def accuracy(net: nn.Module, rows: list[tuple[int, int, int]]) -> dict:
    s, a, y = zip(*rows)
    predicted = predict(net, s, a)
    correct = sum(p == t for p, t in zip(predicted, y))
    return dict(correct=correct, n=len(y), accuracy=correct / len(y))


def experiment() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(21)
    torch.use_deterministic_algorithms(True)
    train_s, train_a, train_y = zip(*explore(2101, 80, DRY))
    demo_s, demo_a = zip(*demonstrations(2102, 40))
    predictor = fit(inputs(train_s, train_a), train_y, GOAL + 1)
    clone = fit(inputs(demo_s), demo_a, len(ACTIONS))

    pairs = [(s, a) for s in range(GOAL) for a in ACTIONS]
    table = dict(zip(pairs, predict(predictor, *zip(*pairs))))
    clone_actions = predict(clone, list(range(GOAL)))
    learned_plan = planner(lambda s, a: table[s, a])

    starts_rng = random.Random(2104)
    starts = [starts_rng.randrange(4) for _ in range(32)]
    results = {}
    for label, wet in (("seco", DRY), ("molhado", WET)):
        held_out = explore(2103, 32, wet)
        result = {"preditor": accuracy(predictor, held_out)}
        # Pares alterados: onde o passo comum parte das casas 3 ou 5.
        critical = [row for row in held_out if row[0] in WET and row[1] == STEP]
        result["preditor_pares_criticos"] = accuracy(predictor, critical)
        for name, policy in (
            ("clonagem", lambda s, h: clone_actions[s]),
            ("planejador_modelo_aprendido", learned_plan),
            ("planejador_modelo_exato", planner(lambda s, a: transition(s, a, wet))),
        ):
            runs = [episode(s, policy, wet) for s in starts]
            result[name] = dict(
                successes=sum(run["success"] for run in runs),
                n=len(runs),
                mean_return=fmean(run["total"] for run in runs),
                from_zero=episode(0, policy, wet),
            )
        results[label] = result

    return dict(torch_version=torch.__version__, device="cpu", seed=21,
                train_transition_rows=len(train_y), train_demo_rows=len(demo_a),
                start_positions=starts, results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Também grava as métricas em JSON.")
    args = parser.parse_args()
    output = json.dumps(experiment(), indent=2, ensure_ascii=False)
    print(output)
    if args.json:
        args.json.write_text(output + "\n", encoding="utf-8")

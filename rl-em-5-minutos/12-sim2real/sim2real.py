"""Transferência para uma planta simulada, usando somente a biblioteca padrão."""
from collections import deque
from dataclasses import dataclass
from itertools import product
from random import Random
from statistics import mean

TARGET = 1.0
STEPS = 80


@dataclass(frozen=True)
class Plant:
    gain: float = 0.8
    drag: float = 0.25
    delay: int = 0
    noise: float = 0.0


@dataclass(frozen=True)
class Policy:
    ff: float
    kp: float

    def action(self, observed_speed, target):
        # A política não recebe ganho, atrito, atraso nem velocidade sem ruído.
        return min(1.0, max(-1.0, self.ff * target + self.kp * (target - observed_speed)))


@dataclass(frozen=True)
class Score:
    reward: float
    mae: float
    action_change: float


def rollout(policy, plant, seed, steps=STEPS):
    rng = Random(seed)
    pending = deque([0.0] * plant.delay)
    v = previous = 0.0
    rewards, errors, changes = [], [], []
    for t in range(steps):
        # Mudança de referência para que estabilidade e atraso sejam visíveis.
        target = TARGET if t < steps // 2 else -TARGET
        a = policy.action(v + rng.gauss(0, plant.noise), target)
        pending.append(a)
        applied = pending.popleft()
        v += 0.15 * (plant.gain * applied - plant.drag * v)
        error, delta = target - v, a - previous
        rewards.append(-(error**2 + 0.03 * a**2 + 0.02 * delta**2))
        errors.append(abs(error))
        changes.append(delta**2)
        previous = a
    return Score(mean(rewards), mean(errors), mean(changes))


def scenarios(seed, count, stress=False):
    rng = Random(seed)
    for _ in range(count):
        yield Plant(gain=rng.uniform(0.50, 1.05) if not stress else rng.uniform(0.32, 0.49),
                    drag=rng.uniform(0.15, 0.40),
                    delay=rng.randrange(5) if not stress else rng.randrange(5, 8),
                    noise=rng.uniform(0, 0.06)), rng.getrandbits(64)


def evaluate(policy, cases):
    scores = [rollout(policy, plant, seed) for plant, seed in cases]
    return Score(mean(x.reward for x in scores), mean(x.mae for x in scores),
                 mean(x.action_change for x in scores))


def train(cases):
    # Busca de política por recompensa, sem gradiente ou supervisor.
    candidates = (Policy(ff, kp) for ff, kp in product(
        (0.10, 0.25, 0.40, 0.55, 0.70), (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)))
    return max(candidates, key=lambda p: evaluate(p, cases).reward)


def main():
    nominal_cases = [(Plant(), 0)]
    randomized_train = list(scenarios(17, 36))
    holdout = list(scenarios(71, 100))
    stress = list(scenarios(123, 100, stress=True))

    for name, cases in (("nominal", nominal_cases), ("randomizada", randomized_train)):
        policy = train(cases)
        print(f"{name}: ff={policy.ff:.2f}, kp={policy.kp:.1f}")
        for label, evaluation in (("nominal", nominal_cases), ("holdout", holdout), ("estresse", stress)):
            m = evaluate(policy, evaluation)
            print(f"  {label:8s} recompensa={m.reward:.4f} MAE={m.mae:.4f} Δação²={m.action_change:.4f}")


if __name__ == "__main__":
    main()

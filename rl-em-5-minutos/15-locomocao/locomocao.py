"""Locomoção sagital didática: alvos articulares, PD e busca por retorno."""

import argparse
import math
import random
import statistics
from dataclasses import dataclass

POLICY_DT = 0.04
PD_STEPS = 8
DT = POLICY_DT / PD_STEPS
HORIZON = 125  # segundos simulados: 125 × 0,04 = 5
FALL = 0.50   # radianos de inclinação, limiar artificial
TAU_MAX = 5.0
BOUNDS = ((0.0, 0.60), (0.0, 0.40), (0.0, 5.0), (0.0, 1.0))
BASELINE = (0.0, 0.0, 0.0, 0.0)
INITIAL = (0.20, 0.12, 1.0, 0.15)


def clamp(value, low, high):
    return max(low, min(high, value))


def pd_torque(target, position, velocity):
    """Servo para alvo articular local, com saturação de torque."""
    return clamp(30.0 * (target - position) - 1.7 * velocity, -TAU_MAX, TAU_MAX)


@dataclass(frozen=True)
class Scenario:
    command: float
    traction: float
    pitch: float
    push_speed: float
    push_pitch: float


@dataclass(frozen=True)
class Episode:
    ret: float
    steps: int
    mean_error: float
    fell: bool


def scenario(seed, difficulty):
    """Aumenta faixa de comandos, variação de tração e empurrões."""
    rng = random.Random(seed)
    return Scenario(
        command=rng.uniform(0.25, 0.45 + 0.25 * difficulty),
        traction=rng.uniform(1.0 - 0.35 * difficulty, 1.0 + 0.20 * difficulty),
        pitch=rng.uniform(-0.035, 0.035),
        push_speed=rng.uniform(-0.35, 0.35) * difficulty,
        push_pitch=rng.uniform(-0.45, 0.45) * difficulty,
    )


def targets(weights, phase, pitch, pitch_rate, command):
    """Uma política gera os dois alvos a partir do comando e do estado."""
    amp_base, amp_speed, balance, damping = weights
    amplitude = clamp(amp_base + amp_speed * command, 0.0, 0.65)
    correction = clamp(-balance * pitch - damping * pitch_rate, -0.45, 0.45)
    swing = amplitude * math.cos(phase)
    return correction - swing, correction + swing


def rollout(weights, config, horizon=HORIZON):
    speed, pitch, pitch_rate = 0.0, config.pitch, 0.0
    left = right = dleft = dright = 0.0
    previous = (0.0, 0.0)
    phase = 0.0
    phase_rate = 2.0 * math.pi * (1.0 + config.command)
    total = error_sum = 0.0
    for t in range(horizon):
        target_l, target_r = targets(weights, phase, pitch, pitch_rate, config.command)
        if t == horizon // 2:
            speed += config.push_speed
            pitch_rate += config.push_pitch
        effort = 0.0
        fell = False
        for _ in range(PD_STEPS):
            tau_l = pd_torque(target_l, left, dleft)
            tau_r = pd_torque(target_r, right, dright)
            dleft += DT * (tau_l - 0.12 * dleft) / 0.18
            dright += DT * (tau_r - 0.12 * dright) / 0.18
            left += DT * dleft
            right += DT * dright
            # Contato prescrito: na meia-volta positiva, a esquerda
            # apoia; na outra meia-volta, a direita apoia.
            stance = math.sin(phase)
            drive = config.traction * 0.98 * (
                max(0.0, stance) * dleft + max(0.0, -stance) * dright
            )
            acceleration = drive - 2.7 * speed
            speed += DT * acceleration
            balance = 8.0 * (left + right) / 2.0
            pitch_rate += DT * (3.0 * pitch + balance + 0.18 * acceleration - 0.9 * pitch_rate)
            pitch += DT * pitch_rate
            phase = (phase + phase_rate * DT) % (2.0 * math.pi)
            effort += (tau_l * tau_l + tau_r * tau_r) / PD_STEPS
            if abs(pitch) >= FALL:
                fell = True
                break
        error = config.command - speed
        jitter = (target_l - previous[0]) ** 2 + (target_r - previous[1]) ** 2
        previous = (target_l, target_r)
        total += math.exp(-3.0 * error ** 2) - 3.0 * pitch ** 2 - 0.08 * jitter - 0.002 * effort
        error_sum += abs(error)
        if fell:
            return Episode(total - 5.0, t + 1, error_sum / (t + 1), True)
    return Episode(total, horizon, error_sum / horizon, False)


def score(weights, scenarios):
    return statistics.mean(rollout(weights, config).ret for config in scenarios)


def train(seed, iterations=14, population=18):
    """CEM: selecionar políticas pelo retorno do episódio, sem gradiente."""
    if iterations < 1 or population < 4:
        raise ValueError("use ao menos uma iteração e quatro candidatos")
    rng = random.Random(seed)
    center = list(INITIAL)
    spread = [0.20, 0.15, 1.4, 0.30]
    selection_scenarios = [scenario(rng.getrandbits(64), 1.0) for _ in range(4)]
    best = tuple(center)
    best_score = score(best, selection_scenarios)
    for iteration in range(iterations):
        difficulty = (iteration + 1) / iterations
        configs = [scenario(rng.getrandbits(64), difficulty) for _ in range(3)]
        candidates = [tuple(center)]
        for _ in range(population - 1):
            candidates.append(tuple(clamp(rng.gauss(m, s), lo, hi)
                                    for m, s, (lo, hi) in zip(center, spread, BOUNDS)))
        ranked = sorted(candidates, key=lambda w: score(w, configs), reverse=True)
        elite = ranked[:max(2, population // 4)]
        for i, column in enumerate(zip(*elite)):
            center[i] = statistics.mean(column)
            spread[i] = max(0.025, statistics.pstdev(column) * 1.2)
        for candidate in (ranked[0], tuple(center)):
            value = score(candidate, selection_scenarios)
            if value > best_score:
                best, best_score = candidate, value
    return best


def evaluation(weights, scenarios):
    results = [rollout(weights, config) for config in scenarios]
    complete = [row for row in results if not row.fell]
    return (
        statistics.mean(row.ret for row in results),
        len(results) - len(complete),
        statistics.mean(row.mean_error for row in complete) if complete else None,
        statistics.mean(row.steps for row in results),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iteracoes", type=int, default=14)
    parser.add_argument("--populacao", type=int, default=18)
    args = parser.parse_args()
    if args.iteracoes < 1 or args.populacao < 4:
        parser.error("use ao menos uma iteração e quatro candidatos")
    trained = train(args.seed, args.iteracoes, args.populacao)
    # Sementes pequenas reservadas à avaliação; o treino não usa este conjunto.
    configs = [scenario(seed, 1.0) for seed in range(10001, 10021)]
    print(f"pesos treinados={tuple(round(x, 4) for x in trained)}")
    for label, weights in (("alvos neutros", BASELINE), ("política inicial", INITIAL),
                           ("política treinada", trained)):
        ret, falls, error, steps = evaluation(weights, configs)
        error_text = f"{error:.3f}" if error is not None else "n/a"
        print(f"{label:18s} retorno={ret:7.2f}  quedas={falls:2d}/{len(configs)}  "
              f"erro|completo={error_text}  passos={steps:.1f}")


if __name__ == "__main__":
    main()

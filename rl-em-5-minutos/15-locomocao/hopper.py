"""Hopper-v5 no MuJoCo, sem janela: política → alvos → ações PD normalizadas."""

import argparse
import math
import statistics
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

NOMINAL = np.array([0.0, 0.0, 0.0])
JOINTS = ("thigh_joint", "leg_joint", "foot_joint")
PD_PER_POLICY = 4
INITIAL = np.zeros(11)
FITNESS_SEEDS = (123, 456)
EVALUATION_SEEDS = tuple(range(901, 909))


@dataclass(frozen=True)
class Result:
    ret: float
    steps: int
    displacement: float
    terminated: bool
    truncated: bool


def make_hopper(horizon=240):
    return gym.make("Hopper-v5", render_mode=None, max_episode_steps=horizon)


def joint_addresses(model):
    """Converte nomes do XML em índices, sem supor a ordem do vetor global."""
    dof = []
    for name in JOINTS:
        joint = model.joint(name)
        dof.append(int(joint.dofadr[0]))
    return np.array(dof)


def action_pd(target, q, dq):
    # Saída é o comando normalizado do Hopper, não torque em N m.
    return np.clip(5.0 * (target - q) - 0.22 * dq, -1.0, 1.0).astype(np.float32)


def pose(weights, t, tilt):
    offset, sin_weight, cos_weight = weights[:3], weights[3:6], weights[6:9]
    correction = weights[9] * tilt
    rate = math.tau * (1.5 + weights[10])
    return np.clip(NOMINAL + offset + sin_weight * math.sin(rate * t)
                   + cos_weight * math.cos(rate * t) + np.array([correction, 0.0, -correction]),
                   -1.2, 0.8)


def rollout(env, weights, seed, dof):
    env.reset(seed=seed)
    data = env.unwrapped.data
    root = env.unwrapped.model.joint("rootx").qposadr[0]
    qpos = [int(env.unwrapped.model.joint(name).qposadr[0]) for name in JOINTS]
    tilt_idx = env.unwrapped.model.joint("rooty").qposadr[0]
    x_start = data.qpos[root]
    total = 0.0
    step = 0
    while True:
        if step % PD_PER_POLICY == 0:
            target = pose(weights, data.time, data.qpos[tilt_idx])
        action = action_pd(target, data.qpos[qpos], data.qvel[dof])
        _, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        step += 1
        if terminated or truncated:
            return Result(total, step, float(data.qpos[root] - x_start),
                          bool(terminated), bool(truncated))


def train(env, dof, iterations, population, seed):
    rng = np.random.default_rng(seed)
    center = INITIAL.copy()
    spread = np.array([0.25, 0.25, 0.25, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.5, 0.5])
    best, best_return = INITIAL.copy(), -math.inf
    for _ in range(iterations):
        candidates = np.clip(center + spread * rng.standard_normal((population, len(center))),
                             -1.0, 1.0)
        candidates[0] = best
        returns = [statistics.mean(rollout(env, w, s, dof).ret for s in FITNESS_SEEDS)
                   for w in candidates]
        order = np.argsort(returns)[::-1]
        if returns[order[0]] > best_return:
            best, best_return = candidates[order[0]].copy(), returns[order[0]]
        elite = candidates[order[:max(2, population // 4)]]
        center = elite.mean(axis=0)
        spread = np.maximum(0.07, 1.25 * elite.std(axis=0))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteracoes", type=int, default=14)
    parser.add_argument("--populacao", type=int, default=24)
    parser.add_argument("--horizonte", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.iteracoes < 1 or args.populacao < 4 or args.horizonte < 1:
        parser.error("use ao menos uma iteração, quatro candidatos e um passo")
    env = make_hopper(args.horizonte)
    try:
        dof = joint_addresses(env.unwrapped.model)
        weights = train(env, dof, args.iteracoes, args.populacao, args.seed)
        print(f"Gymnasium {gym.__version__} / MuJoCo {mujoco.__version__} / Hopper-v5")
        print(f"PD={1 / env.unwrapped.dt:.2f} Hz, política={1 / (PD_PER_POLICY * env.unwrapped.dt):.2f} Hz")
        for name, w in (("alvo neutro", INITIAL), ("política CEM", weights)):
            rows = [rollout(env, w, s, dof) for s in EVALUATION_SEEDS]
            print(f"{name:14s} retorno={statistics.mean(r.ret for r in rows):7.2f}  "
                  f"dx={statistics.mean(r.displacement for r in rows):6.3f} m  "
                  f"passos={statistics.mean(r.steps for r in rows):5.1f}  "
                  f"terminais={sum(r.terminated for r in rows)}/{len(rows)} "
                  f"limite={sum(r.truncated for r in rows)}/{len(rows)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

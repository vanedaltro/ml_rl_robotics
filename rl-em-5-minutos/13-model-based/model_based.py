"""Modelo tabular aprendido, MPC e Dyna-Q em um corredor didático."""
from collections import Counter, defaultdict
from functools import lru_cache
from random import Random
from statistics import mean

STATES = tuple(range(6))
GOAL = 6
ACTIONS = ("anda", "salta")
GAMMA = 0.95


def step(state, action, rng):
    """Ambiente estocástico. O planejador não usa esta função."""
    if state == GOAL:
        raise ValueError("o episódio terminou")
    if action == "anda":
        nxt = state + 1
    elif action == "salta":
        p_stall = 0.60 if state in (2, 3) else 0.20
        nxt = state if rng.random() < p_stall else min(GOAL, state + 2)
    else:
        raise ValueError(f"ação desconhecida: {action}")
    return nxt, (9 if nxt == GOAL else -1)


class Model:
    """Estima P(s', r | s, a) pelas contagens observadas."""

    def __init__(self):
        self.counts = defaultdict(Counter)

    def observe(self, state, action, nxt, reward):
        self.counts[state, action][nxt, reward] += 1

    def predict(self, state, action):
        counts = self.counts.get((state, action))
        if not counts:
            raise ValueError("o modelo não observou este par estado/ação")
        n = sum(counts.values())
        return tuple((count / n, nxt, reward)
                     for (nxt, reward), count in sorted(counts.items()))


def train(samples, seed):
    """Amostragem balanceada em simulador com reinício livre em todo estado."""
    rng, model = Random(seed), Model()
    for s in STATES:
        for a in ACTIONS:
            for _ in range(samples):
                nxt, reward = step(s, a, rng)
                model.observe(s, a, nxt, reward)
    return model


def action_values(model, state, horizon):
    if state == GOAL or horizon < 1:
        raise ValueError("use um estado não terminal e horizonte positivo")

    @lru_cache(maxsize=None)
    def value(s, depth):
        return 0.0 if depth == 0 or s == GOAL else max(scores(s, depth).values())

    def scores(s, depth):
        return {a: sum(p * (reward + GAMMA * value(nxt, depth - 1))
                       for p, nxt, reward in model.predict(s, a))
                for a in ACTIONS}

    return scores(state, horizon)


def policy_mpc(model, horizon):
    """Replaneja a cada ação, usando o estado observado no ambiente."""
    @lru_cache(maxsize=None)
    def policy(state):
        # O cache reutiliza planos para estados iguais pois o modelo é congelado.
        values = action_values(model, state, horizon)
        return max(values, key=values.get)
    return policy


def evaluate(policy, seed, episodes=300, limit=25):
    rng = Random(seed)
    returns, successes = [], []
    for _ in range(episodes):
        s, total = 0, 0.0
        for t in range(limit):
            s, reward = step(s, policy(s), rng)
            total += GAMMA**t * reward
            if s == GOAL:
                break
        returns.append(total)
        successes.append(s == GOAL)
    return mean(returns), mean(successes)


def brier(model, seed, samples=100):
    """Brier multiclasses para o estado seguinte, em dados novos e balanceados."""
    rng, errors = Random(seed), []
    for s in STATES:
        for a in ACTIONS:
            probs = {nxt: p for p, nxt, _ in model.predict(s, a)}
            for _ in range(samples):
                nxt, _ = step(s, a, rng)
                errors.append(sum((probs.get(i, 0.0) - (i == nxt))**2
                                  for i in (*STATES, GOAL)))
    return mean(errors)


def q_update(q, s, a, nxt, reward, alpha=0.15):
    target = reward if nxt == GOAL else reward + GAMMA * max(q[nxt, b] for b in ACTIONS)
    q[s, a] += alpha * (target - q[s, a])


def dyna(real_data, planning_steps, seed):
    """Dyna-Q: mesma atualização Q na amostra real e nas amostras imaginadas."""
    q, model, rng = defaultdict(float), Model(), Random(seed)
    for s, a, nxt, reward in real_data:
        q_update(q, s, a, nxt, reward)
        model.observe(s, a, nxt, reward)
        for _ in range(planning_steps):
            s_m, a_m = rng.choice(tuple(model.counts))
            outcomes = model.predict(s_m, a_m)
            _, nxt_m, r_m = rng.choices(outcomes, weights=[item[0] for item in outcomes])[0]
            q_update(q, s_m, a_m, nxt_m, r_m)
    return q


def dataset(count, seed):
    rng = Random(seed)
    for _ in range(count):
        s, a = rng.choice(STATES), rng.choice(ACTIONS)
        nxt, reward = step(s, a, rng)
        yield s, a, nxt, reward


def main():
    models = {n: [train(n, seed) for seed in range(20)] for n in (1, 80)}
    for n, group in models.items():
        score = mean(brier(m, 99) for m in group)
        print(f"Brier holdout ({n} amostra(s)/par): {score:.4f}")
    ret, success = evaluate(lambda _: "anda", 101)
    print(f"só anda: retorno={ret:.4f} sucesso={success:.1%}")
    for n, horizon in ((1, 5), (80, 1), (80, 5)):
        scores = [evaluate(policy_mpc(m, horizon), 101) for m in models[n]]
        print(f"MPC ({n} amostra(s)/par, H={horizon}): "
              f"retorno={mean(pair[0] for pair in scores):.4f} "
              f"sucesso={mean(pair[1] for pair in scores):.1%}")

    for steps in (0, 10):
        scores = []
        for i in range(20):
            q = dyna(dataset(36, i), steps, seed=77)
            policy = lambda s: max(ACTIONS, key=lambda a: q[s, a])
            scores.append(evaluate(policy, 101, episodes=50)[0])
        print(f"Dyna: 36 transições coletadas/réplica, {steps} planos por transição, "
              f"retorno médio (20 réplicas)={mean(scores):.4f}")


if __name__ == "__main__":
    main()

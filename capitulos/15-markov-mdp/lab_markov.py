"""Capítulo 15: cadeia de Markov, dependência de memória e MDP, sem pacotes externos."""
import argparse
import random
from math import isclose

WEATHER_STATES = ("SOL", "CHUVA")
WEATHER = ((0.8, 0.2), (0.4, 0.6))
SECOND_ORDER_RAIN = {
    ("SOL", "SOL"): 0.1,
    ("SOL", "CHUVA"): 0.4,
    ("CHUVA", "SOL"): 0.3,
    ("CHUVA", "CHUVA"): 0.8,
}

STATES = ("partida", "corredor", "meta", "falha")
TERMINALS = {"meta", "falha"}
# Entrada: (probabilidade, próximo estado, recompensa da transição).
MDP = {
    "partida": {
        "atalho": ((0.7, "meta", 10.0), (0.3, "falha", -20.0)),
        "desvio": ((1.0, "corredor", -1.0),),
    },
    "corredor": {"avançar": ((1.0, "meta", 8.0),)},
    "meta": {},
    "falha": {},
}


def step(distribution, matrix):
    """Avança um vetor-linha de probabilidades: p(t+1) = p(t) P."""
    return tuple(sum(distribution[i] * row[j] for i, row in enumerate(matrix))
                 for j in range(len(matrix)))


def pair_matrix():
    """Transforma a cadeia de ordem 2 em uma cadeia nos pares consecutivos."""
    pairs = tuple(SECOND_ORDER_RAIN)
    rows = []
    for a, b in pairs:
        rain = SECOND_ORDER_RAIN[(a, b)]
        rows.append(tuple(
            (rain if d == "CHUVA" else 1 - rain) if c == b else 0.0
            for c, d in pairs
        ))
    return pairs, tuple(rows)


def q_value(outcomes, values, gamma):
    return sum(p * (reward + gamma * values[next_state])
               for p, next_state, reward in outcomes)


def value_iteration(gamma, tol=1e-12, max_iterations=100):
    """Atualiza todos os estados a partir da mesma iteração anterior."""
    if not 0 <= gamma < 1:
        raise ValueError("Use 0 <= gamma < 1.")
    if tol <= 0 or max_iterations < 1:
        raise ValueError("Use tol > 0 e max_iterations >= 1.")
    values = {state: 0.0 for state in STATES}
    for iteration in range(1, max_iterations + 1):
        next_values = {
            state: max((q_value(outcomes, values, gamma)
                        for outcomes in MDP[state].values()), default=0.0)
            for state in STATES
        }
        delta = max(abs(next_values[state] - values[state]) for state in STATES)
        values = next_values
        if delta < tol:
            break
    else:
        raise RuntimeError("A iteração de valores não convergiu no limite.")
    scores = {
        state: {action: q_value(outcomes, values, gamma)
                for action, outcomes in MDP[state].items()}
        for state in STATES if MDP[state]
    }
    # Empates seguem a ordem de inserção do dicionário de ações.
    policy = {state: max(actions, key=actions.get) for state, actions in scores.items()}
    return values, policy, scores, iteration


def policy_matrix(policy):
    """Matriz da cadeia induzida por uma política fixa determinística."""
    matrix = []
    for state in STATES:
        row = [0.0] * len(STATES)
        if state in TERMINALS:
            row[STATES.index(state)] = 1.0  # após terminar: absorção, recompensa zero
        else:
            for p, target, _ in MDP[state][policy[state]]:
                row[STATES.index(target)] += p
        matrix.append(tuple(row))
    return tuple(matrix)


def simulate_weather(trials, seed):
    """Trajetórias independentes, sempre partindo de SOL."""
    rng = random.Random(seed)
    rainy_tomorrow = rainy_second = 0
    for _ in range(trials):
        day1 = int(rng.random() < WEATHER[0][1])
        day2 = int(rng.random() < WEATHER[day1][1])
        rainy_tomorrow += day1
        rainy_second += day2
    return rainy_tomorrow / trials, rainy_second / trials


def print_matrix(title, states, matrix):
    print(title)
    print("  ordem:", ", ".join(states))
    for row in matrix:
        print("  [" + ", ".join(f"{p:.2f}" for p in row) + "]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20_000,
                        help="trajetórias para simular; 0 pula a simulação")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trials < 0:
        parser.error("--trials deve ser >= 0")

    print("1. CADEIA DO CLIMA")
    print_matrix("P:", WEATHER_STATES, WEATHER)
    tomorrow = step((1.0, 0.0), WEATHER)
    in_two = step(tomorrow, WEATHER)
    rain_stationary = WEATHER[0][1] / (WEATHER[0][1] + WEATHER[1][0])
    stationary = (1 - rain_stationary, rain_stationary)
    assert all(isclose(a, b) for a, b in zip(step(stationary, WEATHER), stationary))
    print(f"Partindo de SOL: chuva amanhã = {tomorrow[1]:.2%}; em dois dias = {in_two[1]:.2%}.")
    print(f"Probabilidade estacionária de chuva: {rain_stationary:.2%} (1/3).")
    if args.trials:
        sim1, sim2 = simulate_weather(args.trials, args.seed)
        print(f"Simulação ({args.trials} trajetórias, semente {args.seed}): {sim1:.2%} e {sim2:.2%}.")

    print("\n2. MODELO COM MEMÓRIA DE DOIS DIAS")
    for earlier in WEATHER_STATES:
        p = SECOND_ORDER_RAIN[(earlier, "CHUVA")]
        print(f"Ontem {earlier}, hoje CHUVA: chuva amanhã = {p:.2%}.")
    pairs, paired = pair_matrix()
    print_matrix("Matriz nos pares:", tuple(f"{a}/{b}" for a, b in pairs), paired)

    print("\n3. MDP DE NAVEGAÇÃO")
    for gamma in (0.9, 0.0):
        values, policy, scores, n = value_iteration(gamma)
        q = scores["partida"]
        print(f"gamma = {gamma:.1f}; atualizações síncronas = {n}")
        print(f"  Q(partida, atalho) = {q['atalho']:.2f}; Q(partida, desvio) = {q['desvio']:.2f}.")
        print(f"  V(partida) = {values['partida']:.2f}; V(corredor) = {values['corredor']:.2f}.")
        print("  política:", ", ".join(f"{state} → {action}" for state, action in policy.items()))
        print_matrix("  cadeia induzida (sem recompensas):", STATES, policy_matrix(policy))


if __name__ == "__main__":
    main()

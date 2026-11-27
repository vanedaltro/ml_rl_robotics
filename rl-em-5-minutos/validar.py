"""Valida testes e demonstrações da série RL em 5 minutos."""
from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
SMOKE = {
    "01-introducao": ("grid.py",),
    "02-bandits": (
        "bandits.py", "--rodadas", "100", "--repeticoes", "20",
        "--seed", "42", "--csv", "{tmp}/bandits.csv",
    ),
    "03-mdp": ("mdp.py",),
    "04-politica-e-valor": ("valores.py",),
    "05-programacao-dinamica": ("programacao_dinamica.py",),
    "06-monte-carlo-e-td": ("previsao.py",),
    "07-q-learning-e-sarsa": ("cliff.py", "--episodios", "500", "--repeticoes", "12", "--seed", "42"),
    "08-dqn": ("dqn.py", "--episodios", "250", "--seed", "42"),
    "09-policy-gradient": ("reinforce.py", "--episodios", "400", "--seed", "7"),
    "10-actor-critic": ("actor_critic.py", "--episodios", "300", "--seed", "7"),
    "11-trpo-ppo": ("ppo.py", "--atualizacoes", "16", "--seed", "7"),
    "12-sim2real": ("sim2real.py",),
    "13-model-based": ("model_based.py",),
    "14-marl": ("marl.py", "--sementes", "4", "--episodios", "200", "--seed", "42"),
    "15-locomocao": ("locomocao.py",),
}


def run(folder, args, timeout):
    print(f"$ {Path(sys.executable).name} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=folder, check=True, timeout=timeout)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("capitulos", nargs="*", metavar="CAPÍTULO", help="prefixos como 03 ou 08")
    parser.add_argument("--timeout", type=int, default=120, help="segundos máximos por comando")
    parser.add_argument("--mujoco", action="store_true", help="roda também Hopper-v5 se o capítulo 15 estiver selecionado")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout precisa ser positivo")
    ids = {name.split("-", 1)[0]: name for name in SMOKE}
    invalid = set(args.capitulos) - ids.keys()
    if invalid:
        parser.error("capítulos desconhecidos: " + ", ".join(sorted(invalid)))
    selected = [ids[i] for i in sorted(set(args.capitulos))] if args.capitulos else list(SMOKE)
    failed = []
    with TemporaryDirectory(prefix="rl-em-5-minutos-") as tmp:
        for name in selected:
            print(f"\n== {name} ==", flush=True)
            folder = ROOT / name
            try:
                # O primeiro capítulo tem 8 verificações dentro do próprio grid.py.
                if name != "01-introducao":
                    run(folder, ("-m", "unittest", "discover", "-v"), args.timeout)
                run(folder, tuple(arg.replace("{tmp}", tmp) for arg in SMOKE[name]), args.timeout)
                if name == "15-locomocao" and args.mujoco:
                    run(folder, ("hopper.py",), args.timeout)
            except (subprocess.SubprocessError, OSError) as error:
                print(f"Falha em {name}: {error}", file=sys.stderr, flush=True)
                failed.append(name)
    print(f"\nValidação: {len(selected) - len(failed)}/{len(selected)} capítulos passaram.")
    if failed:
        print("Falharam: " + ", ".join(failed), file=sys.stderr)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())

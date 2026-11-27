"""Gera as figuras a partir das métricas e do exemplo do artigo."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
NAVY, BG, SOFT = "#142c44", "#f7f4ed", "#526679"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                     "text.color": NAVY, "axes.labelcolor": NAVY,
                     "font.family": "DejaVu Sans", "font.size": 11})


def save(fig, name):
    fig.savefig(ROOT / name, dpi=140, facecolor=BG)
    plt.close(fig)


def cover():
    fig, ax = plt.subplots(figsize=(11, 6.1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.02, .97, "RL EM 5 MINUTOS   /   EPISÓDIO 02", color="#92621f",
            weight="bold", va="top", fontsize=12)
    ax.text(.02, .87, "Explorar ou aproveitar?", fontsize=37, weight="bold", va="top")
    ax.text(.02, .73, "Uma puxada ruim pode esconder a melhor alavanca.", fontsize=16, color=SOFT)
    for i, (label, mean, first) in enumerate(zip("ABC", (4, 6, 3), (4, 2, 3))):
        x = .02 + i * .33
        sel = label == "B"
        patch = FancyBboxPatch((x, .22), .30, .43, boxstyle="round,pad=0.008,rounding_size=0.02",
                               facecolor="#fff0d8" if sel else "#e6eff5",
                               edgecolor="#deab65" if sel else "#b2c7d6", linewidth=1.5)
        ax.add_patch(patch)
        ax.text(x+.025,.58,f"ALAVANCA {label}",size=14,weight="bold")
        ax.text(x+.025,.45,f"{first} pontos",size=24,weight="bold")
        ax.text(x+.025,.392,"na primeira puxada",size=11,color=SOFT)
        ax.text(x+.025,.288,f"Média real: {mean}",size=15,color="#815314" if sel else NAVY)
    ax.text(.02, .11, "O agente vê cada recompensa, mas não conhece as médias reais.", size=13, color=SOFT)
    save(fig, "01-alavancas.png")


def charts(source):
    data = defaultdict(list)
    with source.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["politica"]].append(row)
    opts = {"guloso": ("Guloso", "#486881"), "eps-010": ("ε = 0,10", "#bc562e"),
            "eps-001": ("ε = 0,01", "#77609f"), "ucb": ("UCB (c = 3)", "#197962"),
            "oraculo": ("Oráculo", "#75828c")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), layout="constrained")
    fig.suptitle("Quanto custa insistir cedo demais?", fontsize=23, fontweight="bold")
    for k, vals in data.items():
        label, color = opts[k]
        rounds = [int(r["rodada"]) for r in vals]
        for ax, metric in zip(axes, ("recompensa_media", "pseudo_regret")):
            ax.plot(rounds, [float(r[metric]) for r in vals], color=color,
                    lw=2.2, ls="--" if k == "oraculo" else "-", label=label)
    axes[0].set(xlabel="Rodadas", ylabel="Pontos por rodada", title="Recompensa média acumulada", xlim=(20, 1000), ylim=(5.0, 6.15))
    axes[1].set(xlabel="Rodadas", ylabel="Pontos esperados perdidos", title="Pseudo-regret acumulado")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].legend(loc="lower right", frameon=False, fontsize=10)
    axes[1].text(.03, .92, "500 repetições · semente 42", transform=axes[1].transAxes, color=SOFT)
    save(fig, "02-resultados.png")


if __name__ == "__main__":
    cover()
    charts(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "resultados.csv")

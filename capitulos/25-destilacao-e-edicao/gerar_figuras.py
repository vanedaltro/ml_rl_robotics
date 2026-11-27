from pathlib import Path
import json
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("figuras")
OUT.mkdir(exist_ok=True)
PAPER, INK, DIM = "#f8f3e8", "#172c2b", "#475b56"
GRN, RST, BLU, OCH = "#166456", "#a74c37", "#315675", "#885f15"

def base(num, title, sub, h=6.75):
    fig = plt.figure(figsize=(12, h), facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    ax.text(5, 93, "SÉRIE ML  /  CAPÍTULO 25     •     " + num, fontsize=12,
            weight="bold", family="monospace", color=GRN)
    ax.text(5, 83, title, fontsize=30, weight="bold", family="serif", color=INK)
    ax.text(5, 75, sub, fontsize=15, color=DIM)
    return fig, ax

def save(fig, name):
    fig.savefig(OUT / name, dpi=133, facecolor=PAPER)
    plt.close(fig)

def box(ax, x, y, w, h, color, label, detail=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.2,rounding_size=1.4",
                                facecolor="white", edgecolor=color, linewidth=1.4))
    label_args = dict(fontsize=18 if w < 27 else 22, color=color, weight="bold", va="center")
    ax.text(x+2, y+h*(.65 if detail else .51), label, **label_args)
    if detail:
        ax.text(x+2, y+h*.30, detail, fontsize=13, color=INK, va="center")

def arrow(ax, x1, y1, x2, y2, color=DIM):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle="-|>", mutation_scale=22, linewidth=1.6, color=color))

fig, ax = base("MAPA", "Quatro caminhos para melhorar um modelo",
               "O que muda: o sinal de treino, o fluxo de produção ou o próprio checkpoint?")
box(ax, 5, 45, 43, 20, GRN, "01  Destilação", "Professor ensina → aluno aprende")
box(ax, 52, 45, 43, 20, BLU, "02  Verificação", "Respostas → regra → treino")
box(ax, 5, 17, 43, 20, RST, "03  Roteamento", "Caso difícil → modelo mais forte")
box(ax, 52, 17, 43, 20, OCH, "04  Edição de pesos", "Hipótese → patch → validação")
ax.text(5, 7, "O modelo grande pode ser professor, reforço em produção ou editor. Cada papel tem um custo.", fontsize=12, color=DIM)
save(fig, "01-mapa.png")

res = json.loads(Path("resultados.json").read_text())["metricas"]
fig, ax = base("LABORATÓRIO", "O que mudou com o professor?",
               "Os três alunos têm 710 parâmetros e treinam nas mesmas 48 entradas.")
chart = fig.add_axes((.26, .20, .63, .50), facecolor=PAPER)
names = ["Só rótulos", "Rótulos + classe do professor", "Rótulos + distribuição", "Professor (referência)"]
keys = ["labels", "hard", "soft", "professor"]
for i, (key, color) in enumerate(zip(keys, (BLU, "#467e79", GRN, OCH))):
    d = res[key]
    m, s = d["media_pct"], d["dp_amostral_pp"]
    chart.barh(i, m, xerr=s, height=.58, color=color,
               error_kw={"ecolor": INK, "capsize": 4, "elinewidth": 1.2})
    chart.text(m+s+1.5, i, f"{m:.1f}%".replace(".", ","), va="center",
               color=INK, weight="bold", fontsize=14)
chart.set(xlim=(0, 100), yticks=range(4), yticklabels=names,
          xticks=[0, 25, 50, 75, 100], xticklabels=["0%", "25%", "50%", "75%", "100%"])
chart.invert_yaxis()
chart.tick_params(axis="both", colors=INK, labelsize=11, length=0, pad=9)
chart.grid(axis="x", color="#d8d7c9")
chart.set_axisbelow(True)
for spine in chart.spines.values():
    spine.set_visible(False)
ax.text(5, 9, "Acurácia média no teste • 3 sementes • Barras finas = desvio-padrão amostral, não IC", fontsize=12, color=DIM)
ax.text(5, 5, "CPU, PyTorch 2.9.1 • Dados sintéticos • O professor treinou com 4.096 exemplos.", fontsize=12, color=DIM)
save(fig, "02-lab.png")

fig, ax = base("CHECKPOINTS", "Editar pesos sem perder o original",
               "A edição é aplicada numa cópia. O modelo original segue como linha de base.")
box(ax, 5, 44, 25, 20, GRN, "Original", "Checkpoint imutável")
box(ax, 38, 44, 24, 20, OCH, "Cópia + patch", "Mudança no tensor")
box(ax, 70, 44, 25, 20, BLU, "Avaliar", "Teste + regressões")
arrow(ax, 31, 54, 37, 54)
arrow(ax, 63, 54, 69, 54)
arrow(ax, 82, 43, 82, 34)
box(ax, 54, 13, 41, 19, RST, "Manter ou descartar", "Só aceite se os critérios passarem")
ax.text(5, 23, "Exemplo do lab", fontsize=16, color=INK, weight="bold")
ax.text(5, 16, r"$x_1+3x_2\quad\rightarrow\quad x_1+0x_2$", fontsize=22, color=GRN)
ax.text(5, 5, "Demonstração algébrica no classificador linear; não é uma edição de LLM reproduzida.", fontsize=12, color=DIM)
save(fig, "03-edicao.png")

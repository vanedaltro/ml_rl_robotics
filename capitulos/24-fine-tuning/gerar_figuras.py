import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

here = Path(__file__).resolve().parent
out = here / "figuras"
out.mkdir(exist_ok=True)
runs = json.loads((here / "resultados.json").read_text())["runs"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
bg, fg, mint, blue, gray = "#101a32", "#f0f6ff", "#5de6b3", "#91b8ff", "#c5d2e7"

def save(fig, name):
    fig.savefig(out / (name + ".png"), dpi=155, facecolor=bg)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 6.7), facecolor=bg)
fig.subplots_adjust(left=.24, right=.93, top=.72, bottom=.15)
ax.set_facecolor(bg)
fig.text(.075, .91, "CAPÍTULO 24   /   LABORATÓRIO PYTORCH", color=mint, fontweight="bold")
fig.text(.075, .825, "Aprender B pode derrubar o desempenho em A", color=fg, fontsize=24, fontweight="bold")
labels = ("Base antes de B", "Ajuste completo", "Replay 50/50", "Adapter sempre ligado", "Adapter com roteamento")
keys = ("base", "full", "replay", "adapter_on", "adapter_routed")
for i, (name, color) in enumerate((("Tarefa A · anterior", blue), ("Tarefa B · nova", mint))):
    values = [mean(r[k][i] for r in runs) * 100 for k in keys]
    yy = [y + (i - .5) * .32 for y in range(len(labels))]
    ax.barh(yy, values, height=.27, color=color, label=name)
    for y, value in zip(yy, values):
        ax.text(value + 1, y, f"{value:.1f}".replace(".", ",")+"%", color=fg, va="center", fontsize=10)
ax.set(yticks=list(range(len(labels))), yticklabels=labels, xlim=(0, 110), xticks=(0,20,40,60,80,100))
ax.invert_yaxis()
ax.tick_params(colors=gray, length=0)
ax.grid(axis="x", color="#30425d", alpha=.65)
ax.set_axisbelow(True)
for sp in ax.spines.values(): sp.set_visible(False)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, facecolor=bg, edgecolor=bg, labelcolor=fg)
fig.text(.075,.055,"A e B têm rótulos conflitantes. Roteamento recebe a tarefa correta. Médias de 3 sementes em CPU.",color=gray)
save(fig,"01-resultados")

fig, ax = plt.subplots(figsize=(12, 6.7), facecolor=bg)
ax.set(xlim=(0,12),ylim=(0,6.7)); ax.axis("off")
fig.subplots_adjust(0,0,1,1)
ax.text(.7,6.05,"CAPÍTULO 24   /   ADAPTERS",color=mint,weight="bold")
ax.text(.7,5.38,"A base fica intacta. O comportamento pode mudar.",color=fg,fontsize=22,weight="bold")
for y,col,heading,line in [(3.45,blue,"Tarefa A | desliga o adapter","Entrada → base congelada → cabeça → saída original"),
                          (1.8,mint,"Tarefa B | liga o adapter","Entrada → base + correção do adapter → cabeça → saída adaptada")]:
    ax.add_patch(FancyBboxPatch((.7,y),10.55,1.15,boxstyle="round,pad=.12",facecolor="#192945",edgecolor=col))
    ax.text(.98,y+.74,heading,color=col,fontsize=16,weight="bold")
    ax.text(.98,y+.32,line,color=fg,fontsize=13)
ax.text(.7,.83,"Desligar restaura o caminho. Preservar A com o adapter ligado é outro teste.",color=gray,fontsize=12)
save(fig,"02-adapter-roteamento")

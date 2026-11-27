from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).with_name("figuras")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
bg, white = "#0b1529", "#f4f8ff"
sky, mint, dim = "#8bc7ff", "#82f1c5", "#b4c5dc"

def setup(label, title, desc):
    fig, ax = plt.subplots(figsize=(11.2, 6.3), facecolor=bg)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0,1120), ylim=(0,630), facecolor=bg)
    ax.axis("off")
    ax.text(60, 575, label, color=mint, fontsize=12, fontweight="bold")
    ax.text(60, 523, title, color=white, fontsize=28, fontweight="bold")
    ax.text(61, 485, desc, color=dim, fontsize=13)
    return fig, ax

def card(ax, x, y, w, h, accent, title, lines, fill="#142743"):
    box = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=17",fc=fill,ec=accent,lw=1.3)
    ax.add_patch(box)
    ax.text(x+22, y+h-36, title, color=accent, fontsize=16, fontweight="bold")
    ax.text(x+22, y+h-66, lines, color=white, fontsize=12, va="top", linespacing=1.55)

def arrow(ax, a, b, color):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,color=color,lw=1.7))

def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, facecolor=bg)
    plt.close(fig)

fig, ax = setup("PYTORCH  /  TREINO DISTRIBUÍDO", "Uma rede neural. Dois nós.",
                "Cada processo vê dados diferentes; os dois atualizam o mesmo modelo.")
card(ax, 60, 256, 338, 176, sky, "MÁQUINA A · RANK 0",
     "Lote local: 32 exemplos\nForward + backward\nGradiente local de exemplo: 2")
card(ax, 722, 256, 338, 176, sky, "MÁQUINA B · RANK 1",
     "Lote local: 32 exemplos\nForward + backward\nGradiente local de exemplo: 6")
card(ax, 441, 281, 238, 126, mint, "ALL-REDUCE", "Média: (2 + 6) ÷ 2 = 4", fill="#153534")
arrow(ax,(399,357),(440,357),sky); arrow(ax,(721,357),(680,357),sky)
arrow(ax,(440,303),(399,303),mint); arrow(ax,(680,303),(721,303),mint)
arrow(ax,(229,253),(229,216),mint); arrow(ax,(891,253),(891,216),mint)
card(ax, 60, 108, 338, 104, mint, "A ATUALIZA A RÉPLICA", "optimizer.step() usa gradiente 4")
card(ax, 722, 108, 338, 104, mint, "B ATUALIZA A RÉPLICA", "optimizer.step() usa gradiente 4")
ax.text(60, 50, "64 EXEMPLOS POR ATUALIZAÇÃO  ·  1 PROCESSO POR NÓ  ·  LOTES DO MESMO TAMANHO",
        color=dim, fontsize=11, fontweight="bold")
save(fig,"01-ddp")

fig, ax = setup("PYTORCH  /  VALIDAÇÃO", "Três testes, três respostas",
                "Separar correção, rede e desempenho evita confundir um treino que roda com um treino que escala.")
card(ax, 60, 218, 306, 200, sky, "01  CORREÇÃO",
     "Dois ranks no mesmo host.\nAs réplicas concordam?\nA atualização bate com\num lote global equivalente?")
card(ax, 406, 218, 306, 200, mint, "02  CONECTIVIDADE",
     "Duas máquinas na LAN.\nO ponto de encontro funciona?\nOs ranks trocam gradientes?\nO rank 0 grava o checkpoint?")
card(ax, 752, 218, 308, 200, sky, "03  DESEMPENHO",
     "Use o mesmo lote global.\nMeça exemplos por segundo\napós aquecimento. Compare\no tempo do rank mais lento.")
arrow(ax,(367,309),(404,309),mint); arrow(ax,(713,309),(751,309),mint)
ax.text(64, 135, "PONTO DE PARTIDA", color=mint, fontsize=13, fontweight="bold")
ax.text(64, 100, "1 rank × 64 exemplos  |  2 ranks × 32 exemplos",
        color=white, fontsize=19, fontweight="bold")
ax.text(64, 54, "A rede pode custar mais tempo do que a segunda máquina economiza. Meça antes de concluir.",
        color=dim, fontsize=12)
save(fig,"02-validacao")

fig, ax = setup("CAPÍTULO 23 / 03", "O que cada GPU faz?", "Réplicas processam dados diferentes. Partições cooperam no mesmo modelo.")
card(ax, 60, 275, 480, 165, sky, "DDP · dados", "Modelo inteiro em cada GPU\nLotes distintos → gradientes sincronizados")
card(ax, 580, 275, 480, 165, mint, "FSDP2 · dados + estado", "Parâmetros, gradientes e otimizador em partes\nAll-gather no cálculo; reduce-scatter no gradiente")
card(ax, 60, 75, 480, 165, sky, "Tensor · camada", "GPU A: parte da multiplicação\nGPU B: outra parte → resultado combinado")
card(ax, 580, 75, 480, 165, mint, "Pipeline · estágios", "GPU A: primeiras camadas\nGPU B: últimas camadas; fluem ativações")
save(fig, "03-paralelismo")
fig, ax = setup("CAPÍTULO 23 / 04", "Duas formas de servir um modelo", "É diferente distribuir requisições ou dividir uma única previsão.")
card(ax, 60, 230, 480, 205, sky, "Réplicas independentes", "Cliente → balanceador\n               ↳ Nó A: modelo completo\n               ↳ Nó B: modelo completo\nUma réplica resolve a requisição.")
card(ax, 580, 230, 480, 205, mint, "Modelo em estágios", "Cliente → Nó A: primeiras camadas\n               → Nó B: últimas camadas\n               → resposta\nOs dois nós são necessários.")
ax.text(60, 145, "Medição: vazão entregue, p95/p99, erros e comportamento em falhas.", color=white, fontsize=15)
save(fig, "04-inferencia")

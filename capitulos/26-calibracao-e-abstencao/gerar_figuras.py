from pathlib import Path
import json
import math
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

raw = json.loads(Path("resultados.json").read_text())
runs = list(raw["runs"].values())
OUT = Path("figuras")
OUT.mkdir(exist_ok=True)
INK, PAPER, DIM, RULE = "#28332f", "#f6f2e7", "#586963", "#d8d9cb"
BLUE, ORANGE, GREEN = "#3f5c80", "#a34f33", "#216d56"

def figure(section, title, sub):
    fig = plt.figure(figsize=(12, 6.75), facecolor=PAPER)
    fig.text(.06, .92, f'SÉRIE ML     /     CAPÍTULO 26     /     {section}',
             color=GREEN, weight="bold", family="monospace", fontsize=12)
    fig.text(.06, .83, title, color=INK, weight="bold", family="serif", fontsize=29)
    fig.text(.06, .76, sub, color=DIM, fontsize=13)
    return fig

def save(fig, name):
    fig.savefig(OUT / name, dpi=138, facecolor=PAPER)
    plt.close(fig)

def foot(fig, a, b):
    fig.text(.06,.085,a,color=DIM,fontsize=11)
    fig.text(.06,.045,b,color=DIM,fontsize=11)

def chart(fig, rect):
    ax=fig.add_axes(rect,facecolor=PAPER)
    for spine in ax.spines.values():
        spine.set_color(RULE)
    ax.tick_params(colors=INK,labelsize=10)
    ax.grid(color=RULE,alpha=.7)
    ax.set_axisbelow(True)
    return ax

example = runs[0]["illustrative_example"]
fig=figure("EXPOSIÇÃO","Quando o modelo deve dizer “não sei”?",
           "Mesma imagem, três exposições. O rótulo verdadeiro continua sendo diagonal.")
for i, (light, frame) in enumerate(example["frames"].items()):
    ax=fig.add_axes((.08 + i*.30,.25,.24,.43))
    ax.imshow(frame, cmap="gray",vmin=0,vmax=1,interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(light.capitalize(), color=INK,weight="bold",fontsize=15,labelpad=9)
foot(fig,"No escuro: a CNN escolheu vertical com confiança arredondada de 100%.",
         "Exemplo selecionado entre os erros de maior confiança da semente 7. Escala de cinza fixa.")
save(fig,"01-exposicao.png")

fig=figure("CALIBRAÇÃO","A confiança subiu. O acerto não.",
           "Na mudança de iluminação, ajustar a temperatura só no claro piorou a calibração.")
ax=chart(fig,(.12,.20,.78,.47))
lights=["claro","penumbra","escuro"]
colors=[GREEN,BLUE,ORANGE]
items=[("Acurácia", "accuracy", "raw", "#3d5651"),
       ("Confiança antes","mean_confidence","raw",BLUE),
       ("Confiança depois","mean_confidence","calibrated",ORANGE)]
for j,(label,key,stage,color) in enumerate(items):
    ys=[100*sum(r["conditions"][l][stage][key] for r in runs)/len(runs) for l in lights]
    xs=[i+(j-1)*.22 for i in range(3)]
    ax.bar(xs,ys,width=.20,color=color,label=label,zorder=3)
    for x,y in zip(xs,ys):
        ax.text(x,y+2,str(round(y,1)).replace(".",",")+"%",ha="center",color=INK,fontsize=10,weight="bold")
ax.set(ylim=(0,116),xticks=range(3),xticklabels=["Claro","Penumbra","Escuro"],yticks=[0,25,50,75,100],
       yticklabels=["0%","25%","50%","75%","100%"])
ax.legend(loc="upper center",bbox_to_anchor=(.5,1.13),ncol=3,frameon=False,labelcolor=INK)
foot(fig,"Média em três sementes, com 1.024 imagens de teste por semente e condição.",
         "Acurácia = acertos. Confiança = maior probabilidade softmax; são medidas diferentes.")
save(fig,"02-confianca.png")

fig=figure("ABSTENÇÃO","O limiar compra menos erros?",
           "Ele deixa de aceitar alguns erros, mas também pode recusar previsões corretas.")
ax=chart(fig,(.11,.23,.80,.47))
handles=[]
for l,color in zip(lights,colors):
    pts=[]
    for col in zip(*(r["conditions"][l]["curve"] for r in runs)):
        n=sum(p["n"] for p in col); a=sum(p["accepted"] for p in col)
        if a:
            pts.append((100*a/n,100*sum(p["accepted_errors"] for p in col)/a))
    ax.plot(*zip(*pts), color=color,label=l.capitalize(),marker=".",markersize=3)
    sides=[r["conditions"][l]["policy"] for r in runs]
    n=sum(p["n"] for p in sides); a=sum(p["accepted"] for p in sides)
    if a:
        ax.plot(100*a/n,100*sum(p["accepted_errors"] for p in sides)/a,marker="D",mec=INK,
                color=color,markersize=9,linestyle="none",zorder=5)
ax.axhline(5,ls="--",color=DIM,linewidth=1.3, label="Meta de validação: 5%")
ax.set(xlim=(-2,103),ylim=(-3,75),xlabel="Cobertura (% de casos aceitos)",
       ylabel="Risco (% de erros entre aceitos)")
ax.legend(loc="upper left",ncol=2,frameon=False,labelcolor=INK)
foot(fig,"Linhas: limiares comuns explorados no teste. Losangos: só o limiar escolhido na validação de cada semente.",
         "Risco indefinido quando nada é aceito. No escuro, o limiar validado deixou passar 1.731 erros.")
save(fig,"03-risco-cobertura.png")

fig=figure("DECISÃO","Agir, pedir ajuda ou parar",
           "Exemplo de fluxo para um robô que detecta marcas visuais. Não é uma certificação de segurança.")
ax=fig.add_axes((0,0,1,1)); ax.set(xlim=(0,100),ylim=(0,100)); ax.axis("off")
def box(x,y,w,h,color,label,detail):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.2,rounding_size=1.2",
                                facecolor="white",edgecolor=color,linewidth=1.3))
    ax.text(x+2,y+h*.64,label,color=color,fontsize=16,weight="bold",va="center")
    ax.text(x+2,y+h*.28,detail,color=INK,fontsize=10,va="center")
def arr(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=16,color=DIM,linewidth=1.5))
box(6,53,88,13,BLUE,"A câmera produziu uma imagem utilizável?",
    "Antes de classificar: verifique o sensor, a iluminação e as condições conhecidas do ambiente.")
for x in (20,50,80):
    arr((x,52),(x,40))
box(6,23,28,16,GREEN,"AGIR","Imagem válida + risco aceitável.")
box(36,23,28,16,BLUE,"PEDIR AJUDA","Nova imagem ou revisão.")
box(66,23,28,16,ORANGE,"PARAR","Falha crítica ou risco não tolerado.")
foot(fig,"Softmax alto não substitui regras independentes de integridade e segurança.",
         "Defina a ação de contingência antes do deploy e monitore por condição, não só pela média.")
save(fig,"04-decisoes.png")

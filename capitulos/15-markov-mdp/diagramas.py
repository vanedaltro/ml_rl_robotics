"""Gera as quatro figuras. Requer Matplotlib apenas para renderizar."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parent / "figuras"
OUT.mkdir(exist_ok=True)
BG="#132221"; PANEL="#1b3331"; INK="#f9f3e5"; MUTE="#b5d1ca"
MINT="#91ebcc"; GOLD="#f5c47d"; PINK="#f3a695"; GRID="#42615a"
plt.rcParams.update({"font.family":"DejaVu Sans", "mathtext.fontset":"dejavusans",
                     "text.color":INK, "axes.unicode_minus":True})

def canvas(i,title,subtitle):
    fig=plt.figure(figsize=(14,8),dpi=115,facecolor=BG)
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,14);ax.set_ylim(0,8);ax.axis("off")
    lab(ax,.64,7.48,f"SÉRIE ML + ROBÓTICA   /   CAP. 15   /   {i:02d}",13,MINT,bold=True)
    lab(ax,.64,6.83,title,31,INK,bold=True)
    lab(ax,.64,6.27,subtitle,15,MUTE)
    return fig,ax

def lab(ax,x,y,s,fs=15,c=INK,align="left",bold=False):
    ax.text(x,y,s,color=c,fontsize=fs,ha=align,va="center",
            weight="bold" if bold else "normal",linespacing=1.35)

def box(ax,x,y,w,h,s=None,edge=GRID,fc=PANEL,fs=17):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.16",
                               linewidth=1.5,edgecolor=edge,facecolor=fc))
    if s: lab(ax,x+w/2,y+h/2,s,fs,INK,"center",True)

def arrow(ax, x1,y1,x2,y2,c=MINT,rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
                 mutation_scale=20,color=c,lw=2,connectionstyle=f"arc3,rad={rad}"))

def save(fig,fn):
    fig.savefig(OUT / fn,facecolor=BG,dpi=115)
    plt.close(fig)

f,a=canvas(1,"Prever ou decidir?","A diferença é quem escolhe a ação e como o ambiente responde.")
box(a,.65,3.81,12.7,1.87); box(a,.65,1.37,12.7,1.87)
lab(a,1.0,5.2,"CADEIA DE MARKOV",14,MINT,bold=True)
lab(a,1.0,2.76,"PROCESSO DE DECISÃO DE MARKOV",14,GOLD,bold=True)
box(a,1.02,4.07,3.12,.74,r"Estado $S_t$",edge=MINT,fs=17)
box(a,9.82,4.07,3.17,.74,r"Estado $S_{t+1}$",edge=MINT,fs=17)
arrow(a,4.35,4.44,9.58,4.44)
lab(a,7.0,4.71,"transição probabilística",12,MUTE,"center")
box(a,1.02,1.62,3.55,.74,r"Estado $S_t$ + ação $A_t$",edge=GOLD,fs=16)
box(a,9.47,1.62,3.52,.74,r"$S_{t+1}$ + recompensa $R_{t+1}$",edge=GOLD,fs=15)
arrow(a,4.78,1.99,9.22,1.99,GOLD)
lab(a,7.0,2.26,"transição condicionada à ação",12,MUTE,"center")
lab(a,.69,.66,"Markov: dado o estado relevante (e a ação, no MDP), não consultamos o restante da história.",14,MUTE)
save(f,"01-cadeia-vs-mdp.png")

f,a=canvas(2,"De 20% a 28% de chance de chuva","Probabilidades fictícias. Começamos com 100% de chance de sol hoje.")
box(a,.67,1.22,5.36,4.4)
lab(a,1,5.19,"ESTADO ATUAL → PRÓXIMO",13,MINT,bold=True)
box(a,1.1,3.22,1.35,.78,"Sol",GOLD,fs=18)
box(a,4.24,3.22,1.35,.78,"Chuva",MINT,fs=18)
arrow(a,2.48,3.79,4.19,3.79,GOLD,-.35)
arrow(a,4.19,3.42,2.48,3.42,MINT,-.35)
lab(a,3.34,4.18,"20%",14,GOLD,"center",True)
lab(a,3.34,3.06,"40%",14,MINT,"center",True)
lab(a,1.77,2.48,"fica: 80%",13,GOLD,"center")
lab(a,4.91,2.48,"fica: 60%",13,MINT,"center")
lab(a,1,1.57,r"$P(C_2)=0.8(0.2)+0.2(0.6)=0.28$",16,INK)
g=f.add_axes([.49,.22,.43,.43],facecolor=BG)
g.plot(range(9),[100/3*(1-.4**t) for t in range(9)],"-o",color=MINT,lw=3,ms=7)
g.axhline(100/3,color=GOLD,ls="--",lw=1.7)
g.annotate("28%",(2,28),xytext=(2.5,22),color=INK,fontsize=14,
           arrowprops={"arrowstyle":"->","color":MINT})
g.text(8,34.6,"33⅓% estacionária",ha="right",va="bottom",color=GOLD,fontsize=12)
g.set(xlim=(-.2,8.2),ylim=(-2,40),xlabel="Dias após hoje",ylabel="P(chuva) em %")
g.xaxis.label.set_color(MUTE);g.yaxis.label.set_color(MUTE)
g.tick_params(colors=MUTE,labelsize=12)
for spine in g.spines.values(): spine.set_color(GRID)
g.grid(color=GRID,lw=.6,alpha=.55)
lab(a,.69,.66,"28% no segundo dia. Pelo menos um dia de chuva? Aí a pergunta vale 36%.",14,MUTE)
save(f,"02-clima.png")

f,a=canvas(3,"Quando a história muda a previsão","Mesma observação hoje. Duas probabilidades diferentes para amanhã.")
lab(a,1.0,5.46,"ONTEM",13,MUTE,bold=True);lab(a,5.1,5.46,"HOJE",13,MUTE,bold=True)
lab(a,9.14,5.46,"AMANHÃ: P(CHUVA)",13,MUTE,bold=True)
for y,ant,p,c in [(4.36,"Sol","40%",GOLD),(3.05,"Chuva","80%",MINT)]:
    box(a,1,y,2.93,.83,ant,c)
    box(a,5.1,y,2.93,.83,"Chuva",MINT)
    box(a,9.14,y,3.86,.83,p,c)
    arrow(a,4.14,y+.41,4.91,y+.41,c)
    arrow(a,8.23,y+.41,8.96,y+.41,c)
box(a,1,1.2,12,1.22)
lab(a,1.31,1.81,"Estado composto:",16,MINT,bold=True)
lab(a,4.15,1.81,r"$Z_t=(S_{t-1},S_t)$",23)
lab(a,8.5,1.81,r"$(a,b)\ \longrightarrow\ (b,c)$",23)
lab(a,.69,.66,"Ordem 2 na observação original. Primeira ordem sobre o par que desliza.",14,MUTE)
save(f,"03-estado-composto.png")

f,a=canvas(4,"Bellman: o atalho vale a pena?","A ação define as possibilidades; o ambiente sorteia o resultado.")
box(a,.82,3.16,2.62,.94,"Partida")
box(a,5.95,2.22,2.65,.94,"Corredor",MINT)
box(a,10.53,4.4,2.63,.94,"Meta",MINT)
box(a,10.53,3.02,2.63,.94,"Falha",PINK)
arrow(a,3.55,3.99,10.41,4.86,GOLD,-.1)
arrow(a,3.56,3.68,10.41,3.5,PINK,0.02)
arrow(a,3.55,3.22,5.83,2.68,MINT,0.11)
arrow(a,8.72,2.72,10.44,4.68,MINT,.17)
lab(a,6.56,5.15,"Atalho: 70% / +10",15,GOLD,"center",True)
lab(a,6.7,3.94,"Atalho: 30% / −20",15,PINK,"center",True)
lab(a,3.85,2.5,"Desvio: 100% / −1",13,MINT)
lab(a,8.96,2.52,"Avançar: 100% / +8",13,MINT)
box(a,.82,.73,12.32,.96)
lab(a,1.05,1.22,r"$V^*(s)=\max_a\sum_{s^\prime}P(s^\prime|s,a)[r(s,a,s^\prime)+\gamma V^*(s^\prime)]$",21,INK)
lab(a,.84,.36,"Com γ = 0,9  •  desvio 6,2  >  atalho 1,0",14,MINT,bold=True)
save(f,"04-bellman-robo.png")

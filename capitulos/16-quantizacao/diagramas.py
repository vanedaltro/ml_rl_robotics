"""Quatro gráficos para o capítulo. Matplotlib é necessário só aqui."""
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=Path(__file__).resolve().parent
OUT=HERE/"figuras"; OUT.mkdir(exist_ok=True)
BG="#16172b"; PANEL="#23243f"; FG="#f8f2ff"; INK="#c7c3de"
MINT="#82e6cb"; PINK="#f38ab4"; GOLD="#ffc680"; LILA="#bbabff"; GRID="#494568"
plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans","text.color":FG})

def t(a,x,y,s,n=16,c=FG,ha="left",b=False):
    a.text(x,y,s,ha=ha,va="center",fontsize=n,color=c,weight="bold" if b else "normal",linespacing=1.46)

def box(a,x,y,w,h,edge=GRID):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.17",facecolor=PANEL,edgecolor=edge,lw=1.6))

def line(a,x,y,u,v,c=LILA):
    a.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle="-|>",mutation_scale=18,color=c,lw=2))

def canvas(i,title,subtitle):
    f=plt.figure(figsize=(14,8),dpi=115,facecolor=BG)
    a=f.add_axes((0,0,1,1));a.set(xlim=(0,14),ylim=(0,8));a.axis("off")
    t(a,.63,7.5,f"SÉRIE ML + ROBÓTICA   /   CAP. 16   /   {i:02d}",14,MINT,b=True)
    t(a,.63,6.83,title,31,FG,b=True)
    t(a,.63,6.28,subtitle,15,INK)
    return f,a

def save(f,n):
    f.savefig(OUT/n,facecolor=BG,dpi=115)
    plt.close(f)

f,a=canvas(1,"Quantização: o que acontece com o número?","Um valor real vira um código. A reconstrução pode ser diferente do original.")
for x,tag,big,sub,c in [(0.66,"01  VALOR REAL","0,73","x em ponto flutuante",LILA),(5.05,"02  CÓDIGO INTEIRO","9","16 códigos possíveis",PINK),(9.44,"03  RECONSTRUÇÃO","0,80","volta como aproximação",MINT)]:
    box(a,x,3.28,3.85,2.37,c)
    t(a,x+.23,5.22,tag,13,c,b=True)
    t(a,x+.23,4.59,big,39,FG,b=True)
    t(a,x+.23,3.77,sub,14,INK)
line(a,4.56,4.45,4.99,4.45)
line(a,8.95,4.45,9.38,4.45)
box(a,.66,1.18,12.63,1.55)
t(a,.99,2.11,r"$q=\mathrm{clip}(\mathrm{round}(x/s)+z,\ q_{\min},q_{\max})$",23)
t(a,9.18,2.11,r"$\hat{x}=s(q-z)$",23,MINT)
t(a,.99,1.56,"Exemplo afim: escala 0,2 • ponto zero 5 • códigos de 0 a 15",13,INK)
t(a,.68,.53,"Arredondamento perde resolução. Clipping satura os valores fora do intervalo.",14,INK)
save(f,"01-quantizacao.png")

f,a=canvas(2,"Três formas de chegar a poucos bits","Qual parte realmente usa baixa precisão?")
rows=[
    (4.27,"PTQ","Treino normal","Escolher escalas","Converter para o destino",LILA),
    (2.62,"QAT","Modelo inicial","Forward simulado + ajuste","Converter para o destino",GOLD),
    (.97,"Ternário","Pesos mestres em float","Forward com −1 / 0 / +1","Backward com STE",MINT),
]
for y,label,l,m,r,c in rows:
    box(a,.66,y,12.62,1.39)
    t(a,.94,y+.93,label,22,c,b=True)
    t(a,3.11,y+.58,l,13,FG)
    t(a,6.46,y+.58,m,13,FG)
    t(a,9.85,y+.58,r,13,FG)
    line(a,5.45,y+.57,6.15,y+.57,c)
    line(a,9.0,y+.57,9.57,y+.57,c)
t(a,.68,.39,"No laboratório, todas as multiplicações de pesos reconstruídos ainda são em float.",13,INK)
save(f,"02-treino.png")

runs=json.loads((HERE/"resultados.json").read_text())["runs"]
archs=["MLP","CNN","Transformer"]
modes=["fp32","int8","int4","ternary"]
values={(a,m):100*mean(r["held_out_accuracy"] for r in runs if r["architecture"]==a and r["mode"]==m) for a in archs for m in modes}
f,a=canvas(3,"Poucos bits, efeitos diferentes","Acurácia média no mesmo teste sintético, em três inicializações.")
colors=[LILA,MINT,GOLD,PINK]
g=f.add_axes((.105,.17,.82,.48),facecolor=BG)
x=[0,1.35,2.7]
offsets=[-.36,-.12,.12,.36]
for m,o,c,label in zip(modes,offsets,colors,["FP32","INT8 simulado","INT4 simulado","Ternário do zero"]):
    heights=[values[(a,m)] for a in archs]
    xs=[xx+o for xx in x]
    bars=g.bar(xs,heights,width=.205,color=c,label=label)
    for xx,h,b in zip(xs,heights,bars):
        g.text(xx,h+1.0,f"{h:.1f}".replace(".",","),ha="center",va="bottom",fontsize=11,color=c,weight="bold")
g.set(ylim=(0,108),xticks=x,xticklabels=archs,ylabel="Acurácia em %")
g.tick_params(colors=INK,labelsize=13)
g.yaxis.label.set_color(INK)
g.grid(axis="y",alpha=.5,color=GRID)
g.set_axisbelow(True)
for p in g.spines.values(): p.set_color(GRID)
leg=g.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncol=4,fontsize=11,frameon=False)
for label in leg.get_texts(): label.set_color(INK)
t(a,.68,.47,"Teste em CPU. Este gráfico mede acurácia, não tamanho de arquivo nem latência.",13,INK)
save(f,"03-resultados.png")

f,a=canvas(4,"Quanto custa servir um LLM?","Reduzir pesos ajuda. Contexto, lote e runtime ainda mandam na memória.")
for x,tag,body,c in [
    (.67,"PESOS","7 bilhões de pesos:\n16 bits → 14 GB\n4 bits → 3,5 GB\nSó códigos, sem escalas.",LILA),
    (4.95,"ATIVAÇÕES","Valores temporários e\nárea de trabalho.\nDependem de lote,\nsequência e kernel.",GOLD),
    (9.23,"KV CACHE","Chaves e valores de\ntokens anteriores.\nPode usar precisão\ndiferente dos pesos.",MINT),
]:
    box(a,x,2.76,4.05,2.9,c)
    t(a,x+.23,5.19,tag,17,c,b=True)
    t(a,x+.23,3.98,body,16,FG)
box(a,.67,1.16,12.63,1.08)
t(a,.94,1.72,r"$\mathrm{bytes}_{\mathrm{KV}} \approx 2\,B\,T\,L\,H_{\rm KV}\,D_{\rm head}\,b$",23,MINT)
t(a,8.07,1.72,"K e V × lote × tokens × camadas ×\ncabeças KV × dimensão × bytes",13,INK)
t(a,.68,.48,"Qualidade, pico de memória e latência são métricas diferentes. Meça as três.",14,INK)
save(f,"04-llm.png")

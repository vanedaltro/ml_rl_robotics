"""Diagramas didáticos. Rode a partir de qualquer pasta."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
bg, ink, soft, teal, orange, blue = "#f2f7fc", "#172d48", "#45617c", "#13766d", "#b65231", "#3b65a2"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":ink,"axes.labelcolor":soft,
                     "xtick.color":soft,"ytick.color":soft})
def panel(num, title, subtitle):
    f=plt.figure(figsize=(13,7.5),dpi=120,facecolor=bg)
    f.text(.055,.922,"DEEP LEARNING E ROBÓTICA   /   14   /   "+num,fontsize=11,weight="bold",color=teal)
    f.text(.055,.844,title,fontsize=26,weight="bold")
    f.text(.055,.785,subtitle,fontsize=12,color=soft)
    return f
def save(f, name):
    f.savefig(OUT/name,facecolor=bg)
    plt.close(f)

rng=np.random.default_rng(42)
x=rng.normal(size=(180,2))@np.array([[2.4,1.4],[.12,.50]])
x-=x.mean(0)
_,s,vh=np.linalg.svd(x,full_matrices=False)
w=vh[0]
if w[0]<0:w=-w
z=(x@w)[:,None]*w
f=panel("01","PCA procura a direção de maior variância",
        "A nuvem é centralizada. Cada ponto recebe uma nova coordenada por projeção sobre o eixo.")
a=f.add_axes([.075,.16,.55,.55],facecolor=bg)
a.grid(alpha=.15)
a.scatter(x[:,0],x[:,1],color=blue,alpha=.35,s=14)
t=np.array([-7,7])[:,None]*w
a.plot(t[:,0],t[:,1],c=teal,lw=2.4)
for i in [1,9,21,34,52,64,89]:
    a.plot([x[i,0],z[i,0]],[x[i,1],z[i,1]],c=orange,lw=1.2)
    a.scatter(*z[i],c=orange,s=19,zorder=4)
a.set(xlabel="variável 1",ylabel="variável 2");a.set_aspect("equal",adjustable="datalim")
f.text(.66,.64,"01  Centralizar",fontsize=16,weight="bold",color=teal)
f.text(.66,.58,r"$X_c=X-\mathbf{1}\mu^\top$",fontsize=17)
f.text(.66,.48,"02  Decompor por SVD",fontsize=16,weight="bold",color=teal)
f.text(.66,.42,r"$X_c=U\Sigma V^\top$",fontsize=17)
f.text(.66,.32,"03  Projetar nos primeiros k eixos",fontsize=15,weight="bold",color=teal)
f.text(.66,.26,r"$Z=X_cV_k$",fontsize=17)
f.text(.055,.066,"Linha verde: primeira componente. Segmentos em laranja: diferenças entre os pontos e suas projeções.",fontsize=11,color=soft)
save(f,"01-como-funciona-pca.png")

f=panel("02","Centralizar muda a origem do cosseno",
        "Um giro ortogonal preserva produtos escalares. Transladar a origem não preserva.")
Q=np.array([1.,0]);A=np.array([3.,0]);B=np.array([1.,1])
for i,mu in enumerate([np.zeros(2),np.array([2.,.5])]):
    a=f.add_axes([.075+.47*i,.22,.39,.44],facecolor=bg)
    a.set(xlim=(-1.45,3.5),ylim=(-.85,1.5),xlabel="coordenada 1",ylabel="coordenada 2")
    a.set_aspect("equal",adjustable="box");a.grid(alpha=.16)
    for v,name,color,delta in [(A,"A",teal,(.08,.05)),(B,"B",orange,(.08,.06)),(Q,"consulta",ink,(.08,-.25))]:
        v=v-mu
        a.annotate("",xy=v,xytext=(0,0),arrowprops={"arrowstyle":"-|>","color":color,"lw":2.6})
        a.text(*(v+delta),name,color=color,fontsize=11,weight="bold")
    f.text(.083+.47*i,.70,["Antes: A vence (cos = 1,00)","Depois: B vence (cos = 0,60)"][i],
           fontsize=15,weight="bold",color=[teal,orange][i])
f.text(.055,.073,"Neste exemplo, mantemos o espaço 2D inteiro. Nenhuma componente foi removida; a centralização já troca o ranking.",fontsize=11,color=soft)
save(f,"02-pca-e-cosseno.png")

f=panel("03","O que cada método tenta preservar?",
        "Quatro objetivos diferentes. Compare os mapas com uma métrica que faça sentido para a tarefa.")
a=f.add_axes([0,0,1,1]);a.axis("off")
rows=[
("PCA","Variação total","Projeta em eixos lineares; bom ponto de partida.", "#dbebfb"),
("UMAP","Vizinhanças locais","Constrói um grafo de afinidade e otimiza seu mapa.", "#cceee5"),
("PaCMAP","Pares em três faixas","Vizinhos, intermediários e distantes; pesos por fase.", "#f9e4d1"),
("TriMAP","Ordem relativa","Tenta manter relações do tipo A perto de B e longe de C.", "#e4e0f9"),
]
for i,(name,goal,how,c) in enumerate(rows):
    y=.66-i*.145
    a.add_patch(FancyBboxPatch((.055,y),.89,.115,boxstyle="round,pad=0.006,rounding_size=.018",
                              facecolor=c,edgecolor="none",transform=a.transAxes))
    f.text(.081,y+.057,name,fontsize=17,weight="bold",va="center")
    f.text(.272,y+.057,goal,fontsize=14,weight="bold",va="center")
    f.text(.52,y+.057,how,fontsize=12,color=soft,va="center")
f.text(.055,.065,"Mapas 2D não têm, por padrão, eixos ou distâncias globais diretamente interpretáveis.",fontsize=11,color=soft)
save(f,"03-objetivos-dos-metodos.png")

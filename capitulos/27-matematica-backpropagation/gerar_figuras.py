from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT=Path("figuras"); OUT.mkdir(exist_ok=True)
PAPER,INK,DIM,TEAL,RUST,BLUE,RULE="#f6f0e1","#24332e","#515f52","#206253","#9b4433","#385d79","#d4d6c4"
def page(section,title,sub):
    f=plt.figure(figsize=(12,6.75),facecolor=PAPER)
    a=f.add_axes((0,0,1,1)); a.set(xlim=(0,100),ylim=(0,100)); a.axis("off")
    a.text(5,93,"SÉRIE ML   /   CAPÍTULO 27   /   "+section,color=TEAL,family="monospace",weight="bold",fontsize=12)
    a.text(5,83,title,color=INK,family="serif",weight="bold",fontsize=28)
    a.text(5,75,sub,color=DIM,fontsize=13)
    return f,a
def txt(a,x,y,msg,color=INK,size=14,**kw):
    a.text(x,y,msg,color=color,fontsize=size,va="center",**kw)
def box(a,x,y,w,h,title,body,color):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.2,rounding_size=1.3",fc="white",ec=color,lw=1.4))
    txt(a,x+2,y+h*.66,title,color,16,weight="bold")
    txt(a,x+2,y+h*.29,body,INK,12)
def arrow(a,x1,y1,x2,y2,color):
    a.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=17,lw=1.5,color=color))
def save(f,name):
    f.savefig(OUT/name,dpi=140,facecolor=PAPER);plt.close(f)
f,a=page("REGRA DA CADEIA","Backpropagation: faça o caminho de volta",
         "Um neurônio, uma ativação e uma loss. Na volta, multiplique pelas derivadas locais.")
box(a,5,42,19,20,r"$w, b$","0,5 e 0,5",BLUE)
box(a,29,42,19,20,r"$z=wx+b$","1,5",TEAL)
box(a,53,42,19,20,r"$a=\mathrm{ReLU}(z)$","1,5",TEAL)
box(a,77,42,18,20,r"$L=\frac{1}{2}(a-y)^2$","0,125",RUST)
for x in (24,48,72): arrow(a,x+.3,52,x+4.7,52,TEAL)
arrow(a,87,32,14,32,RUST)
for x,t in [(17,r"$dw=1\quad db=0{,}5$"),(39,r"$dz=0{,}5$"),(62,r"$da=0{,}5$"),(86,r"$dL/dL=1$")]:
    txt(a,x,24,t,RUST,16,ha="center")
txt(a,5,9,"Um passo SGD com taxa 0,1 reduz a perda de 0,125 para 0,03125 neste exemplo.",DIM,12)
save(f,"01-cadeia.png")
f,a=page("ATIVAÇÕES","O que passa pelo backward?",
         "A ativação transforma o sinal. Sua derivada local transforma o gradiente.")
z=np.linspace(-5,5,501);s=1/(1+np.exp(-z));t=np.tanh(z)
for j,(n,y,g) in enumerate([("Sigmoide",s,s*(1-s)),("Tanh",t,1-t*t),("ReLU",np.maximum(z,0),z>0)]):
    ax=f.add_axes((.075+.30*j,.22,.25,.43),facecolor=PAPER)
    ax.plot(z,y,label="Forward",color=TEAL,lw=2.5)
    if j<2: ax.plot(z,g,label="Derivada",color=RUST,lw=2.2,ls="--")
    else:
        ax.plot([-5,0],[0,0],color=RUST,lw=2.2,ls="--",label="Derivada")
        ax.plot([0,5],[1,1],color=RUST,lw=2.2,ls="--")
        ax.scatter([0,0],[0,1],s=26,edgecolors=RUST,facecolors=[RUST,PAPER],zorder=3)
    ax.axvline(0,color=RULE);ax.grid(color=RULE)
    ax.set_title(n,color=INK,fontweight="bold");ax.set_xlabel("Pré-ativação z",color=DIM)
    ax.tick_params(colors=DIM,labelsize=9)
    for sp in ax.spines.values():sp.set_color(RULE)
    ax.legend(loc="upper left",fontsize=9,frameon=False)
txt(a,5,8,"Sigmoide e tanh saturam nos extremos. Na quina da ReLU, o lab adota derivada local zero.",DIM,12)
save(f,"02-ativacoes.png")
f,a=page("DIMENSÕES","Faça as formas baterem antes de rodar",
         "Um exemplo por linha. B: lote · D: entradas · H: unidades escondidas · C: classes.")
for x,w,key,body,color in [(5,24,"X  [B, D]","Entrada",BLUE),(37,25,"A₁  [B, H]","Ativação escondida",TEAL),(70,25,"Z₂  [B, C]","Logits de saída",RUST)]:
    box(a,x,48,w,17,key,body,color)
arrow(a,30,56,36,56,TEAL);arrow(a,63,56,69,56,TEAL)
txt(a,6,34,"dW₂ = dZ₂.T @ A₁",RUST,16,weight="bold")
txt(a,45,34,"[C, B] @ [B, H] → [C, H]",INK,15)
txt(a,6,24,"dZ₁ = (dZ₂ @ W₂) * máscara",RUST,16,weight="bold")
txt(a,55,24,"[B, H] * [B, H]",INK,15)
txt(a,6,10,"@ combina dimensões. * multiplica elemento a elemento. A média do lote entra só uma vez.",DIM,12)
save(f,"03-shapes.png")
f,a=page("PONTE PARA A CNN","Uma janela vira um escore",
         "Forward local: some os produtos. Backward local: multiplique pelo gradiente recebido.")
def card(x,title,arr,color):
    txt(a,x,64,title,color,15,weight="bold")
    for r in range(2):
        for c in range(2):
            xx=x+c*7; yy=53-r*8
            a.add_patch(FancyBboxPatch((xx,yy-3),6,6,boxstyle="round,pad=.16,rounding_size=.6",ec=RULE,fc="white"))
            txt(a,xx+3,yy,str(arr[r][c]),INK,15,ha="center",weight="bold")
card(6,"Janela",[[1,2],[3,0]],BLUE); card(28,"Filtro",[[1,-1],[0,2]],TEAL)
card(53,"Grad. do filtro",[[3,6],[9,0]],RUST); card(76,"Grad. da janela",[[3,-3],[0,6]],RUST)
txt(a,6,30,"Escore = −1",TEAL,15,weight="bold")
txt(a,53,30,"δ = 3 multiplica cada entrada relevante",RUST,14)
txt(a,6,14,"O filtro é compartilhado entre posições; cada posição soma sua contribuição em dK.",DIM,12)
txt(a,6,8,"Se duas janelas usam o mesmo pixel, somamos as duas contribuições em dX.",DIM,12)
save(f,"04-cnn.png")
f,a=page("CROSS-ENTROPY E KL","De onde vem o gradiente q − p?",
         "O alvo p é fixo. q é a distribuição prevista pelo softmax dos logits z.")
for i,(nm,eq) in enumerate([
("SOFTMAX",r"$q_j=\frac{e^{z_j}}{\sum_k e^{z_k}}$"),
("CROSS-ENTROPY",r"$H(p,q)=-\sum_k p_k\log q_k$"),
("DIVERGÊNCIA KL",r"$D_{\mathrm{KL}}(p\Vert q)=\sum_{k:p_k>0}p_k\log\frac{p_k}{q_k}$")]):
    y=60-16*i
    txt(a,6,y,nm,TEAL,12,family="monospace",weight="bold")
    txt(a,36,y,eq,INK,22)
txt(a,6,11,r"$H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)$",RUST,17)
txt(a,58,11,r"$\frac{\partial H}{\partial z_j}=q_j-p_j$",RUST,17)
save(f,"05-cross-entropy-kl.png")

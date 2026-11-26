from pathlib import Path
import csv
from collections import defaultdict
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
BASE=Path(__file__).resolve().parent
NAVY='#101c31'; BLUE='#2451aa'; TEAL='#047d72'; ORANGE='#ba481d'; MUTED='#475675'; WHITE='#f6f8ff'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelcolor':NAVY,
                     'text.color':NAVY,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#a3afc5','xtick.color':MUTED,'ytick.color':MUTED})
def heading(fig,kicker,title,sub):
    fig.text(.07,.92,kicker,weight='bold',fontsize=11,color=TEAL)
    fig.text(.07,.83,title,weight='bold',fontsize=26)
    fig.text(.07,.765,sub,fontsize=12,color=MUTED)
def save(fig,name):
    fig.savefig(BASE/name,dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
fig=plt.figure(figsize=(11,5.2),facecolor=NAVY)
ax=fig.add_axes([0,0,1,1]);ax.axis('off')
fig.text(.065,.85,'PYTORCH NA PRÁTICA  /  PARTE 5',color='#6debd1',weight='bold',fontsize=13)
fig.text(.065,.64,'Generalização\nem PyTorch',color='white',weight='bold',fontsize=41,linespacing=1.08,va='top')
fig.text(.07,.16,'Treino, teste e o que a rede realmente aprende.',color='#d0ddf7',fontsize=14)
for i, (label,color) in enumerate([('TREINO','#96b7ff'),('VALIDAÇÃO','#ffc491'),('TESTE','#6debd1')]):
    y=.69-i*.19
    ax.add_patch(FancyBboxPatch((.74,y),.19,.11,boxstyle='round,pad=.013,rounding_size=.026',facecolor='#243956',linewidth=0))
    fig.text(.835,y+.055,label,ha='center',va='center',color=color,weight='bold',fontsize=12)
save(fig,'00-capa.png')
fig=plt.figure(figsize=(11,5.1),facecolor=WHITE)
heading(fig,'PROTOCOLO','Separe antes de treinar','Cada autor, episódio ou grupo deve ficar inteiro na mesma divisão.')
ax=fig.add_axes([0,0,1,1]);ax.axis('off')
for i,(title,sub,col) in enumerate([('1 · Treino','Ajusta pesos\ne aprende transformações',BLUE),('2 · Validação','Escolhe arquitetura,\nregularização e checkpoint',ORANGE),('3 · Teste','Mede o desempenho\napós as escolhas',TEAL)]):
    x=.068+i*.3
    ax.add_patch(FancyBboxPatch((x,.29),.255,.31,boxstyle='round,pad=.01,rounding_size=.02',facecolor='white',edgecolor=col,linewidth=1.8))
    fig.text(x+.018,.49,title,weight='bold',color=col,fontsize=17)
    fig.text(x+.018,.42,sub,fontsize=11,va='top')
    if i<2: fig.text(x+.263,.425,'→',weight='bold',fontsize=15,color=MUTED)
fig.text(.07,.15,'Exemplo OCR: a caligrafia da mesma pessoa não aparece em duas divisões.',fontsize=11,color=MUTED)
save(fig,'01-split.png')
rows=list(csv.DictReader((BASE/'curvas.csv').open()))
fig,(a,b)=plt.subplots(1,2,figsize=(11,5.9),facecolor=WHITE,sharey=True)
heading(fig,'EXPERIMENTO EM CPU','Quando a loss de validação vira','FashionMNIST · 600 imagens de treino · 20% dos rótulos trocados · média de três sementes')
fig.subplots_adjust(left=.10,right=.965,bottom=.19,top=.67,wspace=.18)
for ax,key,title in [(a,'grande','MLP grande, sem regularização'),(b,'dropout','Mesma MLP, dropout 0,35')]:
    ax.set_title(title,loc='left',fontsize=13,fontweight='bold',pad=10)
    data=defaultdict(list)
    for r in rows:
        if r['caso']==key:data[int(r['epoca'])].append(r)
    ep=sorted(data)
    for k,label,col in [('treino_nll','Treino',BLUE),('val_nll','Validação',ORANGE)]:
        ax.plot(ep,[mean(float(r[k]) for r in data[e]) for e in ep],label=label,color=col,linewidth=2.6)
    ax.set_xlabel('Época');ax.grid(axis='y',alpha=.23);ax.set_axisbelow(True)
    ax.set_xticks([1,10,20,30,40])
a.set_ylabel('Entropia cruzada padrão')
handles,labels=a.get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.018),ncol=2,frameon=False)
save(fig,'02-curvas.png')
rows=list(csv.DictReader((BASE/'resultados.csv').open()))
items=[('pequena','MLP pequena'),('grande','MLP grande'),('dropout','+ Dropout'),('adamw','+ Weight decay'),('smoothing','+ Label smoothing'),('mais_dados','+ 5× dados')]
fig,ax=plt.subplots(figsize=(11,6.3),facecolor=WHITE)
heading(fig,'TESTE OFICIAL','Qual checkpoint você levaria?','Acurácia em 10.000 imagens · porcentagens médias de três sementes')
fig.subplots_adjust(left=.255,right=.91,bottom=.19,top=.68)
for i,(k,label) in enumerate(items):
    rs=[r for r in rows if r['caso']==k]
    q,w=[100*mean(float(r[f]) for r in rs) for f in ('test_acc_final','test_acc_melhor')]
    ax.plot([q,w],[i,i],color='#a3afc5',linewidth=2,zorder=1)
    ax.scatter(q,i,color=BLUE,s=80,zorder=2,label='Última época' if i==0 else '')
    ax.scatter(w,i,marker='D',color=TEAL,s=57,zorder=3,label='Menor loss de validação' if i==0 else '')
    ax.text(max(q,w)+.52,i,f'{w:.2f}%'.replace('.',','),va='center',color=TEAL,fontweight='bold',fontsize=10)
ax.set_yticks(range(len(items)),[v for _,v in items]);ax.invert_yaxis()
ax.set_xlim(64,85);ax.set_xticks([65,70,75,80,85]);ax.set_xlabel('Acurácia no teste (%)')
ax.grid(axis='x',alpha=.23);ax.legend(loc='upper center',bbox_to_anchor=(.49,-.26),ncol=2,frameon=False)
save(fig,'03-resultados.png')

from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT=Path('images');OUT.mkdir(exist_ok=True)
BG='#0c1726';FG='#ecf4ff';MUT='#a7bed4';CYAN='#46d5cd';GOLD='#ffce73';BLUE='#6aa9ff'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'text.color':FG,'axes.labelcolor':FG,'xtick.color':MUT,'ytick.color':MUT,'axes.edgecolor':MUT,'savefig.facecolor':BG})
def canvas(title,subtitle):
    fig=plt.figure(figsize=(16,10),facecolor=BG)
    fig.text(.06,.92,title,fontsize=29,weight='bold')
    fig.text(.06,.865,subtitle,fontsize=17,color=MUT)
    fig.text(.06,.035,'CAPÍTULO 20  /  LABORATÓRIO PYTORCH  /  DADOS SINTÉTICOS',fontsize=11,color=MUT)
    return fig

def card(ax,x,y,w,h,label,color=CYAN):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.02',facecolor='#16283c',edgecolor=color,lw=2))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=17)

f=canvas('Um token à frente. Nenhum olhar para o futuro.','O alvo é deslocado; a máscara causal protege o treinamento.')
a=f.add_axes([.06,.54,.88,.24]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
for i,(x,y) in enumerate(zip(['<bos>','ana','testa','sensor','oficina'],['ana','testa','sensor','oficina','<eos>'])):
    card(a,.02+i*.197,.61,.15,.27,x)
    card(a,.02+i*.197,.12,.15,.27,y,GOLD)
    a.annotate('',xy=(.095+i*.197,.40),xytext=(.095+i*.197,.59),arrowprops={'arrowstyle':'->','color':MUT,'lw':2})
f.text(.06,.79,'ENTRADAS',fontsize=12,color=CYAN);f.text(.06,.55,'ALVOS',fontsize=12,color=GOLD)
a=f.add_axes([.08,.12,.31,.33]);a.imshow(np.tril(np.ones((5,5))),cmap=matplotlib.colors.ListedColormap(['#26374b',CYAN]),vmin=0,vmax=1)
for i in range(5):
    for j in range(5):a.text(j,i,'✓' if j<=i else '×',ha='center',va='center',color=BG if j<=i else FG,fontsize=20)
a.set_xticks(range(5),range(1,6));a.set_yticks(range(5),range(1,6));a.set_xlabel('Posição consultada');a.set_ylabel('Posição atual')
f.text(.47,.39,r'$A=\mathrm{softmax}(QK^{T}/\sqrt{d_k}+M)$',fontsize=24)
f.text(.47,.30,'M = 0 no presente e no passado\nM = −∞ no futuro',fontsize=20,linespacing=1.6)
f.text(.47,.18,'Teste de causalidade: alterar o futuro\nnão pode alterar os logits do prefixo.',fontsize=18,color=MUT)
f.savefig(OUT/'01-shift-causal.png',dpi=140);plt.close(f)

r=json.loads(Path('results.json').read_text())
f=canvas('Do pré-treino à especialização','Pesos aleatórios → corpus geral → adaptação ao domínio técnico.')
a=f.add_axes([.06,.48,.88,.31]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
for x,label,col in [(0.02,'INÍCIO\nPesos aleatórios',MUT),(.36,'PRÉ-TREINO\n300 atualizações',BLUE),(.70,'FINE-TUNING\n150 atualizações',CYAN)]:card(a,x,.18,.26,.62,label,col)
for x in [.30,.64]:a.annotate('',xy=(x+.045,.49),xytext=(x-.015,.49),arrowprops={'arrowstyle':'->','color':FG,'lw':2})
f.text(.08,.37,'CORPUS GERAL',fontsize=13,color=BLUE);f.text(.08,.30,'ana le livro sala',fontsize=23)
f.text(.55,.37,'CORPUS TÉCNICO',fontsize=13,color=CYAN);f.text(.55,.30,'ana testa sensor oficina',fontsize=23)
f.text(.08,.19,'Por domínio: 1.075 treino  /  230 validação  /  231 teste',fontsize=21,color=GOLD)
f.text(.08,.12,'Frases únicas e conjuntos disjuntos. Mesma gramática: não é português aberto.',fontsize=17,color=MUT)
f.savefig(OUT/'02-pretrain-finetune.png',dpi=140);plt.close(f)

f=canvas('O que o experimento realmente mediu','Perplexidade no teste técnico: mesmos 1.155 alvos e mesmo vocabulário.')
keys=['domain_unigram','domain_pretrain','domain_finetune','domain_scratch_150']
labels=['Unigrama\n(treino técnico)','Pré-treino\n(sem adaptar)','Pré + ajuste\n(300 + 150)','Do zero técnico\n(150 passos)']
a=f.add_axes([.10,.36,.82,.42],facecolor=BG)
vals=[r['metrics'][k]['ppl'] for k in keys]
b=a.bar(labels,vals,color=[MUT,BLUE,CYAN,GOLD],width=.58)
a.set_yscale('log');a.set_ylabel('PPL · escala logarítmica');a.set_ylim(1,max(vals)*2.3)
a.spines[['top','right']].set_visible(False)
for rect,v in zip(b,vals):a.text(rect.get_x()+rect.get_width()/2,v*1.1,f'{v:.2f}',ha='center',fontsize=21,weight='bold')
f.text(.10,.24,r'$\mathrm{PPL}=\exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log p(x_t\mid x_{<t})\right)$',fontsize=24)
f.text(.10,.12,'Menor é melhor neste teste. Não mede verdade, raciocínio ou utilidade.\nUm seed, corpus artificial, orçamento total desigual entre as rotas.',fontsize=17,color=MUT)
f.savefig(OUT/'03-resultados-ppl.png',dpi=140);plt.close(f)

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from geometria import vector_demo, manifold_demo, probability_demo

ink, soft, teal, orange, blue, bg = '#152941','#415a76','#14776b','#ba5126','#365ca3','#f3f7fb'
P = Path(__file__).parent
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':ink,
                     'axes.labelcolor':soft,'xtick.color':soft,'ytick.color':soft})
def frame(number, title, subtitle):
    f=plt.figure(figsize=(13,7.5),dpi=120,facecolor=bg)
    f.text(.055,.924,'DEEP LEARNING E ROBÓTICA  /  13  /  '+number,fontsize=11,color=teal,weight='bold')
    f.text(.055,.844,title,fontsize=26,weight='bold')
    f.text(.055,.785,subtitle,fontsize=12,color=soft)
    return f

def save(f, name):
    f.savefig(P/name,facecolor=bg)
    plt.close(f)

q,x,c,d=vector_demo()
f=frame('01','Mesmo ângulo, vizinhos diferentes','A medida muda o ranking mesmo quando os vetores continuam iguais.')
a=f.add_axes([.08,.18,.54,.51],facecolor=bg)
for v,name,color,dy in [(q,'consulta',ink,-.085),(x[0],'A',teal,.035),(x[1],'B',orange,.035),(x[2],'C',blue,.035)]:
    v=v.numpy()
    a.annotate('',v,(0,0),arrowprops={'arrowstyle':'-|>','color':color,'lw':2.5})
    a.text(v[0]+.04,v[1]+dy,name,color=color,weight='bold',fontsize=12)
a.set(xlim=(-.12,5.45),ylim=(-.24,1.65),xlabel='coordenada 1',ylabel='coordenada 2')
a.set_aspect('equal', adjustable='box'); a.grid(alpha=.18)
f.text(.68,.64,'Por cosseno: A',fontsize=17,weight='bold',color=teal)
f.text(.68,.575,'A aponta na mesma direção.',fontsize=12,color=soft)
f.text(.68,.445,'Por euclidiana: B',fontsize=17,weight='bold',color=orange)
f.text(.68,.38,'B está a 0,2828 da consulta.',fontsize=12,color=soft)
f.text(.68,.245,r'$\|\hat{x}-\hat{y}\|_2^2=2(1-\cos(x,y))$',fontsize=15)
f.text(.68,.18,'Após normalizar, os rankings coincidem.',fontsize=11,color=soft)
save(f,'01-cosseno-e-euclidiana.png')

xy,i,j,chord,arc=manifold_demo(); v=xy.numpy()
f=frame('02','Perto em linha reta, longe pela curva','Uma espiral tem duas coordenadas ambientes, mas apenas um parâmetro intrínseco.')
a=f.add_axes([.08,.14,.59,.59],facecolor=bg)
a.plot(v[:,0],v[:,1],color=blue,alpha=.34,lw=2)
a.plot(v[i:j+1,0],v[i:j+1,1],color=teal,lw=3)
a.plot(v[[i,j],0],v[[i,j],1],'--o',color=orange,lw=2)
for idx,name in [(i,'A'),(j,'B')]:
    a.annotate(name,v[idx],xytext=(7,8),textcoords='offset points',weight='bold')
a.set_aspect('equal'); a.axis('off')
f.text(.66,.55,f'Corda: {chord:.2f}'.replace('.',','),fontsize=20,color=orange,weight='bold')
f.text(.66,.45,f'Arco: {arc:.2f}'.replace('.',','),fontsize=20,color=teal,weight='bold')
f.text(.66,.34,'O arco segue a ordem conhecida\ndo gerador. Em dados reais,\nessa ordem costuma ser desconhecida.',fontsize=12,color=soft,linespacing=1.45)
f.text(.055,.061,'O laboratório também leva a curva a 64 dimensões com uma transformação que preserva as distâncias euclidianas.',fontsize=10,color=soft)
save(f,'02-espiral-manifold.png')

_,_,kl,rev,js,_,_=probability_demo()
f=frame('03','KL e Jensen–Shannon com as contas abertas','P = (0,8; 0,2), Q = (0,5; 0,5). Logs naturais, em nats.')
f.text(.06,.63,r'$D_{\mathrm{KL}}(P\Vert Q)=\sum_i p_i\log\frac{p_i}{q_i}$',fontsize=23)
f.text(.06,.45,r'$M=(P+Q)/2$',fontsize=21)
f.text(.06,.29,r'$\mathrm{JSD}(P,Q)=\frac{1}{2}D_{\mathrm{KL}}(P\Vert M)+\frac{1}{2}D_{\mathrm{KL}}(Q\Vert M)$',fontsize=20)
f.text(.59,.64,f'KL(P ‖ Q) = {kl:.4f}'.replace('.',','),fontsize=17,weight='bold',color=teal)
f.text(.59,.54,f'KL(Q ‖ P) = {rev:.4f}'.replace('.',','),fontsize=17,weight='bold',color=orange)
f.text(.06,.135,f'JSD = {js:.4f}  •  A raiz da JSD é a distância Jensen–Shannon.'.replace('.',',',1),fontsize=14,color=soft)
save(f,'03-formulas-kl-js.png')

f=frame('04','Mesma divergência, outro custo de transporte','Os três picos carregam uma unidade de massa. O eixo é uma variável numérica.')
a=f.add_axes([.09,.205,.84,.45],facecolor=bg)
for y,end,c,t in [(.72,1,teal,'W₁ = 1'),(.28,5,orange,'W₁ = 5')]:
    a.scatter([0,end],[y,y],s=120,c=[blue,c],zorder=3)
    a.annotate('',(end-.05,y),(0.05,y),arrowprops={'arrowstyle':'->','color':c,'lw':2.4})
    a.text(end+.16,y,t,color=c,weight='bold',va='center',fontsize=14)
a.set(xlim=(-.3,6.05),ylim=(0,1),yticks=[],xticks=range(6),xlabel='posição da massa')
a.grid(axis='x',alpha=.18)
for side in ['top','left','right']: a.spines[side].set_visible(False)
f.text(.09,.09,'Sobre a grade comum {0, 1, 5}, KL é infinita e JSD vale ln 2 para ambos os deslocamentos.',fontsize=12,color=soft)
save(f,'04-wasserstein.png')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from grid import *

paper='#f7f2e8'; ink='#162844'; blue='#244c88'; red='#9c3f27'; green='#17684f'; gray='#455472'
def card(title, subtitle, label):
    f=plt.figure(figsize=(12,6),facecolor=paper)
    f.text(.065,.91,title,weight='bold',family='DejaVu Serif',size=26,color=ink)
    f.text(.065,.845,subtitle,size=13,color=gray)
    f.text(.065,.065,'RL EM 5 MINUTOS  /  01 DE 15     •     '+label,size=11,weight='bold',color=blue)
    return f
f=card('O agente age. O ambiente responde.','O estado e a recompensa do próximo passo só chegam depois da ação.', 'CICLO AGENTE–AMBIENTE')
a=f.add_axes([.065,.17,.87,.6]);a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off')
for x, title, body in [(.21,'Agente','Escolhe uma ação\ncom sua política'),(.79,'Ambiente','Aplica as regras\ne devolve o resultado')]:
    a.add_patch(FancyBboxPatch((x-.16,.38),.32,.26,boxstyle='round,pad=.018',edgecolor=blue,facecolor='white',lw=2))
    a.text(x,.56,title,ha='center',size=19,weight='bold',color=ink)
    a.text(x,.45,body,ha='center',va='center',size=12,color=gray)
for p,q,rad,y,txt,c in [((.36,.65),(.64,.65),-.6,.83,'ação  aₜ',blue),((.64,.37),(.36,.37),-.6,.17,'novo estado  sₜ₊₁  +  recompensa  rₜ₊₁',red)]:
    a.add_patch(FancyArrowPatch(p,q,connectionstyle=f'arc3,rad={rad}',arrowstyle='-|>',mutation_scale=19,color=c,lw=2))
    a.text(.5,y,txt,ha='center',size=13,weight='bold',color=c)
f.savefig('01-ciclo.png',dpi=150);plt.close(f)

f=card('O mesmo início, dois retornos','A recompensa mede um passo. O retorno soma as consequências.', 'GRIDWORLD EM PYTHON')
for idx,(seq,head,txt,col) in enumerate([('DDDDCC','Chegou à bandeira','5 passos × (−1)  +  10  =  +5',green),('EC','Caiu no buraco','1 batida × (−1)  −  10  =  −11',red)]):
    a=f.add_axes([.065+idx*.46,.24,.41,.49]);a.set(xlim=(0,5),ylim=(3,0));a.set_aspect('equal');a.axis('off')
    for r in range(3):
        for c in range(5):
            a.add_patch(Rectangle((c,r),1,1,facecolor='#d9e0eb' if (r,c)==PAREDE else 'white',edgecolor='#a6b4c6',lw=1))
    for (r,c),label in [(INICIO,'S'),(BANDEIRA,'+10'),(BURACO,'−10'),(PAREDE,'■')]:
        a.text(c+.5,r+.50,label,ha='center',va='center',color=ink,weight='bold',size=15)
    for (r,c),_,_,(r2,c2),_ in episodio(seq):
        if (r,c)==(r2,c2):
            a.add_patch(Circle((c+.34,r+.65),.15,fill=False,edgecolor=col,lw=2))
            continue
        dx,dy=c2-c,r2-r
        a.add_patch(FancyArrowPatch((c+.5+.23*dx,r+.5+.23*dy),(c2+.5-.25*dx,r2+.5-.25*dy),arrowstyle='-|>',mutation_scale=15,color=col,lw=2))
    f.text(.065+idx*.46,.77,head,weight='bold',color=col,size=17)
    f.text(.065+idx*.46,.176,txt,weight='bold',color=col,size=14)
f.savefig('02-trajetorias.png',dpi=150);plt.close(f)

"""Gera os quatro PNGs do artigo (Python com Pillow e Matplotlib)."""
from pathlib import Path
import csv
import math
import statistics as st
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BG, PANEL, INK, MUTED = '#10201d', '#19322e', '#f0f5e9', '#b4c9bf'
MINT, GOLD, BLUE, GRID = '#8ae6bb', '#ffc47d', '#b9d9ff', '#3a5550'
FONTS=Path('/usr/share/fonts/truetype/dejavu')

def f(n,bold=False):
    try:
        return ImageFont.truetype(str(FONTS / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), n)
    except OSError:
        return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', n)

def panel(title, subtitle, height=750):
    im=Image.new('RGB',(1440,height),BG)
    d=ImageDraw.Draw(im)
    d.text((64,37),'PYTORCH / ROBÓTICA / 03',font=f(22,True),fill=MINT)
    d.text((64,85),title,font=f(51,True),fill=INK)
    d.text((66,157),subtitle,font=f(25),fill=MUTED)
    return im,d

def box(d, bb, txt, c=INK, size=34):
    d.rounded_rectangle(bb,18,fill=PANEL,outline=c,width=3)
    d.text(((bb[0]+bb[2])/2,(bb[1]+bb[3])/2),txt,font=f(size,True),fill=c,anchor='mm')

def arrow(d, pts, c, width=5):
    d.line(pts,fill=c,width=width,joint='curve')
    a,b=pts[-2:]
    ang=math.atan2(b[1]-a[1],b[0]-a[0])
    d.polygon([b,*[(b[0]-18*math.cos(ang+a), b[1]-18*math.sin(ang+a)) for a in (-.5,.5)]],fill=c)

im,d=panel('Como funciona o atalho da ResNet?', 'A mesma entrada percorre dois caminhos; no fim, os resultados são somados.',736)
box(d,(70,388,211,484),'x')
arrow(d,[(211,436),(364,436)],GOLD)
box(d,(365,380,785,492),'F(x)  •  ramo treinável',GOLD,29)
arrow(d,[(785,436),(996,436)],GOLD)
arrow(d,[(262,436),(262,286),(1054,286),(1054,380)],MINT)
d.text((650,249),'Atalho identidade • passa x sem alterar',font=f(29,True),fill=MINT,anchor='mm')
box(d,(997,380,1111,492),'+',MINT,46)
arrow(d,[(1111,436),(1164,436)],INK)
box(d,(1165,388,1375,484),'x + F(x)',INK,30)
d.ellipse((255,429,269,443),fill=INK)
d.rounded_rectangle((64,574,1376,661),12,fill=PANEL)
d.text((90,618),'No backward:   ∂L/∂x = g + J_F(x)ᵀ g      |      g = ∂L/∂y',font=f(28,True),fill=INK,anchor='lm')
im.save('01-bloco-residual.png')

plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':PANEL,'axes.edgecolor':GRID,
    'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
    'font.family':'DejaVu Sans','font.size':13,'legend.labelcolor':INK,'legend.frameon':False,
    'grid.color':GRID,'axes.titlecolor':INK,'axes.spines.top':False,'axes.spines.right':False})

def canvas(title, sub, h=7.5):
    fig=plt.figure(figsize=(14.4,h),dpi=100)
    fig.text(.047,.937,'PYTORCH / ROBÓTICA / 03',color=MINT,fontsize=16,fontweight='bold')
    fig.text(.045,.83,title,fontsize=34,fontweight='bold')
    fig.text(.047,.764,sub,fontsize=16,color=MUTED)
    return fig

fig=canvas('Por que o gradiente fica tão pequeno?', 'Regra da cadeia: cada etapa multiplica a derivada. Eixo vertical logarítmico.')
ax=fig.add_axes([.10,.15,.845,.54])
for base, lbl, col in [(0.5,'Sem atalho: F(z) = 0,5z',BLUE),(.01,'Sem atalho: F(z) = 0,01z',GOLD),(1.01,'Com atalho: z + 0,01z',MINT)]:
    ax.plot(range(21),[base**n for n in range(21)],label=lbl,color=col,linewidth=3,marker='o',markevery=5)
ax.set_yscale('log');ax.set_xlabel('Etapas da cadeia');ax.set_ylabel('|dy/dx|')
ax.set_xticks([0,5,10,15,20]);ax.set_yticks([1,1e-10,1e-20,1e-30,1e-40])
ax.set_ylim(1e-42,20);ax.grid();ax.legend(loc='lower left',fontsize=13)
fig.savefig('02-gradiente-escalar.png');plt.close(fig)

def read(p):
    with open(p) as f:
        return list(csv.DictReader(f))

r=read('treino.csv');g=read('gradientes.csv')
fig=canvas('O atalho mudou o treino. Como?', 'FashionMNIST • 3 sementes • mesmas camadas e lotes • medianas',8)
ax=fig.add_axes([.09,.20,.38,.44])
bx=fig.add_axes([.58,.20,.36,.44])
for mode,col,label in [('plain',GOLD,'Sem atalho'),('residual',MINT,'Com atalho')]:
    ax.plot(range(9),[st.median(float(x['grad_rms']) for x in g if x['modelo']==mode and int(x['estagio'])==i) for i in range(9)],label=label,color=col,marker='o',linewidth=2.5)
    bx.plot(range(1,4),[st.median(float(x['loss_treino']) for x in r if x['modelo']==mode and int(x['epoca'])==i) for i in range(1,4)],label=label,color=col,marker='o',linewidth=2.5)
ax.set_title('Gradiente nas ativações (início)',pad=13); ax.set_yscale('log')
ax.set_xlabel('Estágio: entrada (0) → último bloco (8)');ax.set_ylabel('RMS de ∂L/∂a')
bx.set_title('Perda de treino',pad=13);bx.set_xticks([1,2,3]);bx.set_xlabel('Época');bx.set_ylabel('Entropia cruzada')
for axis in [ax,bx]:axis.grid()
ax.legend(loc='upper right',fontsize=12);bx.legend(loc='upper right',fontsize=12)
fig.text(.047,.067,'O backward vai do estágio 8 ao 0. Aqui a CNN simples amplificou o sinal nessa direção.',color=MUTED,fontsize=15)
fig.savefig('03-gradiente-treino.png');plt.close(fig)

s=read('mujoco.csv')
def ms(v):
    return f"{v:.2f}".replace('.',',')
fig=canvas('A câmera ainda domina o ciclo', 'MuJoCo em CPU • 3 × 120 quadros • mediana das rodadas',7.5)
ax=fig.add_axes([.17,.24,.72,.43])
for idx,(name,title) in enumerate([('plain','Sem atalho'),('residual','Com atalho')]):
    row=[x for x in s if x['modelo']==name]
    start=0
    for k,c,label in [('fisica_ms',GOLD,'Física'),('render_ms',MINT,'Câmera'),('preproc_ms',GRID,'Preparo'),('modelo_ms',BLUE,'Modelo')]:
        v=st.median(float(x[k]) for x in row)
        ax.barh(idx,v,left=start,color=c,height=.44,label=label if idx==0 else None)
        if label=='Câmera':
            ax.text(start+v/2,idx,ms(v)+' ms • câmera',color=BG,va='center',ha='center',fontweight='bold',fontsize=15)
        start+=v
    ax.text(start+.35,idx,'modelo '+ms(st.median(float(x['modelo_ms']) for x in row))+' ms',va='center',fontsize=12)
ax.set_yticks([0,1],['Sem atalho','Com atalho']);ax.invert_yaxis()
ax.set_xlabel('Milissegundos');ax.set_xlim(0,30);ax.legend(loc='upper center',bbox_to_anchor=(.50,-.235),ncol=4,fontsize=13)
fig.text(.047,.047,'Barras somam medianas por etapa; a mediana do ciclo por quadro é calculada à parte.',color=MUTED,fontsize=14)
fig.savefig('04-mujoco.png');plt.close(fig)

m=read('mlp_gradiente.csv')
fig=canvas('Quando o gradiente já nasce minúsculo', 'MLP de 20 camadas • tarefa sintética • antes e depois de 150 atualizações')
ax=fig.add_axes([.12,.205,.33,.43]); bx=fig.add_axes([.60,.205,.33,.43])
for i,(name,color) in enumerate([('plain',GOLD),('residual',MINT)]):
    v=st.median(float(k['norma_grad_peso_1']) for k in m if k['modelo']==name and k['passo']=='0')
    ax.bar(i,v,bottom=1e-15,color=color,width=.58)
    ax.text(i,v*1.65,('2,59 × 10⁻¹⁴' if i==0 else '0,79'),ha='center',fontsize=13)
    x=[0,150]
    y=[st.median(float(k['mse_val']) for k in m if k['modelo']==name and int(k['passo'])==j) for j in x]
    bx.plot(x,y,color=color,label=('Sem atalho' if i==0 else 'Com atalho'),marker='o',linestyle=':',linewidth=3)
ax.set_yscale('log');ax.set_ylim(1e-15,8)
ax.set_xticks([0,1],['Sem atalho','Com atalho'])
ax.set_title('Gradiente dos pesos • 1ª camada',pad=12)
bx.set_yscale('log');bx.set_xticks([0,150]);bx.set_xlabel('Atualizações')
bx.set_title('Erro quadrático na validação',pad=12);bx.legend(loc='center left',fontsize=11)
for a in (ax,bx):
    a.grid(axis='y')
fig.text(.047,.065,'A tarefa, por construção, pede uma pequena correção da entrada. É um laboratório controlado.',color=MUTED,fontsize=14)
fig.savefig('05-mlp-diagnostico.png');plt.close(fig)

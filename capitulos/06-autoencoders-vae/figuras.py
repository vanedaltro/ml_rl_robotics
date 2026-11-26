from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
D=Path(__file__).resolve().parent
BG='#f3f6fc'; NAVY='#111c31'; MUTED='#435671'; TEAL='#05796b'; BLUE='#2755a7'; RUST='#ba532d'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':NAVY,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED})
def save(f,p):
 f.savefig(D/p,dpi=155,facecolor=f.get_facecolor());plt.close(f)
def label(f,k,t,s):
 f.text(.055,.931,k,weight='bold',size=11,color=TEAL)
 f.text(.055,.84,t,weight='bold',size=26)
 f.text(.055,.775,s,size=12,color=MUTED)
def box(ax,x,y,w,h,text,c):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.025',fc='white',ec=c,lw=1.5))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,fontweight='bold',color=c)
def flow(ax,y,labels):
 for i,(t,c) in enumerate(labels):
  x=.05+i*.235;box(ax,x,y,.196,.11,t,c)
  if i+1<len(labels):ax.annotate('',(x+.232,y+.055),(x+.201,y+.055),arrowprops=dict(arrowstyle='->',color=MUTED,lw=1.5))
with np.load(D/'visual.npz') as data: d={k:data[k] for k in data.files}
f=plt.figure(figsize=(11,5.3),facecolor=NAVY)
f.text(.06,.86,'PYTORCH NA PRÁTICA  /  PARTE 6',color='#79f5d8',weight='bold',size=13)
f.text(.06,.68,'Autoencoders\ne VAEs',color='white',weight='bold',size=43,va='top',linespacing=1.04)
f.text(.065,.19,'Da reconstrução à geração de imagens',color='#d0ddf7',size=15)
for i,ids in enumerate([[0,4,8],[9,12,15],[19,23,30]]):
 for j,k in enumerate(ids):
  a=f.add_axes([.62+j*.103,.17+i*.23,.097,.205]);a.imshow(d['random'][1,k],cmap='gray',vmin=0,vmax=1);a.axis('off')
save(f,'00-capa.png')
f=plt.figure(figsize=(11,6.2),facecolor=BG)
label(f,'MODELOS','Três caminhos pelo espaço latente','O decoder reconstrói a entrada ou sintetiza uma imagem a partir de um código.')
a=f.add_axes([0,0,1,1]);a.axis('off')
for y,title,labels in [(.55,'AUTOENCODER', [('Imagem x',BLUE),('Encoder',BLUE),('Código fixo z',RUST),('Decoder → imagem',TEAL)]),
                       (.34,'VAE · TREINO', [('Imagem x',BLUE),('Encoder → μ, log σ²',BLUE),('Amostra de q(z|x)',RUST),('Decoder → imagem',TEAL)]),
                       (.13,'VAE · GERAÇÃO', [('Sem imagem',BLUE),('Prior N(0, I)',BLUE),('Amostra aleatória z',RUST),('Decoder → imagem',TEAL)])]:
 a.text(.05,y+.135,title,color=MUTED,weight='bold',size=11);flow(a,y,labels)
save(f,'01-fluxo.png')
def image_grid(name,kicker,title,sub,arr,labels,cols):
 f=plt.figure(figsize=(11,5.8 if len(arr)==4 else 5.1),facecolor=BG)
 label(f,kicker,title,sub)
 gap=.51/len(arr)
 for i,(data,lab) in enumerate(zip(arr,labels)):
  y=.675-gap*(i+1)
  f.text(.055,y+gap/2,lab,weight='bold',size=11,va='center')
  for j,img in enumerate(data):
   a=f.add_axes([.205+j*.737/cols,y,.688/cols,gap*.94])
   a.imshow(img,cmap='gray',vmin=0,vmax=1);a.axis('off')
 save(f,name)
image_grid('02-recon.png','TESTE OFICIAL','A mesma roupa, três vezes','Uma imagem de cada classe: original e reconstruções pelos centros dos códigos.',
 [d['original'],d['recon'][0],d['recon'][1]],['Originais','AE','VAE'],10)
image_grid('04-geracao.png','AMOSTRAGEM','Sem imagem','Os dois decoders recebem as mesmas 16 amostras de N(0, I), sem seleção manual.',
 [d['random'][0,:8],d['random'][0,8:16],d['random'][1,:8],d['random'][1,8:16]],
 ['AE · 01–08','AE · 09–16','VAE · 01–08','VAE · 09–16'],8)
f,axs=plt.subplots(1,2,figsize=(11,6.2),facecolor=BG)
label(f,'REPRESENTAÇÃO','O prior cobre os códigos?','Pontos = centros dos códigos de teste; círculo = região de 95% do prior normal 2D.')
f.subplots_adjust(left=.07,right=.965,top=.65,bottom=.15,wspace=.22)
lo=np.floor(np.min(d['latents'])/5)*5;hi=np.ceil(np.max(d['latents'])/5)*5
for a,z,t in zip(axs,d['latents'],['AE: códigos sem prior imposto','VAE: centros aproximados ao prior']):
 a.scatter(z[:,0],z[:,1],c=d['labels'],cmap='tab10',s=7,alpha=.4,linewidths=0)
 a.add_patch(Circle((0,0),np.sqrt(5.991),fill=False,ls='--',lw=1.6,ec=NAVY))
 a.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Latente 1',ylabel='Latente 2',title=t)
 a.set_aspect('equal');a.grid(alpha=.2)
f.text(.056,.067,'As cores distinguem as dez classes, mas os rótulos não foram usados para treinar os modelos.',size=10,color=MUTED)
save(f,'03-latente.png')
f=plt.figure(figsize=(11,4.8),facecolor=BG)
label(f,'SISTEMA HÍBRIDO','Geração em espaço latente','Exemplo conceitual de texto para imagem. O prompt condiciona o gerador, não o decoder final.')
a=f.add_axes([0,0,1,1]);a.axis('off')
flow(a,.35,[('Ruído latente',BLUE),('Difusão ou flow',RUST),('Latente refinado',RUST),('Decoder → pixels',TEAL)])
a.text(.385,.59,'Prompt → encoder de texto',ha='center',weight='bold',color=BLUE,size=11)
a.annotate('',(.38,.472),(.38,.57),arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.5))
a.text(.055,.18,'Durante o treino, um codec cria os mapas latentes a partir das imagens reais.',color=MUTED,size=11)
save(f,'05-difusao.png')

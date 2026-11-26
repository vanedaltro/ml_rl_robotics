from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from math import atan2, cos, sin

OUT=Path(__file__).resolve().parent/'figuras'
OUT.mkdir(exist_ok=True)
BASE='/usr/share/fonts/truetype/dejavu/DejaVuSans'
bg='#10192c'; panel='#16243c'; line='#344865'; fg='#f1f5ff'; dim='#b8c9df'; blue='#83b4ff'; green='#7de3c8'; gold='#ffd296'
def canvas(title, sub, num):
    im=Image.new('RGB',(1440,850),bg); d=ImageDraw.Draw(im)
    d.rectangle((0,0,1440,7),fill=green)
    label(d,(62,52),f'SÉRIE ML + ROBÓTICA   /   CAPÍTULO 18   /   {num:02}',22,green,True)
    label(d,(62,105),title,52,fg,True)
    label(d,(64,180),sub,26,dim)
    return im,d
def label(d,xy,text,size=26,color=fg,bold=False,anchor='lt'):
    f=ImageFont.truetype(BASE+('-Bold' if bold else '')+'.ttf',size)
    d.text(xy,text,font=f,fill=color,anchor=anchor)
def box(d,xy,edge=line,fill=panel): d.rounded_rectangle(xy,20,fill,edge,2)
def center(d,xy,t,s=28,c=fg,b=True): label(d,xy,t,s,c,b,'mm')
def arrow(d,p,q,c=blue):
    d.line((p,q),fill=c,width=5)
    a=atan2(q[1]-p[1],q[0]-p[0])
    d.polygon([q]+[(q[0]-15*cos(a+x),q[1]-15*sin(a+x)) for x in (-.5,.5)],fill=c)
def note(d,text):
    d.line((64,757,1376,757),fill=line,width=2)
    label(d,(64,781),text,24,dim)
im,d=canvas('Mixture of Experts na prática','Um token chega; só os especialistas escolhidos executam.',1)
box(d,(65,376,278,496));center(d,(171,423),'Token x',32);center(d,(171,463),'vetor em Rᵈ',21,dim,False)
box(d,(340,376,560,496),blue);center(d,(450,423),'Router',32,blue);center(d,(450,463),'pontua 4 experts',21,dim,False)
arrow(d,(280,436),(336,436))
for y,i,w,c in [(250,0,'não executa',dim),(365,1,'peso ≈ 0,75',green),(480,2,'não executa',dim),(595,3,'peso ≈ 0,25',gold)]:
    box(d,(650,y,920,y+91),c if i in (1,3) else line)
    label(d,(672,y+16),f'Expert {i}',27,c,True);label(d,(672,y+52),w,22,dim)
    if i in (1,3): arrow(d,(565,436),(646,y+43),c); arrow(d,(924,y+43),(1011,488),c)
box(d,(1015,435,1375,550),green);center(d,(1195,480),'Saída ponderada',29,green);center(d,(1195,523),'0,75 · E₁(x) + 0,25 · E₃(x)',23)
note(d,'Top-2: a softmax é calculada somente sobre as duas pontuações selecionadas.')
im.save(OUT/'01-router.png')

im,d=canvas('Onde o MoE entra no Transformer?','A atenção continua compartilhada; o MoE assume o lugar da FFN em alguns blocos.',2)
for i,(x,w,t,s,c) in enumerate([(65,230,'Token + posição','[B, T, D]',dim),(330,250,'Atenção causal','mistura posições',blue),(630,260,'Router + experts','transforma o token',green),(946,428,'Próximo bloco / saída','resíduo segue com D dimensões',gold)]):
    box(d,(x,330,x+w,493),c);center(d,(x+w/2,388),t,25,c);center(d,(x+w/2,445),s,22,dim,False)
for p,q in [(295,326),(581,626),(891,942)]: arrow(d,(p,412),(q,412))
box(d,(334,546,1122,684))
label(d,(357,572),'Dentro de um mini bloco pre-norm',26,fg,True)
label(d,(357,625),'h ← h + Atenção(Norm(h))    |    h ← h + MoE(Norm(h))',25,green)
note(d,'O roteamento é por token e por camada; atenção e especialistas têm funções diferentes.')
im.save(OUT/'02-transformer.png')

im,d=canvas('Rotas equilibradas, acertos diferentes','Médias de 3 sementes. No segundo teste, nenhuma combinação apareceu no treino.',3)
box(d,(64,250,710,720));box(d,(735,250,1376,720))
label(d,(91,277),'Carga por expert',30,fg,True)
label(d,(762,277),'Acurácia do MoE com balanceamento',27,fg,True)
label(d,(91,326),'No teste com combinações repetidas',21,dim)
label(d,(762,326),'Mesmo orçamento de treino',21,dim)
for i,(a,b) in enumerate(zip((31.24,44.25,17.20,7.31),(21.57,24.58,26.38,27.46))):
    y=385+i*64
    label(d,(91,y+4),f'E{i}',22,dim,True)
    for v,c,off in [(a,gold,0),(b,green,24)]:
        d.rounded_rectangle((143,y+off,143+10*v,y+off+15),6,fill=c)
        label(d,(153+10*v,y+off-4),f'{v:.2f}%'.replace('.',','),18,c,True)
label(d,(91,661),'Sem termo auxiliar',19,gold,True)
label(d,(330,661),'Com termo auxiliar',19,green,True)
for (y,v,t,c) in [(386,92.77,'511/512 já vistos',blue),(513,29.69,'64 combinações inéditas',green)]:
    label(d,(762,y),t,23,dim,True)
    d.rounded_rectangle((762,y+46,1318,y+65),8,fill=line)
    d.rounded_rectangle((762,y+46,762+5.56*v,y+65),8,fill=c)
    label(d,(762,y+81),f'{v:.2f}%'.replace('.',','),26,c,True)
note(d,'Os testes têm divisões distintas. Não é uma avaliação de LLMs nem um benchmark de velocidade.')
im.save(OUT/'03-resultados.png')

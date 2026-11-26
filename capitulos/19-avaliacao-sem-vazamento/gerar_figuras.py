from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent/'figuras'
OUT.mkdir(exist_ok=True)
W,H=1400,850
BG='#101C30'; INK='#F2F7FF'; MUTED='#B8C8E3'; GRID='#324868'
TEAL='#82E9CA'; BLUE='#79B9FF'; ORANGE='#FFAD7E'; YELLOW='#F6D57B'


def font(size, bold):
    regular='/System/Library/Fonts/Supplemental/Arial.ttf'
    strong='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
    alt='DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(strong if bold else regular, size)
    except OSError:
        return ImageFont.truetype(alt,size)


def text(d,x,y,s,size=30,fill=INK,bold=False,anchor=None):
    d.text((x,y),s,font=font(size,bold),fill=fill,anchor=anchor)


def image(number,title,subtitle):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    d.rectangle((0,0,1400,10),fill=TEAL)
    text(d,72,50,f'PYTORCH NA PRÁTICA  /  CAPÍTULO 19  /  {number:02}',25,TEAL,True)
    text(d,72,100,title,64,INK,True)
    text(d,73,192,subtitle,33,MUTED)
    return im,d


im,d=image(1,'Quando o teste já viu a resposta','Janelas diferentes não são necessariamente dados independentes.')
for x in range(24):
    left=150+45*x
    d.rounded_rectangle((left,327,left+40,415),radius=7,fill=BLUE)
    d.rounded_rectangle((left+90,490,left+130,578),radius=7,fill=ORANGE)
for i in range(2,24):
    x=170+45*i
    d.line((x,415,x,490),fill=GRID,width=2)
text(d,150,277,'TREINO  ·  JANELA A  ·  AMOSTRAS 0 A 23',26,BLUE,True)
text(d,240,601,'TESTE  ·  JANELA B  ·  AMOSTRAS 2 A 25',26,ORANGE,True)
d.line((240,710,1225,710),fill=TEAL,width=3)
d.line((240,700,240,721),fill=TEAL,width=3);d.line((1225,700,1225,721),fill=TEAL,width=3)
text(d,733,757,'22 de 24 amostras compartilhadas',39,TEAL,True,'mm')
im.save(OUT/'01-capa.png')

im,d=image(2,'Três formas de separar os dados','A unidade de separação precisa representar o que vai chegar novo.')
for i,(title,note) in enumerate([
    ('Por janela','Episódios misturados'),
    ('Por episódio','Máquinas conhecidas'),
    ('Por máquina','Máquinas inéditas'),
]):
    x=72+i*426
    d.rounded_rectangle((x,283,x+398,697),radius=22,fill='#162942',outline=GRID,width=2)
    text(d,x+25,312,title,39,INK,True)
    text(d,x+25,371,note,26,MUTED)
    for j in range(3):
        y=449+69*j
        text(d,x+26,y+9,f'M{j+1}',22,MUTED,True)
        for k in range(6):
            if i==0: c=ORANGE if (k+j)%3==0 else BLUE
            elif i==1: c=ORANGE if (j+k//2)%3==2 else BLUE
            else: c=ORANGE if j==2 else BLUE
            l=x+81+45*k
            d.rounded_rectangle((l,y,l+39,y+41),radius=6,fill=c)
            if k%2==0:
                d.line((l-3,y-5,l-3,y+45),fill=GRID,width=2)
text(d,78,754,'■  Treino',26,BLUE,True)
text(d,260,754,'■  Teste',26,ORANGE,True)
text(d,435,754,'M = máquina   ·   a cada par de caixas: um episódio',26,MUTED)
im.save(OUT/'02-splits.png')

im,d=image(3,'O mesmo MLP, três respostas','Acurácia média em dados sintéticos · 3 sementes · erro: ±1 DP amostral')
d.rounded_rectangle((75,267,106,297),6,fill=TEAL)
text(d,120,265,'MLP',29,INK,True)
d.rounded_rectangle((242,267,273,297),6,fill=YELLOW)
text(d,287,265,'Classe majoritária',29,INK,True)
left=300;scale=9.6
for p in range(0,101,25):
    x=left+scale*p
    d.line((x,334,x,733),fill=GRID,width=2)
    text(d,x,759,str(p)+'%',25,MUTED,False,'mm')
with open(Path(__file__).resolve().parent/'resultados.json') as f:
    rows=json.load(f)['summary']
for i,row in enumerate(rows):
    y=350+132*i
    text(d,77,y+32,('Janelas','Episódios','Máquinas')[i],31,INK,True)
    for j,(model,color) in enumerate([('mlp',TEAL),('majority',YELLOW)]):
        st=row[model]['accuracy_pct'];val=st['mean'];sd=st['sample_sd']
        yy=y+j*48; to=left+scale*val
        d.rounded_rectangle((left,yy,to,yy+39),7,fill=color)
        text(d,left+14,yy+5,f'{val:.2f}%'.replace('.',','),25,BG,True)
        lo=left+scale*(val-sd);hi=left+scale*(val+sd)
        if sd:
            d.line((lo,yy+20,hi,yy+20),fill=INK,width=3)
            for end in [lo,hi]:
                d.line((end,yy+10,end,yy+30),fill=INK,width=3)
text(d,77,801,'Todos os protocolos: 1.728 janelas no treino e 576 janelas no teste por execução.',25,MUTED)
im.save(OUT/'03-resultados.png')

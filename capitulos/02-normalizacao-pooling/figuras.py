from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

FONT = Path('/usr/share/fonts/truetype/dejavu')
BG, PANEL, WHITE, MUTED = '#101727', '#1d2a41', '#f6f8ff', '#b8c8df'
BLUE, ORANGE, TEAL = '#8bbcff', '#ffc07f', '#89e3cf'

def ff(n, bold=False):
    return ImageFont.truetype(str(FONT / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), n)

def frame(title, subt, h=735):
    im = Image.new('RGB', (1440,h), BG)
    d = ImageDraw.Draw(im)
    d.text((58, 43), 'PYTORCH  /  ROBÓTICA  /  GUIA DE TREINO', font=ff(25,True), fill=ORANGE)
    d.text((58, 96), title, font=ff(51,True), fill=WHITE)
    d.text((60, 163), subt, font=ff(26), fill=MUTED)
    return im,d

im,d=frame('Qual resposta a rede precisa dar?', 'A tarefa define o formato da saída e a função de perda.')
for i,(c,k,left,mid,right) in enumerate([
    (BLUE,'Uma entre K classes','K logits','CrossEntropyLoss','argmax → classe'),
    (ORANGE,'Sim/não ou várias etiquetas','1 ou K logits','BCEWithLogitsLoss','sigmoid → limiar'),
    (TEAL,'Um número contínuo','1 valor linear','MSE, MAE ou Huber','valor previsto')
]):
    y=235+i*139
    d.rounded_rectangle((56,y,1384,y+114),radius=16,fill=PANEL)
    d.text((80,y+17),k,font=ff(29,True), fill=c)
    d.text((80,y+67),left,font=ff(26),fill=WHITE)
    d.text((560,y+67),'→  '+mid,font=ff(26),fill=WHITE)
    d.text((1050,y+67),'→  '+right,font=ff(25),fill=WHITE)
d.text((58,665),'Na inferência, use sigmoid ou softmax se precisar de probabilidades.',font=ff(23),fill=MUTED)
im.save('01-guia-saidas.png')

im,d=frame('BatchNorm muda com o modo', 'Exemplo em uma CNN com tensor [N, C, H, W].')
for x,col,head,cells in [
    (56,ORANGE,'model.train()', ['Estatísticas do lote atual','Agrega N, H, W por canal','Atualiza as médias móveis']),
    (742,BLUE,'model.eval()', ['Estatísticas armazenadas','O lote novo não atualiza as médias','Pode inferir uma imagem por vez'])
]:
    d.rounded_rectangle((x,255,x+642,571),radius=22,fill=PANEL,outline=col,width=3)
    d.text((x+34,290),head,font=ff(40,True),fill=col)
    for j,lbl in enumerate(cells):
        d.text((x+34,382+j*53),'•  '+lbl,font=ff(27),fill=WHITE)
d.text((58,653),'inference_mode() controla gradientes; não substitui model.eval().',font=ff(25),fill=MUTED)
im.save('02-batchnorm.png')

im,d=frame('Pooling: o que fica do mapa?', 'Mesma entrada, janela 2 × 2, passo 2. Cada cor marca uma janela.', 810)
colors=['#2d4270','#76503a','#285d5c','#666048']
def grade(values,x,y,size,group):
    for i,row in enumerate(values):
        for j,v in enumerate(row):
            color=colors[2*(i//2)+(j//2)] if group else colors[2*i+j]
            dx=x+j*size;dy=y+i*size
            d.rounded_rectangle((dx+2,dy+2,dx+size-2,dy+size-2),radius=7,fill=color)
            d.text((dx+size/2,dy+size/2),str(v),font=ff(30,True),fill=WHITE,anchor='mm')
for x,label in [(60,'Mapa 4 × 4'),(590,'Máximo 2 × 2'),(1020,'Média 2 × 2')]:
    d.text((x,249),label,font=ff(34,True),fill=WHITE)
grade([[1,3,2,0],[4,2,1,5],[0,1,8,2],[2,6,4,11]],65,314,76,True)
grade([[4,5],[6,11]],590,350,97,False)
grade([['2,5','2'],['2,25','6,25']],1020,350,99,False)
d.text((437,429),'→',font=ff(52,True),fill=ORANGE)
d.text((891,429),'ou',font=ff(32,True),fill=MUTED)
d.rounded_rectangle((56,675,1384,755),radius=15,fill=PANEL)
d.text((83,697),'Média global por canal: 3,25 · Um valor para o mapa inteiro.',font=ff(28,True),fill=TEAL)
im.save('03-pooling.png')

import csv, statistics
with open('benchmark_cpu.csv') as f:
    bench=list(csv.DictReader(f))
im,d=frame('O que mudou na CPU?', 'Medianas de 3 sementes · FashionMNIST · 2 épocas · 2 threads',900)
d.text((525,234),'Forward: lote de 32 (ms)',font=ff(28,True),fill=BLUE)
d.text((1050,234),'Validação',font=ff(28,True),fill=ORANGE)
for i,(key,label) in enumerate([
    ('sem_norm+max','Sem norma + MaxPool'),('batch+max','BatchNorm + MaxPool'),
    ('layer_c+max','LayerNorm + MaxPool'),('group+max','GroupNorm + MaxPool'),
    ('batch+avg','BatchNorm + AvgPool'),('batch+stride','BatchNorm + Conv passo 2')
]):
    y=292+i*83
    subset=[r for r in bench if r['modelo']==key]
    ms=statistics.median(float(r['b32_ms']) for r in subset)
    acc=statistics.median(float(r['val_acc']) for r in subset)
    d.rounded_rectangle((56,y,1384,y+64),radius=10,fill=PANEL)
    d.text((75,y+17),label,font=ff(26,True),fill=WHITE)
    d.rounded_rectangle((525,y+19,525+int(365*ms/3.2),y+45),radius=9,fill=BLUE)
    d.text((910,y+20),f'{ms:.2f}'.replace('.',','),font=ff(24),fill=WHITE)
    d.text((1050,y+15),f'{100*acc:.2f}%'.replace('.',','),font=ff(29,True),fill=ORANGE)
d.text((58,825),'Forward mede só a CNN, sem câmera ou física. Não é um ranking universal.',font=ff(23),fill=MUTED)
im.save('04-benchmark-cpu.png')

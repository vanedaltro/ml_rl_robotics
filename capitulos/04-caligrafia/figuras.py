from pathlib import Path
import csv
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, BERRY, AMBER, TEAL, MUTED = "#f6f0e8", "#241733", "#813c75", "#b06a2b", "#19675f", "#5f5264"
CREAM, LINE, PAPER = "#fffaf3", "#dfd1d8", "#eee3e7"
FONTS=Path('/usr/share/fonts/truetype/dejavu')

def f(n,bold=False):
    name='DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path=FONTS/name
    return ImageFont.truetype(str(path) if path.exists() else name,n)

def label(d,p,s,v=25,fill=INK,b=False,anchor=None):
    d.text(p,s,font=f(v,b),fill=fill,anchor=anchor)

def panel(title,sub,h=755):
    im=Image.new("RGB",(1440,h),BG)
    d=ImageDraw.Draw(im)
    label(d,(62,38),"PYTORCH  /  ROBÓTICA  /  04",23,BERRY,True)
    label(d,(62,81),title,51,INK,True)
    label(d,(65,156),sub,25,MUTED)
    return im,d

def box(d, bounds, color=LINE, fill=CREAM):
    d.rounded_rectangle(bounds,18,fill=fill,outline=color,width=3)

def arrow(d,a,b,c=TEAL):
    x,y=b
    ang=math.atan2(y-a[1],x-a[0])
    d.line((a,b),fill=c,width=5)
    d.polygon([b,*[(x-15*math.cos(ang+t),y-15*math.sin(ang+t)) for t in (-.6,.6)]],fill=c)

def flux():
    im,d=panel("Como uma imagem vira texto?", "A largura da imagem vira a sequência que o modelo lê.",740)
    for i,(top,a,b) in enumerate([
        ("01  IMAGEM","24 px de altura","Letras e dígitos"),
        ("02  CNN","Mapa visual","Uma coluna por passo"),
        ("03  ENCODER","BiGRU / BiLSTM","Contexto da linha"),
        ("04  DECODER","GRU + atenção","Um token por vez"),
    ]):
        x=63+338*i
        box(d,(x,255,x+300,512),color=BERRY if i==3 else LINE)
        label(d,(x+20,280),top,24,BERRY,True)
        label(d,(x+20,345),a,32,INK,True)
        label(d,(x+20,430),b,21,MUTED)
        if i<3: arrow(d,(x+305,385),(x+332,385))
    box(d,(63,574,1377,662),color=TEAL)
    label(d,(88,617),"Atenção: a cada saída, o decoder volta a consultar as colunas codificadas.",27,TEAL,True,'lm')
    im.save("01-fluxo-ocr.png")

def gates():
    im,d=panel("LSTM e GRU: o que fica na memória?", "Ambas usam portas aprendidas para preservar e atualizar informação.",770)
    for x,name,c,desc,lines in [
        (62,"LSTM",BERRY,"Estado h + memória c",["f mantém parte da memória c","i grava parte da proposta","o define o que fica visível em h"]),
        (744,"GRU",TEAL,"Um único estado h",["r ajusta como se calcula a proposta","z preserva parte do estado antigo","1 − z incorpora parte da proposta"]),
    ]:
        box(d,(x,233,x+634,664),c)
        label(d,(x+33,264),name,42,c,True)
        label(d,(x+34,339),desc,31,INK,True)
        for i,line in enumerate(lines):
            label(d,(x+34,423+65*i),line,26,MUTED)
    label(d,(64,707),"Se f ≈ 1 (LSTM) ou z ≈ 1 (GRU), a parcela da memória antiga tende a ser preservada.",24,MUTED)
    im.save("02-lstm-gru.png")

def what():
    im,d=panel("Atenção em quatro colunas", "A consulta do decoder dá uma nota para cada posição da imagem.",795)
    weights=[("Coluna 1","0","9%"),("Coluna 2","2","67%"),("Coluna 3","1","24%"),("Padding","8 → −∞","0%")]
    for i,(name,score,pct) in enumerate(weights):
        x=64+336*i
        c=[BERRY,TEAL,BERRY,MUTED][i]
        box(d,(x,228,x+304,548),c if i==1 else LINE,PAPER if i==3 else CREAM)
        label(d,(x+24,251),name,31,c,True)
        label(d,(x+25,322),"Escore  "+score,24,INK)
        label(d,(x+25,405),pct,48,c,True)
        label(d,(x+25,491),"do peso final",23,MUTED)
    box(d,(64,616,1375,710),TEAL)
    label(d,(92,663),"Contexto = soma ponderada dos valores. Padding ignorado antes do softmax.",27,TEAL,True,'lm')
    im.save("03-atencao.png")

def heat():
    spec=json.loads(Path("figura-exemplo.json").read_text())
    result=json.loads(Path(f"resultado-{spec['modelo']}.json").read_text())
    casos=[result['demos'][i] for i in spec['indices_figura']]
    fig=plt.figure(figsize=(14.4,9),dpi=100,facecolor=BG)
    fig.text(.044,.94,"PYTORCH  /  ROBÓTICA  /  04",color=BERRY,fontsize=16,fontweight="bold")
    fig.text(.043,.867,"Onde o decoder consultou a imagem?",color=INK,fontsize=36,fontweight="bold")
    fig.text(.045,.813,f"Pesos reais • encoder Bi{spec['modelo'].upper()} • escritores fora do treino • inferência gulosa",color=MUTED,fontsize=16)
    for n,m in enumerate(casos):
        x=.075+n*.474
        img=np.asarray(Image.open(m['arquivo']).convert('L'))
        w=img.shape[1]
        steps=next((i+1 for i,t in enumerate(m['tokens']) if t==2),len(m["tokens"]))
        a=np.array(m['alpha'][:steps])
        lab=['FIM' if t==2 else ('SP' if t==39 else 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '[t-3]) for t in m['tokens'][:steps]]
        fig.text(x,.738,f"Referência: {m['ref']}   |   Previsão: {m['pred'] or '∅'}",color=INK,fontsize=17,fontweight="bold")
        top=fig.add_axes([x,.594,.354,.107])
        top.imshow(img,cmap='gray_r',vmin=0,vmax=255,aspect='auto',extent=(0,w,0,1))
        top.axis('off')
        ax=fig.add_axes([x,.214,.354,.335])
        edge=np.minimum(np.arange(a.shape[1]+1)*4,w)
        hm=ax.pcolormesh(edge,np.arange(len(a)+1),a,vmin=0,vmax=1,cmap='magma',shading='flat')
        ax.invert_yaxis()
        ax.set_yticks(np.arange(len(lab))+.5,lab)
        ax.tick_params(colors=MUTED,labelsize=12)
        ax.set_xlabel('Posição horizontal na imagem (px)',color=MUTED,fontsize=12)
        ax.set_xlim(0,w)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax=fig.add_axes([.274,.109,.452,.024])
    cb=fig.colorbar(hm,cax=cax,orientation='horizontal',ticks=[0,.25,.5,.75,1])
    cb.ax.tick_params(colors=MUTED,labelsize=10)
    fig.text(.5,.026,"Mais claro = maior peso. Cada linha soma 1. SP = espaço. Peso não é confiança.",ha='center',color=MUTED,fontsize=15)
    fig.savefig('04-mapa-real.png'); plt.close(fig)

def curve():
    with open('metricas.csv') as fcsv:
        m=list(csv.DictReader(fcsv))
    fig=plt.figure(figsize=(14.4,7.5),dpi=100,facecolor=BG)
    fig.text(.044,.935,"PYTORCH  /  ROBÓTICA  /  04",color=BERRY,fontsize=16,fontweight="bold")
    fig.text(.043,.828,"O que aconteceu na validação?",color=INK,fontsize=36,fontweight="bold")
    fig.text(.045,.760,"CER por época • menor é melhor • mesmas linhas compostas, uma semente",color=MUTED,fontsize=16)
    ax=fig.add_axes([.095,.158,.844,.53],facecolor=CREAM)
    for rec,c in [('gru',TEAL),('lstm',BERRY)]:
        data=[r for r in m if r['encoder']==rec]
        xx=[int(i['epoca']) for i in data]; yy=[100*float(i['cer_val']) for i in data]
        ax.plot(xx,yy,label='Bi'+rec.upper(),color=c,lw=3,marker='o')
    ax.set_xlabel('Época',color=MUTED,fontsize=15);ax.set_ylabel('CER (%)',color=MUTED,fontsize=15)
    ax.tick_params(colors=MUTED,labelsize=12);ax.grid(color=LINE,alpha=.7)
    ax.set_xticks(xx)
    for spine in ax.spines.values():spine.set_color(LINE)
    ax.legend(frameon=False,fontsize=14)
    fig.text(.045,.047,"O checkpoint do teste foi selecionado pelo menor CER de validação de cada encoder.",color=MUTED,fontsize=14)
    fig.savefig('05-cer-validacao.png');plt.close(fig)

if __name__=='__main__':
    flux()
    gates()
    what()
    if Path("figura-exemplo.json").is_file():
        heat()
        curve()

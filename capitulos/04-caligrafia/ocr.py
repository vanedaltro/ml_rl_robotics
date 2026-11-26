"""OCR didático de linhas sintéticas feitas com traços humanos UJI Pen Characters 2."""
import argparse
import csv
import json
import random
import string
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

UCI = "https://archive.ics.uci.edu/static/public/177/uji+pen+characters+version+2.zip"
ALFABETO = string.ascii_uppercase + string.digits + " "
PAD, BOS, EOS = 0, 1, 2
ID = {ch: i + 3 for i, ch in enumerate(ALFABETO)}
VOC = len(ALFABETO) + 3
H, W, PASSOS = 24, 176, 8


def uji(arquivo):
    if not arquivo.is_file():
        urlretrieve(UCI, arquivo)
    with ZipFile(arquivo) as z:
        raw = z.read("ujipenchars2.txt").decode("utf-8-sig")
    out = defaultdict(lambda: defaultdict(list))
    autor = ch = None
    strokes = []
    for line in raw.splitlines() + ["WORD EOF none"]:
        parts = line.strip().split()
        if parts and parts[0] == "WORD":
            if autor and strokes:
                out[autor][ch].append(strokes)
            _, ch, session = parts[:3]
            autor = session.rsplit("-", 1)[0] if ch in ID else None
            strokes = []
        elif parts and parts[0] == "POINTS" and autor:
            pos = [int(x) for x in parts[3:]]
            if len(pos) != 2 * int(parts[1]):
                raise ValueError("Quantidade de coordenadas inválida.")
            strokes.append(list(zip(pos[::2], pos[1::2])))
    out = {author: data for author, data in out.items()
           if all(data.get(ch) for ch in ALFABETO[:-1])}
    return out


def rasterizar(strokes, rng):
    xy = list(p for stroke in strokes for p in stroke)
    x0, y0 = min(x for x, y in xy), min(y for x, y in xy)
    dx = max(1, max(x for x, y in xy) - x0)
    dy = max(1, max(y for x, y in xy) - y0)
    size_y = rng.uniform(15, 20)
    size_x = min(24., max(3., dx * size_y / dy * rng.uniform(.88, 1.12)))
    w = int(size_x + 5)
    im = Image.new("L", (w*3, H*3))
    d = ImageDraw.Draw(im)
    top = (H-size_y)/2 + rng.uniform(-1.3,1.3)
    thick = rng.choice((4,5,6))
    for stroke in strokes:
        line = [(round((2 + (x - x0) * size_x / dx)*3),
                 round((top + (y - y0) * size_y / dy)*3)) for x,y in stroke]
        d.line(line, fill=255, width=thick, joint="curve")
        a,b = line[0]
        q = thick//2
        d.ellipse((a-q,b-q,a+q,b+q), fill=255)
    return im.resize((w,H),Image.Resampling.LANCZOS)


def montar(fontes, autores, n, seed, exemplos=()):
    rng = random.Random(seed)
    xs = np.zeros((n,1,H,W), np.uint8)
    lengths = []
    tokens = torch.full((n, PASSOS+1), PAD, dtype=torch.long)
    textos = []
    for i in range(n):
        # A linha inteira usa amostras do mesmo escritor.
        author = rng.choice(autores)
        if i < len(exemplos):
            texto = exemplos[i]
        else:
            m = rng.randint(2,5)
            glyphs = rng.choices(ALFABETO[:-1],k=m)
            if m >= 3 and rng.random() < .5:
                glyphs.insert(rng.randrange(1,m)," ")
            texto = "".join(glyphs)
        x = rng.randint(2,5)
        for ch in texto:
            if ch == " ":
                x += rng.randint(9,14)
                continue
            glyph = rasterizar(rng.choice(fontes[author][ch]), rng)
            a = np.asarray(glyph)
            end=x+a.shape[1]
            if end>W:
                raise ValueError("Linha maior que o canvas.")
            xs[i,0,:,x:end] = a
            x=end+rng.randint(0,2)
        lengths.append(min(W,x+3))
        ids = [BOS]+[ID[c] for c in texto]+[EOS]
        tokens[i,:len(ids)] = torch.tensor(ids)
        textos.append(texto)
    return TensorDataset(torch.from_numpy(xs),torch.tensor(lengths),tokens), textos


class Atencao(nn.Module):
    def __init__(self, d=96, hidden=96, a=64):
        super().__init__()
        self.chave = nn.Linear(d,a,bias=False)
        self.consulta = nn.Linear(hidden,a,bias=False)
        self.score = nn.Linear(a,1,bias=False)

    def forward(self, enc, keys, h, valido):
        e = self.score(torch.tanh(keys + self.consulta(h)[:,None,:])).squeeze(-1)
        e = e.masked_fill(~valido, float("-inf"))
        alpha = e.softmax(dim=-1)
        ctx = torch.bmm(alpha[:,None,:], enc).squeeze(1)
        return ctx, alpha


class OCR(nn.Module):
    def __init__(self, rec="gru"):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,24,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(24,32,3,stride=2,padding=1),nn.ReLU(),
        )
        self.features = nn.Linear(32*(H//4),64)
        self.enc = (nn.GRU if rec=="gru" else nn.LSTM)(
            64,48,bidirectional=True,batch_first=True)
        self.inicio = nn.Linear(96,96)
        self.atencao = Atencao()
        self.emb = nn.Embedding(VOC,32,padding_idx=PAD)
        self.dec = nn.GRUCell(32+96,96)
        self.head = nn.Linear(96+96,VOC)

    def encode(self, x, larguras):
        z = self.cnn(x).permute(0,3,1,2).flatten(2)
        z = self.features(z)
        ts = (larguras + 3)//4
        pack = pack_padded_sequence(z, ts.cpu(), batch_first=True,enforce_sorted=False)
        ye, _ = self.enc(pack)
        e, _ = pad_packed_sequence(ye,batch_first=True,total_length=z.size(1))
        valido = torch.arange(e.size(1),device=e.device)[None,:] < ts.to(e.device)[:,None]
        mean = e.sum(1) / ts.to(e.device)[:,None]
        return e,self.atencao.chave(e),valido,self.inicio(mean).tanh()

    def step(self, token, h, enc, keys, mask):
        ctx, alpha = self.atencao(enc,keys,h,mask)
        h = self.dec(torch.cat((self.emb(token),ctx),-1),h)
        return self.head(torch.cat((h,ctx),-1)), h, alpha

    def forward(self, x, widths, prefix):
        e,k,mask,h=self.encode(x,widths)
        logits=[]
        for prev in prefix.unbind(1):
            out,h,_=self.step(prev,h,e,k,mask)
            logits.append(out)
        return torch.stack(logits,1)

    @torch.inference_mode()
    def ler(self,x,widths):
        e,k,mask,h=self.encode(x,widths)
        prev=torch.full((x.size(0),),BOS,dtype=torch.long,device=x.device)
        seq,weights=[],[]
        ended=torch.zeros_like(prev,dtype=torch.bool)
        for _ in range(PASSOS):
            y,h,a = self.step(prev,h,e,k,mask)
            y[:,[PAD,BOS]]=float("-inf")
            prev = torch.where(ended, EOS, y.argmax(-1))
            ended |= prev.eq(EOS)
            seq.append(prev)
            weights.append(a)
            if ended.all():
                break
        return torch.stack(seq,1),torch.stack(weights,1)


def decodificar(ids):
    out=[]
    for token in ids:
        if token==EOS:
            break
        if token>=3:
            out.append(ALFABETO[token-3])
    return "".join(out)


def dist(a,b):
    row=list(range(len(b)+1))
    for i, x in enumerate(a,1):
        new=[i]
        for j, y in enumerate(b,1):
            new.append(min(row[j]+1,new[j-1]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]


@torch.inference_mode()
def avaliar(model, dados):
    model.eval()
    errors=total=exact=count=0
    for x,w,t in DataLoader(dados,batch_size=128):
        out,_ = model.ler(x.float()/255,w)
        for o,g in zip(out.tolist(),t[:,1:].tolist()):
            pred,ref=decodificar(o),decodificar(g)
            errors+=dist(ref,pred)
            total+=len(ref)
            exact+=pred==ref
            count+=1
    return errors/total,exact/count


def treino(args):
    fontes = uji(args.dados)
    tr = sorted(a for a in fontes if a.startswith("trn_"))
    te = sorted(a for a in fontes if a.startswith("tst_"))
    random.Random(92).shuffle(tr)
    va,tr=tr[:5],tr[5:]
    assert (len(tr),len(va),len(te))==(35,5,20)
    assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
    train,_=montar(fontes,tr,args.treino,100)
    val,_=montar(fontes,va,args.val,200)
    test,_=montar(fontes,te,args.teste,300)
    demos,textos=montar(fontes,te,30,800,("OLA 7","ROBO","CAFE","VOO","A22","LUA","GRU","FOCO","C3PO"))
    print("Torch",torch.__version__,"threads",args.threads,"escritores",len(tr),len(va),len(te),flush=True)
    stats=[]
    for rec in args.modelos:
        torch.manual_seed(42)
        model=OCR(rec)
        opt=torch.optim.AdamW(model.parameters(),lr=.002)
        loader=DataLoader(train,batch_size=96,shuffle=True,generator=torch.Generator().manual_seed(53))
        best=float("inf")
        for ep in range(1,args.epocas+1):
            start=time.perf_counter()
            model.train()
            los=0.
            for x,w,t in loader:
                opt.zero_grad(set_to_none=True)
                logits=model(x.float()/255,w,t[:,:-1])
                loss=F.cross_entropy(logits.transpose(1,2),t[:,1:],ignore_index=PAD)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),3.)
                opt.step()
                los+=loss.item()
            cer,exact=avaliar(model,val)
            dt=time.perf_counter()-start
            print(rec,ep,round(los/len(loader),4),round(cer,4),round(exact,4),round(dt,2),flush=True)
            stats.append([rec,ep,los/len(loader),cer,exact,dt])
            if cer<best:
                best,bestep=cer,ep
                torch.save(model.state_dict(),f"ocr_{rec}.pt")
        model.load_state_dict(torch.load(f"ocr_{rec}.pt",weights_only=True))
        cer,exact=avaliar(model,test)
        one=next(iter(DataLoader(demos,batch_size=len(demos))))
        xs,ws,ts=one
        out,alpha=model.ler(xs.float()/255,ws)
        records=[]
        Path("amostras").mkdir(exist_ok=True)
        for i,(o,a,w) in enumerate(zip(out.tolist(),alpha,ws.tolist())):
            p=Path("amostras")/f"linha-{i:02}.png"
            Image.fromarray(xs[i,0,:,:w].numpy()).save(p)
            records.append({"ref":textos[i],"pred":decodificar(o),"arquivo":str(p),
                            "tokens":o,"alpha":a[:,:((w+3)//4)].tolist()})
        result={"modelo":rec,"melhor_epoca":bestep,"cer_teste":cer,"exata_teste":exact,
                "parametros":sum(x.numel() for x in model.parameters()),"demos":records,
                "treino":args.treino,"val":args.val,"teste":args.teste,"torch":torch.__version__}
        Path(f"resultado-{rec}.json").write_text(json.dumps(result))
        print("TESTE",rec,bestep,cer,exact,flush=True)
    with open("metricas.csv","w",newline="") as f:
        wr=csv.writer(f)
        wr.writerow(["encoder","epoca","loss","cer_val","exata_val","segundos_treino_e_val"])
        wr.writerows(stats)


def inferir(args):
    im=Image.open(args.imagem).convert("L")
    if args.linha_lab:
        if im.height!=H or im.width>W:
            raise ValueError("A linha do laboratório deve ter 24 px de altura e até 176 px de largura.")
    else:
        if np.median(np.asarray(im)[:,0])>127:
            im=ImageOps.invert(im)
        box=im.point(lambda x: 255 if x>45 else 0).getbbox()
        if box is None:
            raise ValueError("Nenhum traço detectado.")
        im=im.crop(box)
        scale=min(19/im.height,(W-8)/im.width)
        im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
        bg=Image.new("L",(im.width+8,H))
        bg.paste(im,(4,(H-im.height)//2))
        im=bg
    x=torch.zeros(1,1,H,W)
    x[0,0,:,:im.width]=torch.from_numpy(np.asarray(im).copy()).float()/255
    model=OCR(args.encoder).eval()
    path=args.pesos or Path(f"ocr_{args.encoder}.pt")
    model.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
    seq,_=model.ler(x,torch.tensor([im.width]))
    texto=decodificar(seq[0].tolist())
    print(texto)
    if args.saida:
        args.saida.write_bytes((texto+"\n").encode("ascii",errors="strict"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dados",type=Path,default=Path("uji.zip"))
    p.add_argument("--modelos",nargs="+",choices=["gru","lstm"],default=["gru","lstm"])
    p.add_argument("--epocas",type=int,default=10)
    p.add_argument("--treino",type=int,default=10000)
    p.add_argument("--val",type=int,default=1000)
    p.add_argument("--teste",type=int,default=1200)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--imagem",type=Path)
    p.add_argument("--linha-lab",action="store_true")
    p.add_argument("--encoder",choices=["gru","lstm"],default="gru")
    p.add_argument("--pesos",type=Path)
    p.add_argument("--saida",type=Path)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    inferir(a) if a.imagem else treino(a)

if __name__=="__main__":
    main()

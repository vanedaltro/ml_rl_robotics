"""Mini LM causal: dados sintéticos próprios, treino e avaliação CPU."""
import copy
import itertools
import json
import math
import random
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20261126
NAMES = 'ana bia caio dani eva fabio gabi hugo iris joao lia nino'.split()
GENERAL = ('le observa organiza revisa'.split(), 'livro mapa texto jornal caderno guia artigo roteiro'.split(), 'sala casa escola parque'.split())
DOMAIN = ('mede testa calibra inspeciona'.split(), 'sensor circuito motor bateria placa cabo antena rele'.split(), 'oficina bancada fabrica laboratorio'.split())
# Vocabulário declarado pela especificação do gerador, não aprendido no teste.
VOCAB = ['<bos>', '<eos>'] + sorted(set(NAMES + sum((sum((list(x) for x in d), []) for d in [GENERAL, DOMAIN]), [])))
STOI = {s: i for i, s in enumerate(VOCAB)}

def documents(domain):
    verbs, objects, places = domain
    return [' '.join(row) for row in itertools.product(NAMES, verbs, objects, places)]

def split(domain, seed):
    docs = documents(domain)
    random.Random(seed).shuffle(docs)
    n = len(docs)
    return docs[:int(n*.7)], docs[int(n*.7):int(n*.85)], docs[int(n*.85):]

def encode(docs):
    return torch.tensor([[STOI[t] for t in ['<bos>'] + d.split() + ['<eos>']] for d in docs])

def xy(batch):
    return batch[:, :-1], batch[:, 1:]

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3*dim)
        self.out = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.h, d//self.h).transpose(1, 2) for z in [q,k,v]]
        scores = q @ k.transpose(-2,-1) / math.sqrt(d//self.h)
        # True = futuro proibido nesta API masked_fill.
        future = torch.ones(t,t,dtype=torch.bool,device=x.device).triu(1)
        scores = scores.masked_fill(future, float('-inf'))
        attention = scores.softmax(dim=-1) @ v
        x = x + self.out(attention.transpose(1,2).contiguous().view(b,t,d))
        return x + self.ff(self.ln2(x))

class MiniLM(nn.Module):
    def __init__(self, dim=48, heads=4, layers=2, context=8):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(len(VOCAB), dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(VOCAB))

    def forward(self, ids):
        assert ids.ndim == 2 and 0 < ids.shape[1] <= self.context
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        return self.head(self.norm(self.blocks(x)))

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    total, tokens = 0., 0
    for batch in data.split(128):
        x,y = xy(batch)
        total += F.cross_entropy(model(x).reshape(-1,len(VOCAB)), y.reshape(-1), reduction='sum').item()
        tokens += y.numel()
    loss = total/tokens
    return {'nll':loss, 'ppl':math.exp(loss), 'tokens':tokens}

def train(model, data, val, steps, lr, seed):
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    best, best_state, best_step, trace = float('inf'), None, 0, []
    for step in range(1,steps+1):
        model.train()
        batch = data[torch.randint(len(data), (64,), generator=rng)]
        x,y = xy(batch)
        loss = F.cross_entropy(model(x).reshape(-1,len(VOCAB)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            val_loss = evaluate(model,val)['nll']
            trace.append({'step':step,'train_batch_nll':loss.item(),'val_nll':val_loss})
            if val_loss < best:
                best, best_state, best_step = val_loss, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best_state)
    return {'best_step':best_step, 'trace':trace}

@torch.no_grad()
def generate(model, prompt, sample=False, temperature=1., max_new=6, seed=7):
    if temperature <= 0:
        raise ValueError('temperature deve ser positiva')
    model.eval()
    ids = [STOI['<bos>']] + [STOI[t] for t in prompt.split()]
    rng = torch.Generator().manual_seed(seed)
    for _ in range(max_new):
        # Janela deslizante: posições reiniciam; aqui frases cabem em context.
        logits = model(torch.tensor([ids[-model.context:]]))[0,-1]
        nxt = torch.multinomial((logits/temperature).softmax(-1),1,generator=rng).item() if sample else logits.argmax().item()
        ids.append(nxt)
        if nxt == STOI['<eos>']: break
    return ' '.join(VOCAB[i] for i in ids)

def unigram(train_data, test_data):
    counts = torch.bincount(train_data[:,1:].reshape(-1),minlength=len(VOCAB)).double() + 1
    p = counts/counts.sum()
    nll = -p[test_data[:,1:]].log().mean().item()
    return {'nll':nll,'ppl':math.exp(nll)}

def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    gs = split(GENERAL,SEED)
    ds = split(DOMAIN,SEED+1)
    gt,gv,gtest = map(encode,gs)
    dt,dv,dtest = map(encode,ds)
    model = MiniLM()
    stages = {'pretrain':train(model,gt,gv,300,3e-3,SEED)}
    pretrained = copy.deepcopy(model)
    stages['finetune'] = train(model,dt,dv,150,1e-3,SEED+1)
    torch.manual_seed(SEED)
    scratch = MiniLM()
    stages['scratch_domain'] = train(scratch,dt,dv,150,1e-3,SEED+1)
    # Testes entram somente aqui, depois de todas as escolhas por validação.
    metrics = {'general_pretrain':evaluate(pretrained,gtest), 'general_after_finetune':evaluate(model,gtest),
               'domain_pretrain':evaluate(pretrained,dtest), 'domain_finetune':evaluate(model,dtest),
               'domain_scratch_150':evaluate(scratch,dtest), 'domain_unigram':unigram(dt,dtest)}
    result = {'seed':SEED,'torch':torch.__version__,'device':'cpu','vocab':len(VOCAB),
              'parameters':sum(p.numel() for p in model.parameters()), 'split_documents':[len(s) for s in ds],
              'metrics':metrics,'stages':stages,'elapsed_seconds':time.perf_counter()-started,
              'samples':{'greedy':generate(model,'ana testa'),
                         'temperature_0.7':generate(model,'ana testa',True,.7),
                         'temperature_1.2':generate(model,'ana testa',True,1.2)}}
    Path('results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    Path('splits.json').write_text(json.dumps({'general':gs,'domain':ds},ensure_ascii=False))
    torch.save({'state_dict':model.state_dict(),'vocab':VOCAB},'model.pt')
    print(json.dumps({k:v for k,v in result.items() if k != 'stages'},indent=2,ensure_ascii=False))

if __name__ == '__main__': main()

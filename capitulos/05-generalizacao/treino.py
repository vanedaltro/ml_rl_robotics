"""Experimento CPU: FashionMNIST com 20% de rótulos corrompidos no treino."""
import argparse, csv, copy, platform
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

CASES = {
    'pequena': (32, 1, 0., 0., 0., 600),
    'grande': (256, 3, 0., 0., 0., 600),
    'dropout': (256, 3, .35, 0., 0., 600),
    'adamw': (256, 3, 0., .05, 0., 600),
    'smoothing': (256, 3, 0., 0., .10, 600),
    'mais_dados': (256, 3, 0., 0., 0., 3000),
}
def model(width, depth, p):
    layers = [nn.Flatten()]
    for i in range(depth):
        layers += [nn.Linear(784 if i == 0 else width, width), nn.ReLU(), nn.Dropout(p)]
    return nn.Sequential(*layers, nn.Linear(width, 10))

@torch.inference_mode()
def score(net, x, y):
    net.eval()
    logits = torch.cat([net(z) for z in x.split(256)])
    return torch.nn.functional.cross_entropy(logits, y).item(), (logits.argmax(1)==y).float().mean().item()

def run(args):
    torch.set_num_threads(args.threads)
    train = datasets.FashionMNIST(args.data, train=True, download=args.download)
    test = datasets.FashionMNIST(args.data, train=False, download=args.download)
    idx = torch.randperm(len(train), generator=torch.Generator().manual_seed(2026))
    vi, ti = idx[:3000], idx[3000:6000]
    vx, vy = train.data[vi].float().div_(255), train.targets[vi]
    tx, ty = test.data.float().div_(255), test.targets
    hist, final = [], []
    print(platform.processor(), torch.__version__, args, flush=True)
    for seed in args.seeds:
        g = torch.Generator().manual_seed(seed + 10000)
        noise_order = torch.randperm(600, generator=g)[:120]
        noise_extra = torch.randperm(2400, generator=g)[:480] + 600
        flip = torch.randint(1, 10, (600,), generator=g)
        for name, (width,depth,p,wd,smooth,n) in CASES.items():
            torch.manual_seed(seed)
            x, y = train.data[ti[:n]].float().div_(255), train.targets[ti[:n]].clone()
            noise = noise_order if n==600 else torch.cat([noise_order,noise_extra])
            y[noise] = (y[noise] + flip[:len(noise)]) % 10
            dl = DataLoader(TensorDataset(x,y),batch_size=64,shuffle=True,
                            generator=torch.Generator().manual_seed(seed + 20000))
            net = model(width,depth,p)
            opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=wd)
            criterion = nn.CrossEntropyLoss(label_smoothing=smooth)
            best = float('inf')
            for epoch in range(1,args.epochs+1):
                net.train()
                for xb,yb in dl:
                    opt.zero_grad(set_to_none=True)
                    loss=criterion(net(xb),yb)
                    loss.backward()
                    opt.step()
                tl,ta=score(net,x,y); vl,va=score(net,vx,vy)
                hist.append(dict(seed=seed,caso=name,epoca=epoch,treino_nll=tl,
                                 treino_acc=ta,val_nll=vl,val_acc=va))
                if vl < best:
                    best, best_ep, checkpoint, best_va = vl, epoch, copy.deepcopy(net.state_dict()), va
            last_test = score(net,tx,ty)
            net.load_state_dict(checkpoint)
            best_test = score(net,tx,ty)
            final.append(dict(seed=seed,caso=name,n=n,parametros=sum(p.numel() for p in net.parameters()),
                              ultima_epoca=args.epochs,treino_acc_final=ta,val_acc_final=va,
                              val_nll_final=vl,test_nll_final=last_test[0],test_acc_final=last_test[1],
                              melhor_epoca=best_ep,val_acc_melhor=best_va,val_nll_melhor=best,
                              test_nll_melhor=best_test[0],test_acc_melhor=best_test[1]))
            print(final[-1],flush=True)
            for fn,rows in [('curvas.csv',hist),('resultados.csv',final)]:
                with (Path(args.out)/fn).open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='./dados')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--out',default='.')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,23,42])
    args=parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    run(args)

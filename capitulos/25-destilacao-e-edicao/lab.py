"""Destilação de um MLP em CPU, sem serviços externos. Python 3 + PyTorch."""
import json
import statistics
import torch
from torch import nn
from torch.nn import functional as F

def data(n, seed):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(6, (n,), generator=g)
    a = 2 * torch.pi * y / 6
    # As classes vizinhas se sobrepõem; 18 dimensões são ruído.
    x = torch.randn(n, 20, generator=g) * 0.60
    x[:, 0] += 2.0 * a.cos()
    x[:, 1] += 2.0 * a.sin()
    return x, y

def net(width):
    return nn.Sequential(nn.Linear(20, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 6))

def kd_loss(z, y, teacher_z, temperature=3., alpha=0.75):
    soft = F.kl_div(F.log_softmax(z / temperature, dim=-1),
                    F.softmax(teacher_z.detach() / temperature, dim=-1),
                    reduction="batchmean")
    return (1 - alpha) * F.cross_entropy(z, y) + alpha * temperature**2 * soft

def fit(model, x, y, teacher_z=None, mode="labels", steps=160):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006)
    for _ in range(steps):
        z = model(x)
        if mode == "soft":
            loss = kd_loss(z, y, teacher_z)
        elif mode == "hard":
            loss = (.25 * F.cross_entropy(z, y) +
                    .75 * F.cross_entropy(z, teacher_z.argmax(dim=-1)))
        else:
            loss = F.cross_entropy(z, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()

@torch.no_grad()
def score(model, x, y):
    return 100 * (model(x).argmax(dim=-1) == y).double().mean().item()

def run(seed):
    x, y = data(4096, 100 + seed)
    xs, ys = data(48, 200 + seed)
    xt, yt = data(2048, 300 + seed)
    torch.manual_seed(seed)
    teacher = fit(net(64), x, y, steps=240)
    with torch.no_grad():
        tz = teacher(xs)
    results = {"professor": score(teacher, xt, yt)}
    for mode in ("labels", "hard", "soft"):
        torch.manual_seed(seed + 1000)
        student = fit(net(16), xs, ys, tz, mode=mode)
        results[mode] = score(student, xt, yt)
    return results

@torch.no_grad()
def edit_demo():
    """Analogia: canal inútil conocido, no un LLM editándose."""
    model = nn.Linear(2, 1, bias=False)
    model.weight.copy_(torch.tensor([[1., 3.]]))
    x = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    before = (model(x).flatten() > 0).tolist()
    original = model.weight.clone()
    model.weight[0, 1] = 0.
    after = (model(x).flatten() > 0).tolist()
    model.weight.copy_(original)
    assert (model(x).flatten() > 0).tolist() == before
    return {"original": before, "editado": after, "esperado": [True, True, False, False]}

if __name__ == "__main__":
    torch.set_num_threads(1)
    runs = {s: run(s) for s in (7, 23, 42)}
    metrics = {k: {"media_pct": statistics.mean(r[k] for r in runs.values()),
                   "dp_amostral_pp": statistics.stdev(r[k] for r in runs.values())}
               for k in next(iter(runs.values()))}
    print(json.dumps({"torch": torch.__version__, "seeds": runs, "metricas": metrics,
                      "edicao_didatica": edit_demo()}, indent=2, ensure_ascii=False))

"""Uma CNN, exposição fora do treino e classificação seletiva."""
from pathlib import Path
import json
import math
import torch
from torch import nn
import torch.nn.functional as F

SEEDS = (7, 23, 42)
CLASSES = ("vertical", "horizontal", "diagonal", "anel")
EXPOSURES = {"claro": (1., .035), "penumbra": (.42, .07), "escuro": (.10, .13)}
GRID = [i / 100 for i in range(25, 101)]
HIGH = .90

def images(n, seed, unknown=False):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(4, (n,), generator=g)
    xx = torch.arange(20).view(1, 1, 20)
    yy = torch.arange(20).view(1, 20, 1)
    cx, cy = (torch.randint(7, 13, (2, n, 1, 1), generator=g))
    dx, dy = xx - cx, yy - cy
    t = torch.rand(n, 1, 1, generator=g) * .75 + .65
    patterns = torch.stack([
        (dx.abs() < t) & (dy.abs() <= 5),
        (dy.abs() < t) & (dx.abs() <= 5),
        ((dx-dy).abs() < 1.4*t) & (dx.abs() <= 4) & (dy.abs() <= 4),
        ((dx.square()+dy.square()).sqrt() - 4).abs() < t,
    ], dim=1)
    masks = patterns[torch.arange(n), y]
    if unknown:
        masks = ((dx.abs() < t) & (dy.abs() <= 5)) | ((dy.abs() < t) & (dx.abs() <= 5))
    gain = .65 + .35 * torch.rand(n, 1, 1, generator=g)
    base = .05 + gain * masks
    return base[:, None], (torch.full_like(y, -1) if unknown else y)

def expose(base, lighting, seed):
    level, sigma = EXPOSURES[lighting]
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(base.shape, generator=g)
    return (level * base + sigma * noise).clamp(0, 1)

def model():
    return nn.Sequential(nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Flatten(), nn.Linear(24*5*5, 4))

def train(x, y, seed):
    torch.manual_seed(seed)
    m = model()
    opt = torch.optim.AdamW(m.parameters(), lr=.002)
    for _ in range(8):
        for idx in torch.randperm(len(y)).split(64):
            loss = F.cross_entropy(m(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()

def fit_temperature(z, y):
    # Uma grade explícita evita correr até T=0 em validação separável.
    z = z.detach().double()
    grid = torch.linspace(math.log(.25), math.log(4.), 161, dtype=z.dtype).exp()
    return min(grid.tolist(), key=lambda t: float(F.cross_entropy(z/t, y)))

def outcome(z, y, t=1.):
    conf, pred = (z / t).softmax(dim=-1).max(dim=-1)
    return conf, pred.eq(y)

def policy_stats(c, ok, tau):
    accept = c >= tau
    n = len(c)
    accepted = int(accept.sum())
    errors = int((accept & ~ok).sum())
    return {"n": n, "accepted": accepted, "accepted_errors": errors,
            "rejected_correct": int((~accept & ok).sum()),
            "rejected_errors": int((~accept & ~ok).sum()),
            "coverage": accepted / n,
            "risk": errors / accepted if accepted else None}

def select_tau(c, ok, cap=.05, min_accept=100):
    candidates = [(tau, policy_stats(c, ok, tau)) for tau in GRID]
    valid = [(tau, s) for tau, s in candidates
             if s["accepted"] >= min_accept and s["risk"] <= cap]
    return max(valid, key=lambda p: (p[1]["accepted"], -p[0]))[0] if valid else None

def reliability(c, ok, bins=10):
    b = torch.floor(c * bins).long().clamp(max=bins-1)
    rows = []
    for j in range(bins):
        mask = b == j
        n = int(mask.sum())
        if n:
            rows.append({"bin": j, "n": n, "confidence": float(c[mask].mean()),
                         "accuracy": float(ok[mask].float().mean())})
    return rows

def metrics(z, y, t):
    c, ok = outcome(z, y, t)
    rows = reliability(c, ok)
    wrong_high = (~ok) & (c >= HIGH)
    return {"accuracy": float(ok.float().mean()), "mean_confidence": float(c.mean()),
            "nll": float(F.cross_entropy(z/t, y)),
            "ece": sum(r["n"] * abs(r["confidence"]-r["accuracy"]) for r in rows) / len(c),
            "wrong_high": int(wrong_high.sum()), "n": len(c), "bins": rows}

@torch.no_grad()
def predict(m, x):
    return torch.cat([m(batch) for batch in x.split(128)])

def run(seed):
    base, y = images(2048, 1000+seed)
    m = train(expose(base, "claro", 10000+seed), y, seed)
    cal, yc = images(512, 2000+seed)
    t = fit_temperature(predict(m, expose(cal, "claro", 20000+seed)), yc)
    pol, yp = images(1024, 3000+seed)
    xpol = torch.cat([expose(pol[:512], "claro", 30000+seed),
                      expose(pol[512:], "penumbra", 31000+seed)])
    cpol, okpol = outcome(predict(m, xpol), yp, t)
    tau = select_tau(cpol, okpol)
    base_test, ytest = images(1024, 4000+seed)
    r = {"temperature": t, "tau": tau, "conditions": {}}
    r["policy_validation"] = policy_stats(cpol, okpol, tau if tau is not None else float("inf"))
    for light in EXPOSURES:
        xt = expose(base_test, light, 40000+seed)
        z = predict(m, xt)
        c, ok = outcome(z, ytest, t)
        r["conditions"][light] = {
            "raw": metrics(z, ytest, 1.), "calibrated": metrics(z, ytest, t),
            "policy": policy_stats(c, ok, tau if tau is not None else float("inf")),
            "fixed90": policy_stats(c, ok, .90),
            "curve": [policy_stats(c, ok, g) for g in GRID]}
    unknown, _ = images(512, 5000+seed, unknown=True)
    xu = expose(unknown, "claro", 50000+seed)
    cu = (predict(m, xu)/t).softmax(-1).max(-1).values
    r["unknown"] = {"n": len(cu), "accepted": int((cu >= (tau if tau is not None else float("inf"))).sum()),
                    "high": int((cu >= HIGH).sum())}
    dark_logits = predict(m, expose(base_test, "escuro", 40000+seed))
    c, ok = outcome(dark_logits, ytest, t)
    pred = dark_logits.argmax(-1)
    j = int(c.masked_fill(ok, -1.).argmax())
    r["illustrative_example"] = {
        "true": CLASSES[int(ytest[j])], "prediction": CLASSES[int(pred[j])],
        "confidence": float(c[j]),
        "frames": {lt: expose(base_test, lt, 40000+seed)[j, 0].tolist() for lt in EXPOSURES}}
    return r

def main():
    torch.set_num_threads(1)
    out = {"torch": torch.__version__, "runs": {s: run(s) for s in SEEDS}}
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()

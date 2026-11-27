"""Backpropagation explícito: NumPy no cálculo, PyTorch só na conferência."""
import json
import numpy as np

def log_softmax(z):
    # Batch na primeira dimensão e classes na última.
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

def activate(z, name):
    if name == "relu":
        return np.maximum(z, 0.)
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        u = np.exp(-np.abs(z))
        return np.where(z >= 0, 1 / (1 + u), u / (1 + u))
    raise ValueError(f"Ativação desconhecida: {name}")

def activation_grad(z, a, name):
    if name == "relu":
        return (z > 0).astype(z.dtype)  # Convenção no ponto z = 0.
    if name == "tanh":
        return 1 - a**2
    if name == "sigmoid":
        return a * (1 - a)
    raise ValueError(f"Ativação desconhecida: {name}")

def loss_and_dlogits(z, p, kind):
    if (z.ndim != 2 or z.shape != p.shape or not len(z)
        or not np.isfinite(z).all() or not np.isfinite(p).all()
        or (p < 0).any() or not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-8)):
        raise ValueError("Esperados logits finitos e distribuições [B, C].")
    lq = log_softmax(z)
    if kind == "ce":
        terms = -p * lq
    elif kind == "kl":
        logp = np.log(np.where(p > 0, p, 1.))  # 0 log 0 = 0 por limite.
        terms = p * (logp - lq)
    else:
        raise ValueError(f"Loss desconhecida: {kind}")
    return float(terms.sum(axis=1).mean()), (np.exp(lq) - p) / len(p)

def forward(x, par, activation):
    z1 = x @ par["W1"].T + par["b1"]
    a1 = activate(z1, activation)
    z2 = a1 @ par["W2"].T + par["b2"]
    return z2, (z1, a1)

def loss_and_grads(x, p, par, activation="relu", kind="ce"):
    z2, (z1, a1) = forward(x, par, activation)
    loss, dz2 = loss_and_dlogits(z2, p, kind)
    dw2, db2 = dz2.T @ a1, dz2.sum(axis=0)
    da1 = dz2 @ par["W2"]
    dz1 = da1 * activation_grad(z1, a1, activation)
    dw1, db1 = dz1.T @ x, dz1.sum(axis=0)
    dx = dz1 @ par["W1"]
    return loss, {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2, "X": dx}

def fixture():
    x = np.array([[1., -2.], [.5, 1.], [-1., .75]])
    p = np.eye(3)[[2, 0, 1]]
    soft = np.array([[.10, .20, .70], [.80, .20, 0.], [.15, .70, .15]])
    par = {"W1": np.array([[.6, -.3], [-.4, .5], [.2, .2], [.9, -.5]]),
           "b1": np.array([.1, -.2, .3, .1]),
           "W2": np.array([[.6, -.2, .1, .3], [-.3, .7, .2, .1], [.2, -.2, .3, -.5]]),
           "b2": np.array([.1, -.1, .15])}
    return x, p, soft, par

def finite_differences(x, p, par, activation, kind, eps=1e-6):
    values = {**{k: v.copy() for k, v in par.items()}, "X": x.copy()}
    def evaluate():
        return loss_and_dlogits(forward(values["X"], values, activation)[0], p, kind)[0]
    grads = {}
    for key, v in values.items():
        g = np.empty_like(v)
        for i in np.ndindex(v.shape):
            old = v[i]
            v[i] = old + eps
            plus = evaluate()
            v[i] = old - eps
            minus = evaluate()
            v[i] = old
            g[i] = (plus - minus) / (2 * eps)
        grads[key] = g
    return grads

def torch_reference(x, p, par, activation, kind):
    import torch
    import torch.nn.functional as F
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    w = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in par.items()}
    tgt = torch.tensor(p, dtype=torch.float64)
    a = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
    z = F.linear(a[activation](F.linear(xt, w["W1"], w["b1"])), w["W2"], w["b2"])
    if kind == "ce":
        loss = F.cross_entropy(z, tgt)
    else:
        loss = F.kl_div(F.log_softmax(z, dim=-1), tgt, reduction="batchmean")
    loss.backward()
    return loss.item(), {k: v.grad.detach().numpy() for k,v in {**w,"X":xt}.items()}

def max_error(a, b):
    return max(float(np.abs(a[k] - b[k]).max()) for k in a)

def summarize(check_torch=True):
    x, hard, soft, par = fixture()
    variants = (("relu", "ce", hard), ("tanh", "ce", soft), ("sigmoid", "kl", soft))
    rows = []
    for activation, kind, p in variants:
        loss, g = loss_and_grads(x, p, par, activation, kind)
        new = {k: v - .1 * g[k] for k, v in par.items()}
        row = {"activation": activation, "loss": kind, "before": loss,
               "after_sgd": loss_and_grads(x, p, new, activation, kind)[0],
               "max_error_finite_difference": max_error(g, finite_differences(x,p,par,activation,kind))}
        if check_torch:
            ref, gt = torch_reference(x, p, par, activation, kind)
            row.update({"max_error_autograd": max_error(g, gt), "loss_error_autograd": abs(loss-ref)})
        rows.append(row)
    return rows

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy-only", action="store_true", help="Não importa PyTorch.")
    print(json.dumps(summarize(not parser.parse_args().numpy_only), indent=2))

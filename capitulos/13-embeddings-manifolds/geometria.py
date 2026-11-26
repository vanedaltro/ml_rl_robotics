"""Capítulo 13: comparando vetores, vizinhanças e distribuições."""
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance, energy_distance

DTYPE = torch.float64

def vector_demo():
    query = torch.tensor([[1., 0.]], dtype=DTYPE)
    candidates = torch.tensor([[5., 0.], [.8, .2], [1., .6]], dtype=DTYPE)
    cosine = F.cosine_similarity(query, candidates, dim=-1)
    l2 = torch.cdist(query, candidates).squeeze(0)
    unit_q, unit_x = F.normalize(query), F.normalize(candidates)
    normalized_l2_sq = torch.cdist(unit_q, unit_x).square().squeeze(0)
    torch.testing.assert_close(normalized_l2_sq, 2 * (1 - cosine))
    assert cosine.argmax().item() == 0 and l2.argmin().item() == 1
    return query[0], candidates, cosine, l2

def probability_demo():
    p = torch.tensor([[0.8, 0.2]], dtype=DTYPE)
    q = torch.tensor([[0.5, 0.5]], dtype=DTYPE)
    # KL(P || Q): input = log(Q), target = P; each row is a distribution.
    kl_pq = F.kl_div(q.log(), p, reduction="batchmean")
    kl_qp = F.kl_div(p.log(), q, reduction="batchmean")
    m = (p + q) / 2
    js = (F.kl_div(m.log(), p, reduction="batchmean")
          + F.kl_div(m.log(), q, reduction="batchmean")) / 2
    tv = 0.5 * (p - q).abs().sum()
    hellinger = (p.sqrt() - q.sqrt()).norm() / math.sqrt(2)
    torch.testing.assert_close(kl_pq, (p * (p.log() - q.log())).sum())
    assert not torch.isclose(kl_pq, kl_qp)
    assert 0 <= js <= math.log(2)
    return p[0], q[0], kl_pq, kl_qp, js, tv, hellinger

def manifold_demo():
    # Uma curva 1D em R² com duas voltas; a ordem t é conhecida no simulador.
    t = torch.linspace(1, 1 + 4 * math.pi, 1201, dtype=DTYPE)
    xy = torch.stack((t * t.cos(), t * t.sin()), dim=1)
    a, b = 180, 780  # Uma volta completa de diferença.
    chord = (xy[b] - xy[a]).norm()
    arc = (xy[a+1:b+1] - xy[a:b]).norm(dim=1).sum()
    g = torch.Generator().manual_seed(13)
    basis, _ = torch.linalg.qr(torch.randn(64, 2, dtype=DTYPE, generator=g))
    embeddings = xy @ basis.T
    torch.testing.assert_close((embeddings[b] - embeddings[a]).norm(), chord)
    assert arc > chord
    return xy, a, b, chord, arc

def mmd2_rbf(x, y, sigma=1.0):
    """Estimador V-statistic (inclui diagonais); custo quadrático."""
    if sigma <= 0:
        raise ValueError("sigma deve ser positivo")
    def k(a, b):
        return torch.exp(-torch.cdist(a, b).square() / (2 * sigma**2))
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()

def sample_demo():
    near = wasserstein_distance([0.], [1.])
    far = wasserstein_distance([0.], [5.])
    energy = energy_distance([0.], [5.])
    # Mesmas marginais, dependência oposta.
    x = torch.tensor([[-1., -1.], [1., 1.]], dtype=DTYPE)
    y = torch.tensor([[-1., 1.], [1., -1.]], dtype=DTYPE)
    assert all(wasserstein_distance(x[:, j], y[:, j]) == 0 for j in (0, 1))
    mmd = mmd2_rbf(x, y)
    assert mmd > 0 and torch.isclose(mmd2_rbf(x, x), torch.tensor(0., dtype=DTYPE))
    return near, far, energy, mmd

def main():
    _, _, cos, l2 = vector_demo()
    _, _, kl, rev, js, tv, he = probability_demo()
    _, _, _, chord, arc = manifold_demo()
    near, far, energy, mmd = sample_demo()
    print(f"PyTorch {torch.__version__} | CPU | float64")
    print("Candidatos A, B, C:", "cosseno", [round(v, 4) for v in cos.tolist()],
          "L2", [round(v, 4) for v in l2.tolist()])
    print(f"Espiral: corda={chord:.4f}; arco aproximado={arc:.4f}")
    print(f"KL(P||Q)={kl:.6f}; KL(Q||P)={rev:.6f}; JS={js:.6f}; "
          f"raiz(JS)={js.sqrt():.6f}; TV={tv:.6f}; Hellinger={he:.6f}")
    print(f"Wasserstein 1D: picos em 0 e 1 => {near:.1f}; em 0 e 5 => {far:.1f}.")
    print(f"Energy (0 vs 5)={energy:.6f}; MMD² RBF (mesmas marginais)={mmd:.6f}.")
    print("Todas as verificações passaram.")

if __name__ == "__main__":
    main()

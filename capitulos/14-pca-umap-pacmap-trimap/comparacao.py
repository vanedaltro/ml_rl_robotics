"""PCA, UMAP, PaCMAP e TriMap em uma faixa enrolada e ruidosa em 48D.

Métricas comparam as mesmas amostras usadas para ajustar o mapa.
As cores representam o comprimento de arco latente; não entram no ajuste.
"""
import argparse
import csv
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path

# Limitar paralelismo antes de importar NumPy/PyTorch/Numba.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pacmap
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
import torch
import trimap
from umap import UMAP


def make_data(n=900, seed=2026):
    """A faixa de coordenadas (comprimento de arco, altura), sem usar rótulos."""
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(1.5 * np.pi, 4.5 * np.pi, 8193)
    # Integral de sqrt(1 + t²), o comprimento de arco da espiral.
    arc_grid = (t_grid * np.sqrt(1 + t_grid**2) + np.arcsinh(t_grid)) / 2
    arc = rng.uniform(arc_grid[0], arc_grid[-1], n)
    t = np.interp(arc, arc_grid, t_grid)
    h = rng.uniform(0, 21, n)
    rolled = np.column_stack((t * np.cos(t), h, t * np.sin(t)))
    basis, _ = np.linalg.qr(rng.normal(size=(48, 3)))
    x = rolled @ basis.T + rng.normal(0, 0.12, size=(n, 48))
    return x.astype(np.float32), np.column_stack((arc, h)), arc


@torch.no_grad()
def pca_torch(x, n_components=2):
    x = torch.as_tensor(x, dtype=torch.float64)
    xc = x - x.mean(dim=0)
    _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    return (xc @ vh[:n_components].T).numpy(), float(s[:n_components].square().sum() / s.square().sum())


def ranked_neighbors(d, k):
    d = np.array(d, copy=True)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def overlap(a, b):
    return float((a[:, :, None] == b[:, None, :]).any(2).mean())


def norm_cosine_pca_check():
    """Separado do benchmark: cosseno em embeddings sintéticos com espectro decrescente."""
    rng = np.random.default_rng(77)
    x = rng.normal(size=(350, 96)) * np.geomspace(1, 0.05, 96)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    idx = ranked_neighbors(cdist(x, x, metric="cosine"), 15)
    checks = []
    for d in (8, 24, 48):
        z, var = pca_torch(x, d)
        idx_z = ranked_neighbors(cdist(z, z, metric="cosine"), 15)
        checks.append({"components": d, "explained_variance": var, "cosine_overlap_k15": overlap(idx, idx_z)})
    return checks


def trimap_estimator(seed):
    return trimap.TorchTRIMAP(n_dims=2, n_inliers=12, n_outliers=4, n_random=3,
                              distance="euclidean", apply_pca=False, device="cpu",
                              knn_backend="torch", random_state=seed)


def save_plot(maps, rows, colors, dest, first_seed):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelcolor": "#DDE8F2", "text.color": "#E9F3FB"})
    fig = plt.figure(figsize=(13.6, 9.2), facecolor="#101A2A", dpi=140)
    fig.text(.05, .959, "CAPÍTULO 14  /  LABORATÓRIO", color="#7DD3C2", fontsize=12, weight="bold")
    fig.text(.05, .914, "Quatro jeitos de projetar os mesmos pontos", fontsize=27, weight="bold")
    fig.text(.05, .876, f"Faixa sintética em 48D • {len(colors)} pontos • saídas 2D • semente {first_seed}",
             fontsize=13, color="#BDD0E3")
    gs = fig.add_gridspec(2, 2, left=.05, right=.92, top=.827, bottom=.111, wspace=.13, hspace=.28)
    for i, (method, z) in enumerate(maps.items()):
        ax = fig.add_subplot(gs[i // 2, i % 2], facecolor="#17273A")
        row = next(r for r in rows if r["method"] == method and r["seed"] == first_seed)
        scatter = ax.scatter(*z.T, c=colors, cmap="viridis", s=12, alpha=.87, linewidths=0)
        ax.set_title(f"{method}   ·   vizinhos {row['overlap_k15']:.2f}   ·   ρ latente {row['rho_latent']:.2f}",
                     fontsize=13, loc="left", color="#E9F3FB", pad=10, weight="bold")
        ax.set(xticks=[], yticks=[])
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(.075)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.941, .32, .012, .325])
    cb = fig.colorbar(scatter, cax=cax)
    cb.set_ticks([])
    cb.set_label("Posição ao longo da faixa", color="#DDE8F2", labelpad=8)
    cb.outline.set_visible(False)
    fig.text(.05, .054, "Cor = posição na superfície original; nenhum algoritmo recebeu as cores. Eixos dos mapas sem unidade comparável.",
             fontsize=11, color="#BDD0E3")
    fig.savefig(dest / "04-comparacao-reducao.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 37])
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    if args.n < 80:
        p.error("--n deve ser pelo menos 80 para este protocolo")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    x, latent, colors = make_data(args.n)
    pca, ev = pca_torch(x)
    high_d = cdist(x, x)
    latent_d = cdist(latent, latent)
    k = 15
    high_knn = ranked_neighbors(high_d, k)
    u, v = np.triu_indices(args.n, k=1)
    rng = np.random.default_rng(2027)
    chosen = rng.choice(u.size, min(20000, u.size), replace=False)
    pairs = (u[chosen], v[chosen])
    ref_id = high_d[pairs]
    ref_lat = latent_d[pairs]
    rows, maps = [], {}
    for seed in args.seeds:
        methods = {
            "PCA (PyTorch)": lambda: pca.copy(),
            "UMAP": lambda: UMAP(n_components=2, n_neighbors=15, min_dist=.1,
                                 metric="euclidean", random_state=seed, n_jobs=1).fit_transform(x.copy()),
            "PaCMAP": lambda: pacmap.PaCMAP(n_components=2, n_neighbors=15,
                                           MN_ratio=.5, FP_ratio=2., distance="euclidean",
                                           apply_pca=False, random_state=seed).fit_transform(x.copy(), init="pca"),
            "TriMap": lambda: trimap_estimator(seed).fit_transform(x.copy()).detach().cpu().numpy(),
        }
        for method, fit in methods.items():
            z = fit()
            if z.shape != (args.n, 2) or not np.isfinite(z).all():
                raise RuntimeError(f"{method}: projeção inválida")
            low_d = cdist(z, z)
            row = dict(method=method, seed=seed,
                       trust_k15=trustworthiness(high_d, z, n_neighbors=k, metric="precomputed"),
                       overlap_k15=overlap(high_knn, ranked_neighbors(low_d, k)),
                       rho_ambient=float(spearmanr(ref_id, low_d[pairs]).statistic),
                       rho_latent=float(spearmanr(ref_lat, low_d[pairs]).statistic))
            rows.append(row)
            if seed == args.seeds[0]:
                maps[method] = z
            print(row, flush=True)
    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator=chr(10))
        wr.writeheader()
        wr.writerows(rows)
    detail = {"n": args.n, "seeds": args.seeds, "dataset_seed": 2026, "pair_seed": 2027,
              "pairs": len(ref_id), "pca_2d_variance": ev,
              "pacmap_signature": str(inspect.signature(pacmap.PaCMAP)),
              "trimap_signature": str(inspect.signature(trimap.TorchTRIMAP)),
              "versions": {pkg: version(pkg) for pkg in ("torch", "numpy", "scipy", "scikit-learn", "numba",
                                                          "umap-learn", "pacmap", "trimap", "matplotlib")},
              "cosine_diagnostic_separate": norm_cosine_pca_check()}
    (args.out / "metadados-benchmark.json").write_text(json.dumps(detail, indent=2) + "\n")
    save_plot(maps, rows, colors, args.out, args.seeds[0])


if __name__ == "__main__":
    main()

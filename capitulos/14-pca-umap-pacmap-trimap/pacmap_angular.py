"""Exemplo opcional: PaCMAP angular com e sem PCA externo + diagnóstico."""
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from pacmap import PaCMAP
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def mapa_pacmap(x, pre_k=None):
    x = np.asarray(x, dtype=np.float32)
    if np.any(np.linalg.norm(x, axis=1) < 1e-12):
        raise ValueError("A distância angular precisa de vetores não nulos.")
    if pre_k is not None:
        x = PCA(n_components=pre_k, random_state=42).fit_transform(x)
    # A implementação PaCMAP 0.9.1 ainda centraliza quando apply_pca=False.
    centered = x - x.mean(axis=0)
    if np.any(np.linalg.norm(centered, axis=1) < 1e-12):
        raise ValueError("Remova ou explique vetores nulos após a centralização interna.")
    return PaCMAP(
        n_components=2, n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0,
        distance="angular", apply_pca=False, random_state=42,
    ).fit_transform(x.copy())


def topk(x, k=15, metric="cosine"):
    d = cdist(x, x, metric=metric)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def overlap(a, b):
    return (a[:, :, None] == b[:, None, :]).any(-1).mean()


if __name__ == "__main__":
    rng = np.random.default_rng(77)
    x = rng.normal(size=(350, 96)) * np.geomspace(1, .05, 96)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    initial = topk(x)
    pca = PCA(n_components=50, random_state=42)
    projected = pca.fit_transform(x)
    print(f"PCA50: variância acumulada {pca.explained_variance_ratio_.sum():.3%}; "
          f"overlap cosine@15 {overlap(initial, topk(projected)):.3%}",flush=True)
    for k in (None,50):
        y = mapa_pacmap(x,k)
        assert y.shape == (len(x),2) and np.isfinite(y).all()
        print(f"PaCMAP angular · pré-PCA={k} · 2D válido · "
              f"overlap@15 contra cosseno original {overlap(initial, topk(y,metric='euclidean')):.3%}",flush=True)

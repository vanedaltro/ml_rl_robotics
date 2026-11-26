"""PCA exato: ajuste, projeção e teste com PyTorch (CPU)."""
import torch


def fit_pca(x: torch.Tensor, k: int):
    if x.ndim != 2 or x.shape[0] < 2 or not 1 <= k <= min(x.shape):
        raise ValueError("Use matriz [n, d], n >= 2 e 1 <= k <= min(n, d).")
    mean = x.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(x - mean, full_matrices=False)
    variance = s.square() / (x.shape[0] - 1)
    if variance.sum() == 0:
        raise ValueError("Dados constantes: variância explicada indefinida.")
    return mean, vh[:k].T.contiguous(), variance[:k] / variance.sum()


def transform(x: torch.Tensor, mean: torch.Tensor, components: torch.Tensor):
    return (x - mean) @ components


def main():
    g = torch.Generator().manual_seed(42)
    x = torch.randn((350, 2), dtype=torch.float64, generator=g)
    x = x @ torch.tensor([[2.4, 1.4], [.12, .50]], dtype=x.dtype)
    center, v, explained = fit_pca(x, 2)
    z = transform(x, center, v)
    recovered = z @ v.T + center
    torch.testing.assert_close(recovered, x)
    torch.testing.assert_close(v.T @ v, torch.eye(2, dtype=x.dtype))
    _, v1, fraction = fit_pca(x, 1)
    z1 = transform(x, center, v1)
    assert z1.shape == (350, 1)
    mse = ((x - (z1 @ v1.T + center)) ** 2).mean()
    query = torch.tensor([[1., 0.]], dtype=x.dtype)
    candidates = torch.tensor([[3., 0.], [1., 1.]], dtype=x.dtype)
    initial = torch.nn.functional.cosine_similarity(query, candidates)
    mu, w, _ = fit_pca(candidates, 2)
    after = torch.nn.functional.cosine_similarity(
        transform(query, mu, w), transform(candidates, mu, w))
    assert (initial.argmax().item(), after.argmax().item()) == (0, 1)
    print("PyTorch", torch.__version__)
    print(f"PC1 {fraction.item():.2%}; erro quadrático médio com PC1: {mse.item():.5f}")
    print(f"Variância explicada em 2D: {explained.sum().item():.2%}")
    print("Cosseno antes/depois da centralização:", initial.tolist(), after.tolist())
    print("Verificações de projeção e reconstrução passaram.")


if __name__ == "__main__":
    main()

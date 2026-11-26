"""Enumerate all nine weight pairs for a two-input regression."""
from itertools import product

import torch


def main():
    x = torch.tensor(list(product((-1., 1.), repeat=2)))
    y = 0.25 * (x[:, 0] - x[:, 1])
    pairs = torch.tensor(list(product((-1., 0., 1.), repeat=2)))
    errors = ((x @ pairs.T - y[:, None]) ** 2).mean(dim=0)
    i = errors.argmin()
    scaled = ((x @ torch.tensor([.25, -.25]) - y) ** 2).mean()
    print("Estrito:", pairs[i].tolist(), "MSE:", errors[i].item())
    print("Com escala:", [.25, -.25], "MSE:", scaled.item())
    assert errors[i].item() == 0.125 and scaled.item() == 0


if __name__ == "__main__":
    main()

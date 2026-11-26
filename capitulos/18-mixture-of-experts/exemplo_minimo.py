import torch
from torch import nn

class SparseTop2(nn.Module):
    def __init__(self, d=32, hidden=64, n_experts=4):
        super().__init__()
        if n_experts < 2:
            raise ValueError("Top-2 precisa de pelo menos dois experts")
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden, bias=False), nn.GELU(),
                          nn.Linear(hidden, d, bias=False))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        scores = self.router(flat).float()
        selected, expert_ids = scores.topk(2, dim=-1)
        weights = selected.softmax(dim=-1).to(flat.dtype)
        merged = torch.zeros_like(flat)

        for i, expert in enumerate(self.experts):
            token, slot = (expert_ids == i).nonzero(as_tuple=True)
            if token.numel() == 0:
                continue
            y = expert(flat.index_select(0, token))
            merged = merged.index_add(
                0, token, (y * weights[token, slot, None]).to(merged.dtype)
            )
        return merged.reshape_as(x), expert_ids

torch.manual_seed(7)
x = torch.randn(2, 5, 32)
y, routes = SparseTop2()(x)
assert y.shape == x.shape and routes.shape == (10, 2)
assert torch.bincount(routes.flatten(), minlength=4).sum() == 20

import torch
import torch.nn.functional as F

torch.manual_seed(7)
logits = torch.randn(2, 3, 32, 32, requires_grad=True)  # [N,C,H,W]
alvo = torch.randint(0, 3, (2, 32, 32))               # [N,H,W], long
ce = F.cross_entropy(logits, alvo)                   # sem softmax

p = logits.softmax(dim=1)                            # só para Dice
y = F.one_hot(alvo, 3).permute(0, 3, 1, 2).float()
dims = (0, 2, 3)
eps = 1e-6
inter = (p * y).sum(dims)
dice = (2 * inter + eps) / (p.sum(dims) + y.sum(dims) + eps)
presentes = y.sum(dims) > 0
loss = ce + (1 - dice[presentes].mean())
loss.backward()

mapa = logits.argmax(dim=1)                         # [N,H,W]
print(f"Loss: {loss.item():.3f}; mapa: {tuple(mapa.shape)}")

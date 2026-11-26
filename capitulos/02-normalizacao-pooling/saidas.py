"""Checagens de formas, losses e comportamento em BatchNorm."""
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)
x = torch.randn(4, 8)

# 3 classes exclusivas. Um índice de classe por amostra.
z = nn.Linear(8, 3)(x)
y = torch.tensor([0, 2, 1, 2], dtype=torch.long)
assert z.shape == (4, 3) and y.shape == (4,)
torch.testing.assert_close(F.cross_entropy(z, y), F.nll_loss(z.log_softmax(1), y))
F.cross_entropy(z, y).backward()
torch.testing.assert_close(z.softmax(1).sum(1), torch.ones(4))

# Binária: um logit por amostra. Multirrótulo segue a mesma ideia com [B, K].
z = nn.Linear(8, 1)(x)
y = torch.tensor([[0.], [1.], [1.], [0.]])
assert y.shape == z.shape
F.binary_cross_entropy_with_logits(z, y).backward()
assert torch.equal(z.sigmoid() >= .5, z >= 0)

zm = nn.Linear(8, 3)(x)
ym = torch.tensor([[1., 0., 1.], [0., 1., 1.], [0., 0., 0.], [1., 1., 0.]])
assert zm.shape == ym.shape
F.binary_cross_entropy_with_logits(zm, ym).backward()

# Regressores com formatos idênticos.
r = nn.Linear(8, 1)(x)
alvo = torch.randn(4, 1)
assert r.shape == alvo.shape
(F.huber_loss(r, alvo, delta=1.) + F.l1_loss(r, alvo) + F.mse_loss(r, alvo)).backward()
assert bool((F.softplus(r) > 0).all())
assert bool(((2 + (5 - 2) * r.sigmoid()) > 2).all())

# Contagens: log-taxa livre na perda; exp na inferência.
log_taxa = nn.Linear(8, 1)(x)
c = torch.tensor([[0.], [3.], [1.], [8.]])
F.poisson_nll_loss(log_taxa, c, log_input=True).backward()
assert bool((log_taxa.exp() > 0).all())

# BN: no treino a média móvel muda; em eval, mesmo com gradientes ativos, não.
bn = nn.BatchNorm2d(4)
a = torch.randn(8, 4, 5, 5)
bn.train()
antes = bn.running_mean.clone()
bn(a)
assert not torch.equal(antes, bn.running_mean)
bn.eval()
antes = bn.running_mean.clone()
bn(a + 10)
assert torch.equal(antes, bn.running_mean)
with torch.inference_mode():
    solo = bn(a[:1])
    grupo = bn(a)[:1]
torch.testing.assert_close(solo, grupo)
print("Formas, backward, logits, limites e BatchNorm validados.")

# Pooling: valores e formatos explícitos. Pooling opera por canal.
pixels = torch.tensor([[[[1., 3., 2., 0.],
                         [4., 2., 1., 5.],
                         [0., 1., 8., 2.],
                         [2., 6., 4., 11.]]]])
maximo = nn.MaxPool2d(2, stride=2)(pixels)
media = nn.AvgPool2d(2, stride=2)(pixels)
global_ = nn.AdaptiveAvgPool2d((1, 1))(pixels)
torch.testing.assert_close(maximo, torch.tensor([[[[4., 5.], [6., 11.]]]]))
torch.testing.assert_close(media, torch.tensor([[[[2.5, 2.], [2.25, 6.25]]]]))
torch.testing.assert_close(global_, torch.tensor([[[[3.25]]]]))
head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(32, 10))
assert head(torch.rand(4, 32, 7, 7)).shape == (4, 10)
assert head(torch.rand(4, 32, 5, 9)).shape == (4, 10)
print("MaxPool, AvgPool e pooling adaptativo validados.")

# LayerNorm pelos canais (em NHWC antes de retornar a NCHW).
from benchmark_cpu import LayerNormC
normalizador = LayerNormC(16)
features = torch.randn(3, 16, 5, 7)
normalizador.train()
junto = normalizador(features)
normalizador.eval()
solo = normalizador(features[:1])
torch.testing.assert_close(junto[:1], solo)
torch.testing.assert_close(junto.mean(1), torch.zeros_like(junto.mean(1)), atol=1e-6, rtol=0)
torch.testing.assert_close(junto.var(1, unbiased=False), torch.ones_like(junto.mean(1)), atol=1e-4, rtol=0)
print("LayerNorm por posição, sem dependência do lote, validada.")

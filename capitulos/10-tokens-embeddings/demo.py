import torch
from torch import nn

torch.manual_seed(7)
D = 8

# Tokens e IDs ilustrativos; não são de um modelo real.
ids = torch.tensor([[0, 1, 2, 3, 1, 4]])
vocab = nn.Embedding(5, D)
pos_texto = nn.Embedding(6, D)
t = vocab(ids)
z_texto = t + pos_texto(torch.arange(ids.size(1)))[None, :, :]

assert z_texto.shape == (1, 6, 8)
torch.testing.assert_close(t[0, 1], t[0, 4])
assert not torch.allclose(z_texto[0, 1], z_texto[0, 4])

# Imagem didática: 1 imagem, 3 canais, altura 8, largura 8.
P, C = 4, 3
imagem = torch.rand(1, C, 8, 8)
patches = nn.Unfold(kernel_size=P, stride=P)(imagem).transpose(1, 2)
proj = nn.Linear(C * P * P, D, bias=False)
tokens_imagem = proj(patches)
pos_imagem = nn.Embedding(4, D)
z_imagem = tokens_imagem + pos_imagem(torch.arange(4))[None, :, :]

# Com os mesmos pesos, uma Conv2d gera as mesmas projeções.
conv = nn.Conv2d(C, D, kernel_size=P, stride=P, bias=False)
with torch.no_grad():
    conv.weight.copy_(proj.weight.reshape(D, C, P, P))
via_conv = conv(imagem).flatten(2).transpose(1, 2)
torch.testing.assert_close(tokens_imagem, via_conv)

assert patches.shape == (1, 4, 48)
assert z_imagem.shape == (1, 4, 8)

print("IDs:", tuple(ids.shape))
print("Texto com posição:", tuple(z_texto.shape))
print("Patches brutos:", tuple(patches.shape))
print("Imagem com posição:", tuple(z_imagem.shape))
print("Projeção linear e convolução são numericamente equivalentes.")

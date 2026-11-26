import torch
import torch.nn.functional as F

q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
k = torch.tensor([[2., 0.], [1., 1.], [0., 2.]])
v = torch.tensor([[2., 0.], [0., 2.], [1., 1.]])

def manual(q, k, v, permitido=None):
    escores = q @ k.T / q.size(-1) ** 0.5
    if permitido is not None:
        if not bool(permitido.any(dim=-1).all()):
            raise ValueError("Consulta sem chave disponível")
        escores = escores.masked_fill(~permitido, float("-inf"))
    pesos = escores.softmax(dim=-1)
    return pesos, pesos @ v

pesos, z = manual(q, k, v)
print(pesos[0])  # tensor([0.5760, 0.2840, 0.1400])
print(z[0])      # tensor([1.2920, 0.7080])

# Em SDPA, True = permitido. Q/K/V: [lote, cabeça, tokens, dimensão].
causal = torch.ones(3, 3, dtype=torch.bool).tril()
validas = torch.tensor([True, True, False])
permitido = causal & validas[None, :]
pesos_m, z_m = manual(q, k, v, permitido)
assert torch.all(pesos_m[:, 2] == 0)

z_api = F.scaled_dot_product_attention(
    q[None, None], k[None, None], v[None, None],
    attn_mask=permitido, dropout_p=0.0,
)[0, 0]
torch.testing.assert_close(z_m, z_api)

v_futuro = v.clone()
v_futuro[2] = 999
_, causal_original = manual(q, k, v, causal)
_, causal_alterado = manual(q, k, v_futuro, causal)
torch.testing.assert_close(causal_original[:2], causal_alterado[:2])

import torch
from torch import nn

torch.manual_seed(7)
x = torch.randn(1, 3, 8)  # [lote, tokens, largura]
mha = nn.MultiheadAttention(8, 2, batch_first=True, dropout=0.0)
mha.eval()

# Em MultiheadAttention, True = bloqueado.
padding = torch.tensor([[False, False, True]])
with torch.no_grad():
    y, a = mha(
        x, x, x, key_padding_mask=padding,
        need_weights=True, average_attn_weights=False,
    )

assert y.shape == (1, 3, 8)
assert a.shape == (1, 2, 3, 3)  # [lote, cabeça, consulta, chave]
assert torch.all(a[..., 2] == 0)

print("Verificações de equivalência, máscaras e dimensões: OK.")

import torch
from torch import nn
from ocr import Atencao


def main():
    torch.manual_seed(42)
    x = torch.randn(2, 5, 8)
    lstm = nn.LSTM(8, 16, batch_first=True)
    gru = nn.GRU(8, 16, batch_first=True)
    y_l, (h_l, c_l) = lstm(x)
    y_g, h_g = gru(x)
    assert y_l.shape == y_g.shape == (2, 5, 16)
    assert h_l.shape == c_l.shape == h_g.shape == (1, 2, 16)
    print("Saídas LSTM/GRU:", y_l.shape, "estados:", h_l.shape)
    scores = torch.tensor([0., 2., 1., 8.])
    valido = torch.tensor([True, True, True, False])
    valores = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[9.,9.]])
    alpha = scores.masked_fill(~valido, -float("inf")).softmax(dim=-1)
    ctx = alpha @ valores
    print("Pesos", alpha.tolist(), "contexto", ctx.tolist())
    assert alpha[-1] == 0
    torch.testing.assert_close(alpha.sum(),torch.tensor(1.))

    banco = torch.randn(2, 5, 16)
    query = torch.randn(2, 1, 16)
    padding = torch.tensor([[False, False, False, True, True],
                            [False, False, False, False, True]])
    mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    contexto, pesos = mha(query,banco,banco,key_padding_mask=padding,
                           need_weights=True,average_attn_weights=False)
    assert contexto.shape==(2,1,16) and pesos.shape==(2,4,1,5)
    assert torch.count_nonzero(pesos.masked_select(padding[:,None,None,:])) == 0
    print("Multihead:",contexto.shape,pesos.shape)

    add = Atencao(d=16, hidden=16)
    c, a = add(banco, add.chave(banco), query[:,0], ~padding)
    assert c.shape==(2,16) and a.shape==(2,5)
    assert torch.count_nonzero(a[padding]) == 0
    torch.testing.assert_close(a.sum(-1), torch.ones(2))
    print("Bahdanau:", c.shape, a.shape)


if __name__=="__main__":
    main()

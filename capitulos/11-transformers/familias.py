import torch
from torch import nn
from mini_transformer import mascara_causal

torch.manual_seed(7)
torch.set_num_threads(1)
x = torch.randn(2, 4, 32)  # embeddings com posição, para isolar o bloco
x_futuro = x.clone()
x_futuro[:, -1, :] += torch.randn(2, 32)
bloco = nn.TransformerEncoderLayer(32, 4, 128, dropout=0,
                                   batch_first=True).eval()
classificador = nn.Linear(32, 3).eval()
vocabulario = nn.Linear(32, 40).eval()
with torch.inference_mode():
    # Encoder-only: contexto nos dois sentidos; mean pooling sem PAD neste lote.
    classe = classificador(bloco(x).mean(dim=1))
    assert classe.shape == (2, 3)
    assert not torch.allclose(bloco(x)[:, 0], bloco(x_futuro)[:, 0])

    # Decoder-only didático: o mesmo tipo de subcamadas, agora causais.
    # Instância separada; sem encoder externo nem atenção cruzada.
    bloco_causal = nn.TransformerEncoderLayer(32, 4, 128, dropout=0,
                                              batch_first=True).eval()
    mask = mascara_causal(x.size(1), x.device)
    logits = vocabulario(bloco_causal(x, src_mask=mask))
    alterados = vocabulario(bloco_causal(x_futuro, src_mask=mask))
    assert logits.shape == (2, 4, 40)
    torch.testing.assert_close(logits[:, :-1], alterados[:, :-1],
                               atol=1e-6, rtol=1e-5)
    print("Encoder-only:", tuple(classe.shape))
    print("Decoder-only:", tuple(logits.shape))

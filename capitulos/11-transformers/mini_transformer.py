import math
import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, EOS = 0, 1, 2


class PosicaoSenoidal(nn.Module):
    def __init__(self, d, max_len=128):
        super().__init__()
        assert d % 2 == 0
        t = torch.arange(max_len, dtype=torch.float32)[:, None]
        i = torch.arange(0, d, 2, dtype=torch.float32)
        ang = t / (10000 ** (i / d))
        pe = torch.empty(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].to(x.dtype)


def mascara_causal(t, device):
    # nn.Transformer: True bloqueia. Em SDPA funcional, True permite.
    return torch.ones(t, t, dtype=torch.bool, device=device).triu(1)


class MiniTransformer(nn.Module):
    def __init__(self, v_src=32, v_tgt=40, d=32, heads=4, layers=2):
        super().__init__()
        self.escala = math.sqrt(d)
        self.src_emb = nn.Embedding(v_src, d, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(v_tgt, d, padding_idx=PAD)
        self.pos = PosicaoSenoidal(d)
        self.emb_dropout = nn.Dropout(0.1)
        self.core = nn.Transformer(
            d_model=d, nhead=heads, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=4*d, dropout=0.1,
            activation="relu", batch_first=True, norm_first=False,
        )
        self.saida = nn.Linear(d, v_tgt)

    def forward(self, src, tgt_in, causal=True):
        src_pad, tgt_pad = src.eq(PAD), tgt_in.eq(PAD)
        x = self.emb_dropout(self.pos(self.src_emb(src) * self.escala))
        y = self.emb_dropout(self.pos(self.tgt_emb(tgt_in) * self.escala))
        mask = mascara_causal(y.size(1), y.device) if causal else None
        h = self.core(
            x, y, tgt_mask=mask,
            src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.saida(h)  # [B, T, V_tgt]: logits, sem softmax


def demo():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    m = MiniTransformer()
    src = torch.tensor([[8, 9, EOS, PAD], [6, 7, 8, EOS]])
    alvo = torch.tensor([[BOS, 11, 12, EOS, PAD],
                         [BOS, 15, 16, 17, EOS]])
    tgt_in, tgt_out = alvo[:, :-1], alvo[:, 1:]

    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    opt.zero_grad()
    logits = m(src, tgt_in)
    loss = F.cross_entropy(logits.flatten(0, 1), tgt_out.reshape(-1),
                           ignore_index=PAD)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.src_emb.weight.grad.abs().sum() > 0
    opt.step()

    m.eval()
    futuro = tgt_in.clone()
    futuro[:, 2] = 20  # Altera um token futuro, não toca nas máscaras de padding.
    with torch.inference_mode():
        a, b = m(src, tgt_in), m(src, futuro)
        torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-6, rtol=1e-5)
        c, d = m(src, tgt_in, causal=False), m(src, futuro, causal=False)
        delta = (c[:, :2] - d[:, :2]).abs().max().item()
        assert delta > 1e-4

        # Teste escalar independente do modelo: scores da 1ª consulta = [1, 2].
        q = torch.ones(1, 1, 2, 1)
        k = torch.tensor([1., 2.]).view(1, 1, 2, 1)
        vals = [torch.tensor(z).view(1, 1, 2, 1) for z in ([2., 10.], [2., 18.])]
        scores = [[F.scaled_dot_product_attention(q, k, v, is_causal=mask,
                    dropout_p=0.0)[0, 0, 0, 0].item() for v in vals]
                  for mask in (False, True)]
        torch.testing.assert_close(torch.tensor(scores[1]), torch.tensor([2., 2.]))

    print("PyTorch", torch.__version__, "| CPU")
    print("logits:", tuple(logits.shape), "| loss no passo:", round(loss.item(), 4))
    print("delta no prefixo com máscara:", (a[:, :2] - b[:, :2]).abs().max().item())
    print("delta no prefixo sem máscara:", round(delta, 5))
    print("microteste 2 tokens:", [[round(x, 5) for x in row] for row in scores])


if __name__ == "__main__":
    demo()

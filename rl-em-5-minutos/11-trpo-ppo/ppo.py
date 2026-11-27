"""PPO-Clip num GridWorld CPU e exemplo de busca em região de confiança."""
import argparse
from dataclasses import dataclass
from statistics import mean

import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from grid_ac import ACOES, BANDEIRA, COLUNAS, INICIO, LINHAS, indice, step

N_ENVS, N_STEPS, LIMITE = 8, 128, 40
GAMMA, LAMBDA, EPS = .99, .95, .2
INICIO_ID = indice(INICIO)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.corpo = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh())
        self.ator, self.critico = nn.Linear(64, len(ACOES)), nn.Linear(64, 1)
        nn.init.orthogonal_(self.ator.weight, gain=.01)
        nn.init.zeros_(self.ator.bias)

    def forward(self, estados):
        # Escala fixa para coordenadas discretas com limites conhecidos.
        x = torch.stack((estados // COLUNAS / (LINHAS - 1),
                         estados % COLUNAS / (COLUNAS - 1)), dim=-1).float()
        h = self.corpo(x)
        return Categorical(logits=self.ator(h)), self.critico(h).squeeze(-1)


class GridBatch:
    def __init__(self, n_envs, limite=LIMITE):
        self.estados = torch.full((n_envs,), INICIO_ID)
        self.passos = torch.zeros(n_envs, dtype=torch.long)
        transicoes = [[step(divmod(s, COLUNAS), a) for a in range(len(ACOES))]
                      for s in range(LINHAS * COLUNAS)]
        self.destinos = torch.tensor([[indice(t[0]) for t in linha]
                                      for linha in transicoes])
        self.recompensas = torch.tensor([[t[1] for t in linha] for linha in transicoes])
        self.terminos = torch.tensor([[t[2] for t in linha] for linha in transicoes])
        self.limite = limite

    def step(self, acoes):
        i = self.estados, acoes
        destino = self.destinos[i]
        r, terminal = self.recompensas[i], self.terminos[i]
        self.passos += 1
        truncado = (~terminal) & (self.passos >= self.limite)
        reinicia = terminal | truncado
        self.estados = torch.where(reinicia, INICIO_ID, destino)
        self.passos[reinicia] = 0
        # Para bootstrap, devolve o destino antes do reset.
        return destino, r, terminal, truncado


@dataclass
class Lote:
    estados: torch.Tensor
    acoes: torch.Tensor
    old_logp: torch.Tensor
    old_logits: torch.Tensor
    vantagem: torch.Tensor
    alvos: torch.Tensor


def gae(r, v, v_proximo, terminal, truncado, gamma=GAMMA, lam=LAMBDA):
    vantagens = torch.empty_like(r)
    acumulado = torch.zeros_like(r[0])
    for t in reversed(range(len(r))):
        delta = r[t] + gamma * v_proximo[t] * (~terminal[t]) - v[t]
        # Truncar conserva bootstrap no alvo, mas não atravessa o reset.
        acumulado = delta + gamma * lam * (~(terminal[t] | truncado[t])) * acumulado
        vantagens[t] = acumulado
    return vantagens


@torch.no_grad()
def coleta(modelo, env, passos):
    estados, acoes, logps, logits, recompensas, vals, prox_vals, terminais, truncados = (
        [] for _ in range(9))
    for _ in range(passos):
        estado = env.estados
        pi, v = modelo(estado)
        acao = pi.sample()
        destino, r, terminou, truncou = env.step(acao)
        _, v_proximo = modelo(destino)
        for serie, x in zip((estados, acoes, logps, logits, recompensas, vals,
                             prox_vals, terminais, truncados),
                            (estado, acao, pi.log_prob(acao), pi.logits, r, v,
                             v_proximo, terminou, truncou)):
            serie.append(x)
    r, v, v_proximo, m, trunc = map(torch.stack, (recompensas, vals, prox_vals,
                                                 terminais, truncados))
    vantagem = gae(r, v, v_proximo, m, trunc)
    return Lote(*(torch.cat(xs) for xs in (estados, acoes, logps, logits)),
                vantagem.flatten(), (vantagem + v).flatten())


def objetivo_clip(ratio, vantagem, eps=EPS):
    return torch.minimum(ratio * vantagem,
                         ratio.clamp(1 - eps, 1 + eps) * vantagem)


def update(modelo, opt, lote):
    n = lote.estados.numel()
    # Normalização uma vez por lote; alvo do crítico mantém a escala original.
    adv = (lote.vantagem - lote.vantagem.mean()) / (lote.vantagem.std(unbiased=False) + 1e-8)
    for _ in range(4):
        for idx in torch.randperm(n).split(128):
            pi, v = modelo(lote.estados[idx])
            ratio = (pi.log_prob(lote.acoes[idx]) - lote.old_logp[idx]).exp()
            lc = objetivo_clip(ratio, adv[idx]).mean()
            lv = (v - lote.alvos[idx]).square().mean()
            loss = -lc + .5 * lv - .01 * pi.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=.5)
            opt.step()
    with torch.no_grad():
        pi, _ = modelo(lote.estados)
        old = Categorical(logits=lote.old_logits)
        ratio = (pi.log_prob(lote.acoes) - lote.old_logp).exp()
        kl = kl_divergence(old, pi).mean()
        fracao = ((ratio - 1).abs() > EPS).float().mean()
        return kl.item(), fracao.item(), pi.entropy().mean().item()


@torch.no_grad()
def avalia(modelo, episodios=64, seed=202, guloso=False):
    # Isola a semente de avaliação do fluxo de aleatoriedade do treino.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        env = GridBatch(episodios)
        acabou = torch.zeros(episodios, dtype=torch.bool)
        chegou = acabou.clone()
        retorno = torch.zeros(episodios)
        for _ in range(LIMITE):
            pi, _ = modelo(env.estados)
            a = pi.logits.argmax(-1) if guloso else pi.sample()
            destino, r, terminal, truncado = env.step(a)
            chegou |= (~acabou) & (destino == indice(BANDEIRA))
            retorno += torch.where(acabou, 0., r)
            acabou |= terminal | truncado
            if acabou.all():
                break
    return retorno.mean().item(), chegou.float().mean().item()


def regiao_de_confianca(limite=.01):
    """Ilustra surrogado e retrocesso por KL, não implementa TRPO."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    antiga = Categorical(logits=logits.detach())
    vantagem = torch.tensor([-1., 1.], dtype=logits.dtype)
    p = Categorical(logits=logits)
    obj = (antiga.probs * (p.probs / antiga.probs) * vantagem).sum()
    d, = torch.autograd.grad(obj, logits)
    passo = 4.
    for _ in range(32):
        nova = Categorical(logits=(logits + passo * d).detach())
        kl = kl_divergence(antiga, nova).item()
        novo_obj = (nova.probs * vantagem).sum().item()
        if kl <= limite and novo_obj > obj.item():
            return passo, kl, nova.probs.tolist()
        passo /= 2
    raise RuntimeError('nenhum passo aceito')


def train(atualizacoes, seed):
    torch.manual_seed(seed)
    modelo = ActorCritic()
    inicial = avalia(modelo)
    opt = torch.optim.Adam(modelo.parameters(), lr=3e-4)
    env = GridBatch(N_ENVS)
    metricas = []
    for _ in range(atualizacoes):
        lote = coleta(modelo, env, N_STEPS)
        metricas.append(update(modelo, opt, lote))
    return modelo, inicial, metricas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--atualizacoes', type=int, default=16)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()
    if args.atualizacoes < 1:
        p.error('use ao menos uma atualização')
    torch.set_num_threads(1)
    passo, kl, probs = regiao_de_confianca()
    print(f'Região de confiança: passo={passo:.3f}, KL={kl:.5f}, '
          f'P(ação 1)={probs[1]:.4f}')
    modelo, inicial, metricas = train(args.atualizacoes, args.seed)
    for titulo, (g, sucesso) in [('início', inicial), ('fim', avalia(modelo)),
                                 ('gulosa', avalia(modelo, guloso=True))]:
        print(f'{titulo}: retorno médio={g:+.3f}; bandeira={sucesso:.1%}')
    kl, clip, entropia = metricas[-1]
    print(f'PPO: transições={args.atualizacoes * N_ENVS * N_STEPS}; '
          f'KL final={kl:.5f}; clip fraction={clip:.3f}; entropia={entropia:.3f}')


if __name__ == '__main__':
    main()

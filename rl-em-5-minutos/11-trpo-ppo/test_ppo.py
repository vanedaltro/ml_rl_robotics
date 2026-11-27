import math
import unittest

import torch

from ppo import (ActorCritic, GridBatch, coleta, gae, objetivo_clip,
                 regiao_de_confianca, update)


class TestPPO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gae_terminal_and_truncation(self):
        r = torch.tensor([[1., 1., 1.], [10., 10., 10.]])
        v = torch.zeros_like(r)
        vp = torch.full_like(r, 4.)
        term = torch.tensor([[True, False, False], [True, True, True]])
        trunc = torch.tensor([[False, True, False], [False, False, False]])
        adv = gae(r, v, vp, term, trunc, gamma=.5, lam=1.)
        torch.testing.assert_close(adv[0], torch.tensor([1., 3., 8.]))

    def test_gae_cut_rollout_bootstraps(self):
        r = torch.tensor([[1.]])
        masks = torch.zeros_like(r, dtype=torch.bool)
        torch.testing.assert_close(gae(r, r, 2*r, masks, masks, .5, .95), r)

    def test_clip_gradient_for_both_signs(self):
        ratio = torch.tensor([1.5, .7, .7, 1.5], requires_grad=True)
        adv = torch.tensor([1., -1., 1., -1.])
        g, = torch.autograd.grad(objetivo_clip(ratio, adv).sum(), ratio)
        torch.testing.assert_close(g, torch.tensor([0., 0., 1., -1.]))

    def test_region_bound(self):
        passo, kl, p = regiao_de_confianca(.01)
        self.assertLess(passo, 4)
        self.assertLessEqual(kl, .01)
        self.assertGreater(p[1], .5)

    def test_collect_freezes_old_logp_and_updates(self):
        torch.manual_seed(7)
        modelo = ActorCritic()
        lote = coleta(modelo, GridBatch(4), 16)
        with torch.no_grad():
            pi, _ = modelo(lote.estados)
            torch.testing.assert_close(pi.log_prob(lote.acoes), lote.old_logp)
        self.assertFalse(lote.old_logp.requires_grad)
        original = lote.old_logp.clone()
        opt = torch.optim.Adam(modelo.parameters(), lr=.003)
        kl, clip, entropia = update(modelo, opt, lote)
        self.assertGreater(kl, 0)
        self.assertTrue(all(math.isfinite(m) for m in (kl, clip, entropia)))
        torch.testing.assert_close(original, lote.old_logp)

import unittest
import numpy as np
from lab import (fixture, forward, loss_and_grads, loss_and_dlogits,
                 activation_grad, activate, summarize, torch_reference, max_error)

class BackpropTests(unittest.TestCase):
    def test_loss_gradients_match_reference(self):
        for row in summarize():
            self.assertLess(row["max_error_autograd"], 1e-13)
            self.assertLess(row["loss_error_autograd"], 1e-13)
            self.assertLess(row["max_error_finite_difference"], 1e-8)
            self.assertLess(row["after_sgd"], row["before"])

    def test_shapes_and_no_double_mean(self):
        x,p,_,par = fixture()
        l,g = loss_and_grads(x,p,par)
        l2,g2 = loss_and_grads(np.tile(x,(2,1)),np.tile(p,(2,1)),par)
        self.assertAlmostEqual(l,l2)
        for k in par:
            self.assertEqual(g[k].shape,par[k].shape)
            np.testing.assert_allclose(g[k],g2[k],atol=1e-14)
        np.testing.assert_allclose(g["X"], 2*g2["X"][:len(x)],atol=1e-14)

    def test_kl_identity_and_gradient(self):
        x,_,p,par = fixture()
        z,_ = forward(x,par,"tanh")
        ce,dce = loss_and_dlogits(z,p,"ce")
        kl,dkl = loss_and_dlogits(z,p,"kl")
        h = -(p*np.log(np.where(p>0,p,1.))).sum(axis=1).mean()
        self.assertAlmostEqual(ce, h+kl)
        np.testing.assert_array_equal(dce,dkl)

    def test_logit_stability(self):
        z = np.array([[1000., -1000., -1001.]])
        p = np.array([[0., 1., 0.]])
        l,d = loss_and_dlogits(z,p,"ce")
        self.assertAlmostEqual(l,2000.)
        np.testing.assert_allclose(d,[[1.,-1.,0.]])

    def test_relu_at_zero(self):
        z=np.array([-1.,0.,1.])
        np.testing.assert_array_equal(activation_grad(z,activate(z,"relu"),"relu"),[0.,0.,1.])

    def test_bad_target(self):
        with self.assertRaises(ValueError):
            loss_and_dlogits(np.array([[1.,2.]]),np.array([[.5,.3]]),"kl")

if __name__ == "__main__":
    unittest.main()

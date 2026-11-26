import unittest
import torch
from train import MiniLM, VOCAB, GENERAL, DOMAIN, SEED, split, encode, xy, generate

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(SEED)
        self.model = MiniLM().eval()
    def test_shift(self):
        batch = encode(['ana testa sensor oficina'])
        x,y = xy(batch)
        self.assertTrue(torch.equal(x[:,1:],y[:,:-1]))
        self.assertEqual(x.shape,(1,5))
        self.assertEqual(y[0,-1].item(),1)
    def test_unique_splits(self):
        all_domains = []
        for domain in [GENERAL,DOMAIN]:
            parts = [set(p) for p in split(domain,SEED)]
            self.assertFalse(parts[0]&parts[1] or parts[0]&parts[2] or parts[1]&parts[2])
            self.assertEqual(sum(map(len,parts)),1536)
            all_domains.append(set.union(*parts))
        self.assertFalse(all_domains[0]&all_domains[1])
    def test_future_does_not_change_prefix(self):
        x = torch.tensor([[0,2,3,4,5]])
        changed = x.clone(); changed[:,3:] = torch.tensor([[8,9]])
        with torch.no_grad():
            a,b = self.model(x),self.model(changed)
        torch.testing.assert_close(a[:,:3],b[:,:3],atol=1e-6,rtol=1e-6)
    def test_gradients_and_shape(self):
        x,y = xy(encode(['ana testa sensor oficina']))
        logits = self.model(x)
        self.assertEqual(tuple(logits.shape),(1,5,len(VOCAB)))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1,len(VOCAB)),y.reshape(-1))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
    def test_sampling_seed_and_temperature(self):
        a = generate(self.model,'ana',True,.8,seed=12)
        b = generate(self.model,'ana',True,.8,seed=12)
        self.assertEqual(a,b)
        with self.assertRaises(ValueError):generate(self.model,'ana',temperature=0)

if __name__=='__main__':unittest.main()

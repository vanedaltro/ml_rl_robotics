import unittest
import torch
from lab_avaliacao import dataset, split, preprocess, score, WIDTH


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = dataset()

    def test_reproducible_and_shapes(self):
        self.assertEqual(self.data.x.shape, (2304, 18))
        self.assertTrue(torch.equal(self.data.x, dataset().x))
        self.assertFalse(torch.equal(self.data.x, dataset(7).x))

    def test_row_split_has_overlapping_raw_windows_in_same_episode(self):
        train, test = split(self.data, "linhas", 7)
        d = self.data
        overlaps = (d.episode[test, None] == d.episode[None, train]) & (
            (d.start[test, None] - d.start[None, train]).abs() < WIDTH)
        self.assertTrue(overlaps.any(1).all())

    def test_groups_disjoint_and_same_sample_count(self):
        d = self.data
        for mode in ("linhas", "episodios", "maquinas"):
            tr, te = split(d, mode, 7)
            self.assertEqual((len(tr), len(te)), (1728, 576))
            self.assertFalse(torch.isin(tr, te).any())
            if mode in ("episodios", "maquinas"):
                self.assertFalse(torch.isin(d.episode[tr], d.episode[te]).any())
            if mode == "maquinas":
                self.assertFalse(torch.isin(d.machine[tr], d.machine[te]).any())
            if mode == "episodios":
                self.assertTrue(torch.isin(d.machine[te], d.machine[tr]).all())

    def test_fit_stats_from_training_rows_only(self):
        tr, te = split(self.data, "maquinas", 7)
        x = self.data.x.clone()
        a = preprocess(x[tr])
        x[te] += 1000
        b = preprocess(x[tr])
        self.assertTrue(all(torch.equal(u,v) for u,v in zip(a,b)))
        self.assertFalse(torch.equal(a[0], preprocess(x)[0]))

    def test_metrics_for_majority_and_single_class_group(self):
        y = torch.tensor([0,0,0,1])
        base = torch.zeros_like(y)
        self.assertEqual(score(y,base)["accuracy_pct"], 75)
        self.assertEqual(score(y,base)["balanced_accuracy_pct"], 50)
        self.assertEqual(score(y,base)["false_negative"], 1)
        self.assertIsNone(score(y[:3],base[:3])["balanced_accuracy_pct"])


if __name__ == "__main__":
    unittest.main()

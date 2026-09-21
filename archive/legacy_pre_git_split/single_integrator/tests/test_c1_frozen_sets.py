"""Guard frozen-set sampling and leakage detection."""
import unittest
import numpy as np
from scripts.build_c1_v2_sets import check_disjoint, wide_starts


class FrozenSetTests(unittest.TestCase):
    def test_sampling_reproducibility_and_support(self):
        a = wide_starts(np.random.default_rng(17), 64)
        np.testing.assert_array_equal(a, wide_starts(np.random.default_rng(17), 64))
        self.assertTrue(np.all((np.abs(a[:, :, 0]) >= .55) & (np.abs(a[:, :, 0]) <= 1.05)))
        self.assertTrue(np.all(a[:, 0, 0] < 0) and np.all(a[:, 1, 0] > 0))
        self.assertTrue(np.all(np.abs(a[:, :, 1]) <= .025))
        check_disjoint([('a', a), ('b', wide_starts(np.random.default_rng(18), 64))])

    def test_rejects_internal_and_cross_set_duplicates(self):
        a = wide_starts(np.random.default_rng(17), 4)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            check_disjoint([('a', a), ('b', a[:1])])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            check_disjoint([('a', np.concatenate((a, a[:1])))])


if __name__ == '__main__':
    unittest.main()

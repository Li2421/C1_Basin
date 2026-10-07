import unittest
import numpy as np
from .motion_selection_headroom import stats, paired


class SelectionHeadroomTest(unittest.TestCase):
    def test_controller_constant_cannot_solve_reversal(self):
        s = np.array([[[16, 0], [0, 16]]])
        f = 16 - s
        constant = np.array([[[3., -3.], [3., -3.]]])
        adaptive = np.array([[[3., -3.], [-3., 3.]]])
        a = stats(constant, s/16, s, f)
        b = stats(adaptive, s/16, s, f)
        self.assertEqual((a['B15'], a['oracle_B15']), (1, 2))
        self.assertEqual(b['B15'], 2)
        self.assertIsNone(a['state_centered_contrast_correlation'])
        self.assertAlmostEqual(a['state_centered_contrast_skill'], 0.)
        self.assertEqual(b['controllers_using_both_eta'], 1)
        pair = paired(adaptive, constant, s/16, s, f, np.array([[0, 1], [1, 0]]))
        self.assertEqual((pair['rescue'], pair['breaks']), (1, 0))

    def test_incomplete_is_not_fabricated_as_failure(self):
        s = np.array([[[14, 15], [13, 16]]])
        f = np.array([[[0, 0], [2, 0]]])
        z = np.array([[[3., -3.], [3., -3.]]])
        result = stats(z, s/(s+f), s, f)
        self.assertEqual(result['unresolved_selected'], 1)
        self.assertEqual(result['non_B15'], 1)
        self.assertEqual(result['B15'], 0)


if __name__ == '__main__':
    unittest.main()

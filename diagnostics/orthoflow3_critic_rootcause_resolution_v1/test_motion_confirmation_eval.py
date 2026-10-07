import json
import unittest
import numpy as np
from .motion_confirmation_eval import measures, normalized_context


class EvaluationContracts(unittest.TestCase):
    def test_b15_and_numeric_unknown_are_separate(self):
        s = np.array([[15, 1], [14, 16], [0, 0]])
        f = np.array([[1, 15], [1, 0], [0, 2]])
        z = np.array([[3., 1.], [2., 1.], [2., 1.]])
        a = measures(z, s, f)
        self.assertEqual(a['oracle_B15'], 2)
        self.assertEqual(a['B15'], 1)
        self.assertEqual(a['unresolved'], 2)
        self.assertEqual(a['non_B15'], 0)
        self.assertEqual(a['successful_selection_if_available'], .5)
        self.assertEqual(a['severe'], 0)
        json.dumps(a, allow_nan=False)

    def test_unobserved_is_not_failure_or_exact_q16(self):
        a = measures(np.ones((4, 2)), np.zeros((4, 2)), np.zeros((4, 2)))
        self.assertIsNone(a['NLL_pair_equal'])
        self.assertIsNone(a['selected_Q_observed'])
        self.assertEqual(a['selected_Q16_lower'], 0.)
        self.assertEqual(a['selected_Q16_upper'], 1.)
        self.assertEqual(a['unresolved'], 4)
        self.assertEqual(a['non_B15'], 0)
        json.dumps(a, allow_nan=False)

    def test_correct_state_preference_reversals(self):
        s = np.array([[16, 2], [0, 15]])
        a = measures(np.array([[3., -3.], [-3., 3.]]), s, 16-s)
        self.assertEqual(a['B15'], 2)
        self.assertEqual(a['hindsight_best_constant_B15'], 1)
        self.assertEqual(a['eta0_only_B15_selected'], 1)
        self.assertEqual(a['eta1_only_B15_selected'], 1)
        self.assertEqual(a['strong_contrast_correct'], 2)

    def test_context_group_order(self):
        initial = dict(context=np.full((2, 24), 1), agent_response=np.full((2, 4, 16), 2))
        response = dict(goal_response=np.full((2, 16), 3), goal_motion_response=np.full((2, 16), 4))
        norm = {p+s: [value]*size for p, size in [('context',24),('agent',16),('goal',16),('goal_motion',16)]
                for s, value in [('_center',0),('_scale',1)]}
        c = normalized_context(initial, response, norm)
        for section, value in [(slice(0,24),1),(slice(24,88),2),(slice(88,104),3),(slice(104,120),4)]:
            self.assertTrue((c[:,section] == value).all())


if __name__ == '__main__':
    unittest.main()

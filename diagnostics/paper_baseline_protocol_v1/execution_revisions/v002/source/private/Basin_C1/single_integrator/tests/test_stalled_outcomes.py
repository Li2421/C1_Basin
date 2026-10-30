import unittest
import numpy as np

from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace


class StalledOutcomeTests(unittest.TestCase):
    def test_only_timeout_is_eligible(self):
        trace = {'max_speed': np.zeros(40), 'goal_errors': np.zeros((40, 2))}
        label, details = classify_timeout_trace(trace, 'success', .05)
        self.assertEqual(label, 'success')
        self.assertFalse(details['eligible'])

    def test_stalled_timeout(self):
        trace = {'max_speed': np.full(40, .049),
                 'goal_errors': np.vstack([np.array([1., 2.]) + i*np.array([.0001, -.0001]) for i in range(40)])}
        label, details = classify_timeout_trace(trace, 'other_timeout', .05)
        self.assertEqual(label, 'stalled_deadlock')
        self.assertTrue(details['stalled'])

    def test_motion_or_progress_remains_timeout(self):
        trace = {'max_speed': np.r_[np.full(39, .01), .05],
                 'goal_errors': np.zeros((40, 2))}
        self.assertEqual(classify_timeout_trace(trace, 'other_timeout', .05)[0], 'other_timeout')
        trace['max_speed'][:] = .01
        trace['goal_errors'][-1, 0] = .02
        self.assertEqual(classify_timeout_trace(trace, 'other_timeout', .05)[0], 'other_timeout')


if __name__ == '__main__':
    unittest.main()

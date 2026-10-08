import unittest
from types import SimpleNamespace

import numpy as np

from new_benchmark_common.dataset import fit_train_normalization


class ActiveActionNormalizationTests(unittest.TestCase):
    def test_physical_action_units_do_not_shrink_with_waiting_agents(self):
        scales = []
        for agents in (2, 50):
            actions = np.zeros((4, agents, 2), dtype=np.float32)
            actions[:, 0, 0] = [.2, .3, -.2, -.3]
            dataset = SimpleNamespace(split='train', actions=actions,
                observations=np.zeros((4, agents, 8), dtype=np.float32))
            normalization = fit_train_normalization(dataset,
                shared_agents=True, active_action_scale=True)
            scales.append(normalization['act_scale'][0])
        self.assertAlmostEqual(scales[0], scales[1], places=10)


if __name__ == '__main__':
    unittest.main()

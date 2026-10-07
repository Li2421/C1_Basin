import unittest

import numpy as np

from double_bottleneck.environment import Config
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.recovery_v2 import (
    goal_anchors,
    observation_from_state,
    query_reference_recovery,
    transition_anchors,
)


DATASET = "diagnostics/double_bottleneck_expert_dataset_8mode"


class RecoveryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = FlowBC4ADataset(DATASET, "train", seed=0)
        cls.episode = cls.dataset.episodes[0]
        cls.config = Config(**cls.dataset.config)

    def test_observation_reconstruction_matches_nominal_episode(self):
        step = 17
        reconstructed = observation_from_state(
            self.episode.positions[step],
            self.episode.observations[step, :, 2:4],
            self.dataset.goals,
        )
        np.testing.assert_allclose(
            reconstructed, self.episode.observations[step], rtol=0, atol=8e-7
        )

    def test_target_anchor_sets_are_bounded_and_nonempty(self):
        transition = transition_anchors(self.episode, self.config)
        goal = goal_anchors(self.episode, self.config)
        self.assertGreater(len(transition), 0)
        self.assertLessEqual(len(transition), 36)
        self.assertGreater(len(goal), 0)
        self.assertLessEqual(len(goal), 18)
        self.assertTrue(all(0 <= row["step"] < self.episode.length for row in transition + goal))

    def test_exact_source_state_recovery_is_safe_and_mode_preserving(self):
        step = transition_anchors(self.episode, self.config)[0]["step"]
        result = query_reference_recovery(
            self.episode,
            step,
            self.episode.positions[step],
            self.episode.observations[step, :, 2:4],
        )
        self.assertTrue(result.success)
        self.assertEqual(result.terminal_reason, "success")
        self.assertFalse(result.wall_collision)
        self.assertFalse(result.agent_collision)
        self.assertTrue(result.mode_signature_match)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np

from new_benchmark_common.dataset import (
    DatasetWriter,
    JointTransitionDataset,
    RecoveryAudit,
    Trajectory,
)


class _Scenario:
    name = "dummy"
    agent_order = ("A", "B", "C", "D")
    observation_shape = (4, 3)
    action_shape = (4, 2)
    environment_fingerprint = "a" * 64


def _trajectory(split="train", source="nominal", audit=None):
    return Trajectory(
        rollout_id=f"{split}_{source}", split=split, source=source,
        initial_state={"position": np.zeros((4, 2))}, states=np.zeros((3, 4, 2)),
        observations=np.zeros((3, 4, 3)), actions=np.zeros((2, 4, 2)),
        metadata={"success": True}, recovery_audit=audit,
    )


class DatasetScaffoldTests(unittest.TestCase):
    def test_round_trip_and_source_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DatasetWriter(directory, _Scenario(), scenario_config={"dt": 0.1})
            writer.add(_trajectory())
            writer.add(_trajectory("dev"))
            writer.add(_trajectory("test"))
            writer.finalize()
            loaded = JointTransitionDataset(directory, "train", seed=4)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.sample(5)["observations"].shape, (5, 4, 3))
            self.assertEqual(loaded.source_counts()["nominal"], 2)

    def test_test_rollout_cannot_source_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = DatasetWriter(directory, _Scenario(), scenario_config={})
            audit = RecoveryAudit("test_failure", "test", 0, "wall", "north", 0.1, 9, True)
            with self.assertRaises(ValueError):
                writer.add(_trajectory("train", "targeted_wall_obstacle", audit))


if __name__ == "__main__":
    unittest.main()

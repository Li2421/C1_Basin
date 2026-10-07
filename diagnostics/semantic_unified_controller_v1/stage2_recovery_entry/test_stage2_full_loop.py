"""CPU-only unit tests for Stage-2 full-episode semantics and manifest freeze."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import build_stage2_full_loop_manifest as builder
from mode_common import content_hash, sha256
from stage2_full_loop_common import Stage2ModeMachine


class FakeEnv:
    def __init__(self, horizon: int):
        self.step_count = 0
        self.horizon = horizon
        self.done = False


class FakeKernel:
    def __init__(self, horizon: int = 3):
        self.env = FakeEnv(horizon)
        self.config = SimpleNamespace(max_speed=1.0)
        self.prepare_count = self.execute_count = self.second_projection_count = 0

    def prepare(self, *, need_feature: bool):
        self.prepare_count += 1
        step = self.env.step_count
        return SimpleNamespace(
            step=step, observation=np.full(4, step, dtype=float),
            u_safe=np.zeros((2, 2)), feature=np.full(214, step, dtype=float) if need_feature else None,
            first_projection_retry=False,
        )

    def second_projection(self, context, correction):
        self.second_projection_count += 1
        return context.u_safe + np.asarray(correction), False

    def execute(self, action):
        self.execute_count += 1
        self.env.step_count += 1
        self.env.done = self.env.step_count >= self.env.horizon
        return self.env.done, {}


class SequenceModeHead:
    def __init__(self, actions):
        self.actions = iter(actions)

    def __call__(self, feature):
        return next(self.actions)


class DirectModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, features):
        self.calls += 1
        return np.full((1, 4), .1)


class EtaModel:
    def __init__(self):
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        eta = np.asarray([[.2, -.1, .3]])
        return eta, eta, eta


class FullLoopMachineTests(unittest.TestCase):
    def test_local_then_recovery_latches_eta_once_and_never_returns(self):
        kernel = FakeKernel(4); direct = DirectModel(); eta = EtaModel()
        factory_calls = []

        def factory(latched):
            factory_calls.append(np.asarray(latched).copy())
            return lambda observation, u_safe, max_speed: np.full((2, 2), .2)

        machine = Stage2ModeMachine(
            kernel=kernel, mode_head=SequenceModeHead([1, 2]), direct_model=direct,
            eta_model=eta, eta_corrector_factory=factory,
        )
        rows = machine.run_to_terminal()
        self.assertEqual([row.decision for row in rows],
                         ["LOCAL", "ENTER_RECOVERY", "CONTINUE_RECOVERY", "CONTINUE_RECOVERY"])
        self.assertEqual(machine.mode_query_count, 2)
        self.assertEqual(direct.calls, 1)
        self.assertEqual(eta.calls, 1)
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(machine.recovery_transition_count, 3)
        self.assertEqual(kernel.prepare_count, kernel.execute_count)
        self.assertEqual(kernel.second_projection_count, 4)
        np.testing.assert_array_equal(machine.eta_latched, [.2, -.1, .3])
        self.assertFalse(machine.eta_latched.flags.writeable)

    def test_safety_action_uses_no_second_projection(self):
        kernel = FakeKernel(2); direct = DirectModel(); eta = EtaModel()
        machine = Stage2ModeMachine(
            kernel=kernel, mode_head=SequenceModeHead([0, 0]), direct_model=direct,
            eta_model=eta, eta_corrector_factory=lambda unused: None,
        )
        rows = machine.run_to_terminal()
        self.assertTrue(all(row.decision == "SAFETY" for row in rows))
        self.assertEqual(kernel.second_projection_count, 0)
        self.assertEqual(direct.calls, 0)
        self.assertEqual(eta.calls, 0)


class ManifestTests(unittest.TestCase):
    @staticmethod
    def checkpoint(path: Path, output: int) -> None:
        np.savez(
            path, normalization_mean=np.zeros(214), normalization_scale=np.ones(214),
            layer_0_weight=np.zeros((214, 64)), layer_0_bias=np.zeros(64),
            layer_1_weight=np.zeros((64, 64)), layer_1_bias=np.zeros(64),
            layer_2_weight=np.zeros((64, output)), layer_2_bias=np.zeros(output),
        )

    def test_only_validation_and_calibration_can_be_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "direct.npz"; np.savez(direct, x=np.asarray(1))
            eta = root / "eta.npz"; np.savez(eta, x=np.asarray(1))
            local = root / "local.npz"; np.savez(local, x=np.asarray(1))
            mode = root / "mode.npz"; self.checkpoint(mode, 3)
            source = {
                "status": "FROZEN_BEFORE_NEW_OUTCOME_EVALUATION", "final_test_generated": False,
                "sources": {
                    "validation": [{"source_id": "v0", "episode_index": 0, "rollout_id": 0,
                                    "flow_root_seed": 7, "initial_positions": [[-1., 0.], [1., 0.]]}],
                    "calibration": [], "train": [],
                },
            }
            source["content_sha256"] = content_hash(source)
            source_path = root / "sources.json"
            source_path.write_text(__import__("json").dumps(source))
            with mock.patch.multiple(
                builder, DIRECT_CHECKPOINT=direct, DIRECT_SHA256=sha256(direct),
                ETA_CHECKPOINT=eta, ETA_SHA256=sha256(eta), LOCAL_HEAD=local,
                SOURCE_MANIFEST=source_path,
            ):
                manifest = builder.build(
                    controller="SEMANTIC_MODE", split="validation",
                    output=root / "unused.json", mode_checkpoint=mode,
                )
                self.assertEqual(manifest["task_count"], 1)
                self.assertFalse(manifest["contains_periodic_timing"])
                self.assertTrue(manifest["final_test_forbidden"])
                with self.assertRaises(ValueError):
                    builder.build(controller="SAFETY", split="train", output=root / "other.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)

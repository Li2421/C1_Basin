"""Regression and source-parity tests for joint 4-agent MACFlow Stage-I."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import inspect
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from double_bottleneck.flowbc_4a_agent import (
    ActorVectorField,
    DoubleBottleneckFlowBCAgent,
    JOINT_ACT_DIM,
    JOINT_OBS_DIM,
    MACFLOW_OFFICIAL,
    ModuleDict,
    TrainState,
    get_config,
    load_checkpoint,
    sample_bounded_actions,
    save_checkpoint,
)
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset, fit_train_normalization
from flowbc.giveway_flowbc_agent import GiveWayFlowBCAgent, get_config as toy_get_config


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET = REPO_ROOT / "diagnostics" / "double_bottleneck_expert_dataset"


def small_config(fingerprint: str):
    config = get_config()
    config.update(
        actor_hidden_dims=(16, 16),
        flow_steps=2,
        environment_fingerprint=fingerprint,
        max_speed=0.5,
        normalize=True,
        obs_mean=(0.0,) * JOINT_OBS_DIM,
        obs_scale=(1.0,) * JOINT_OBS_DIM,
        act_mean=(0.0,) * JOINT_ACT_DIM,
        act_scale=(1.0,) * JOINT_ACT_DIM,
    )
    return config


class RealDatasetTests(unittest.TestCase):
    def test_strict_loader_opens_and_validates_real_pilot_episodes(self):
        train = FlowBC4ADataset(REAL_DATASET, "train", seed=1)
        val = FlowBC4ADataset(REAL_DATASET, "val", seed=2)
        self.assertEqual(len(train.episodes), 18)
        self.assertEqual(len(val.episodes), 6)
        self.assertEqual(len(train), 12907)
        self.assertEqual(len(val), 4303)
        self.assertEqual(train.environment_fingerprint, val.environment_fingerprint)
        self.assertEqual(train.episodes[0].observations.shape[1:], (4, 18))
        self.assertEqual(train.episodes[0].actions.shape[1:], (4, 2))
        expected_indices = np.random.default_rng(1).integers(0, len(train), size=7)
        batch = train.sample(7)
        np.testing.assert_array_equal(
            batch["observations"], train.observations[expected_indices]
        )
        np.testing.assert_array_equal(batch["actions"], train.actions[expected_indices])

    def test_normalization_is_train_only_and_has_joint_dimensions(self):
        train = FlowBC4ADataset(REAL_DATASET, "train")
        stats = fit_train_normalization(train)
        self.assertEqual(len(stats["obs_mean"]), 72)
        self.assertEqual(len(stats["act_mean"]), 8)
        self.assertGreater(min(stats["obs_scale"]), 0)
        with self.assertRaises(ValueError):
            fit_train_normalization(FlowBC4ADataset(REAL_DATASET, "val"))


class MACFlowSourceParityTests(unittest.TestCase):
    def test_toy_and_four_agent_use_identical_macflow_primitive_classes(self):
        db = DoubleBottleneckFlowBCAgent.create(
            0,
            jnp.zeros((1, 4, 18), dtype=jnp.float32),
            jnp.zeros((1, 4, 2), dtype=jnp.float32),
            small_config("f" * 64),
        )
        toy = GiveWayFlowBCAgent.create(
            0,
            jnp.zeros((1, 2, 10), dtype=jnp.float32),
            jnp.zeros((1, 2, 2), dtype=jnp.float32),
            toy_get_config(),
        )
        self.assertIs(type(db.network), type(toy.network))
        self.assertIs(type(db.network.model_def), type(toy.network.model_def))
        self.assertIsInstance(db.network, TrainState)
        self.assertIsInstance(db.network.model_def, ModuleDict)
        self.assertIsInstance(
            db.network.model_def.modules["actor_bc_flow"], ActorVectorField
        )
        self.assertIs(
            type(db.network.model_def.modules["actor_bc_flow"]),
            type(toy.network.model_def.modules["actor_bc_flow"]),
        )
        self.assertTrue(Path(inspect.getfile(ActorVectorField)).is_relative_to(MACFLOW_OFFICIAL))
        self.assertEqual(db.config["use_critic"], toy.config["use_critic"])
        self.assertEqual(db.config["use_q_guidance"], toy.config["use_q_guidance"])
        self.assertEqual(db.config["use_distillation"], toy.config["use_distillation"])

    def test_joint_loss_update_sampling_and_speed_bound_are_finite(self):
        agent = DoubleBottleneckFlowBCAgent.create(
            0,
            jnp.zeros((5, 4, 18), dtype=jnp.float32),
            jnp.zeros((5, 4, 2), dtype=jnp.float32),
            small_config("f" * 64),
        )
        batch = {
            "observations": np.zeros((5, 4, 18), dtype=np.float32),
            "actions": np.zeros((5, 4, 2), dtype=np.float32),
        }
        loss, _ = agent.total_loss(batch, agent.network.params, rng=jax.random.PRNGKey(1))
        self.assertTrue(np.isfinite(float(loss)))
        agent, info = agent.update(batch, 1)
        self.assertTrue(all(np.isfinite(float(value)) for value in info.values()))
        sample = np.asarray(agent.sample_actions(batch["observations"], jax.random.PRNGKey(3)))
        bounded = np.asarray(
            sample_bounded_actions(agent, batch["observations"], jax.random.PRNGKey(3))
        )
        self.assertEqual(sample.shape, (5, 4, 2))
        self.assertTrue(np.isfinite(sample).all())
        self.assertLessEqual(np.linalg.norm(bounded, axis=-1).max(), 0.5 + 1e-6)

    def test_checkpoint_reload_is_exact_and_fingerprint_guarded(self):
        agent = DoubleBottleneckFlowBCAgent.create(
            5,
            jnp.zeros((1, 4, 18), dtype=jnp.float32),
            jnp.zeros((1, 4, 2), dtype=jnp.float32),
            small_config("a" * 64),
        )
        observations = jnp.zeros((2, 4, 18), dtype=jnp.float32)
        key = jax.random.PRNGKey(8)
        expected = np.asarray(agent.sample_actions(observations, key))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pilot.pkl"
            save_checkpoint(checkpoint, agent, {"step": 0})
            restored, metadata = load_checkpoint(checkpoint, "a" * 64)
            actual = np.asarray(restored.sample_actions(observations, key))
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(metadata["step"], 0)
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint, "b" * 64)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest

import jax
import jax.numpy as jnp

from new_benchmark_common.macflow import (
    ActorVectorField,
    JointMACFlowAgent,
    ModuleDict,
    TrainState,
    get_config,
    load_checkpoint,
    save_checkpoint,
)


class OfficialMACFlowTests(unittest.TestCase):
    def test_joint_shapes_official_primitives_and_round_trip(self):
        observations = jnp.zeros((2, 4, 5), dtype=jnp.float32)
        actions = jnp.zeros((2, 4, 2), dtype=jnp.float32)
        config = get_config(
            agent_order=("A", "B", "C", "D"), num_agents=4, obs_dim=5, act_dim=2,
            obs_mean=(0.0,) * 20, obs_scale=(1.0,) * 20,
            act_mean=(0.0,) * 8, act_scale=(1.0,) * 8,
            environment_fingerprint="f" * 64,
        )
        agent = JointMACFlowAgent.create(7, observations, actions, config)
        self.assertIsInstance(agent.network, TrainState)
        self.assertIsInstance(agent.network.model_def, ModuleDict)
        self.assertIsInstance(agent.network.model_def.modules["actor_bc_flow"], ActorVectorField)
        sample = agent.sample_actions(observations, jax.random.PRNGKey(8))
        self.assertEqual(sample.shape, (2, 4, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/checkpoint.pkl"
            save_checkpoint(path, agent, {"smoke": True})
            restored, metadata = load_checkpoint(path, expected_environment_fingerprint="f" * 64)
            self.assertEqual(metadata, {"smoke": True})
            self.assertTrue(jnp.array_equal(sample, restored.sample_actions(observations, jax.random.PRNGKey(8))))


if __name__ == "__main__":
    unittest.main()

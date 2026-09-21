import hashlib
import unittest

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.direction_a.policy import (
    DirectionADistribution, DirectionAResidual,
)
from single_integrator.c1.direction_a.randomness import IndexedRandomTape
from single_integrator.c1.direction_a.rollout import FrozenDirectionAController
from single_integrator.c1.direction_a.rollout import rollout_episode
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import load_policy


jax.config.update("jax_enable_x64", True)


class FrozenControllerIntegrationTests(unittest.TestCase):
    """One-step contract fixture; this is not an empirical pilot."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl"
        cls.baseline, provenance = load_policy(cls.checkpoint)
        cls.plant = Config(**provenance["evaluation_environment"])

    def test_flow_is_frozen_state_is_current_and_prefix_rng_replays(self):
        # Test-only values exercise the adapter; they are not an approved phi_0.
        model = DirectionAResidual(alpha_bias=-100., beta_bias=0.,
                                   hidden_dims=(8,))
        params = model.init(jax.random.PRNGKey(1), jnp.zeros((1, 4)),
                            jnp.zeros((1, 1)), jnp.zeros((1, 20)))
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64),
                                        params)
        controller = FrozenDirectionAController(
            self.baseline, model, params, DirectionADistribution(.01, .02),
            self.plant)
        tape = IndexedRandomTape(123)
        env = GiveWayEnv(self.plant)
        env.reset(np.array([[-.7, 0.], [.7, 0.]]))
        frozen_before = hashlib.sha256(
            flax.serialization.to_bytes(self.baseline)).hexdigest()
        first = controller.act(env, tape, "contract", 0, 0, 0)
        self.assertFalse(first["gate"])
        np.testing.assert_array_equal(first["residual"], np.zeros(4))
        np.testing.assert_allclose(first["applied"], first["safe"], rtol=0, atol=0)
        env.step(first["applied"].reshape(2, 2))
        second = controller.act(env, tape, "contract", 0, 0, 1)
        self.assertEqual(env.step_count, 1)
        self.assertGreater(np.max(np.abs(second["observation"]-first["observation"])), 0.)
        frozen_after = hashlib.sha256(
            flax.serialization.to_bytes(self.baseline)).hexdigest()
        self.assertEqual(frozen_before, frozen_after)
        # Replaying the same live prefix and index produces the same raw draws.
        repeat = controller.act(env, tape, "contract", 0, 0, 1)
        np.testing.assert_array_equal(second["raw_flow"], repeat["raw_flow"])
        np.testing.assert_array_equal(second["epsilon"], repeat["epsilon"])

    def test_complete_adapter_stops_without_post_terminal_padding(self):
        class ZeroFlowNetwork:
            @staticmethod
            def select(name):
                if name != "actor_bc_flow":
                    raise ValueError(name)
                return lambda observation, action, time: -action

        class ZeroFlowBaseline:
            config = {"flow_steps": 1, "normalize": False}
            network = ZeroFlowNetwork()

            @staticmethod
            def _flatten(observation, action):
                return observation.reshape(len(observation), 20), action.reshape(len(action), 4)

        model = DirectionAResidual(alpha_bias=-100., beta_bias=0.,
                                   hidden_dims=(8,))
        params = model.init(jax.random.PRNGKey(2), jnp.zeros((1, 4)),
                            jnp.zeros((1, 1)), jnp.zeros((1, 20)))
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64),
                                        params)
        controller = FrozenDirectionAController(
            ZeroFlowBaseline(), model, params,
            DirectionADistribution(.01, .02), self.plant)
        # Both agents begin inside the goal tolerance.  The monitor evaluates
        # the terminating action and the adapter emits exactly one record.
        initial = GiveWayEnv(self.plant).goals.copy()
        result = rollout_episode(controller, initial, IndexedRandomTape(9),
                                 "contract", 0, 0, np.eye(4))
        self.assertEqual(result["episode_steps"], 1)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["terminal_label"], "success")
        self.assertTrue(result["task_success"])
        self.assertFalse(result["D_H"])
        self.assertEqual(float(result["deformation"]["value"]), 0.)


if __name__ == "__main__":
    unittest.main()

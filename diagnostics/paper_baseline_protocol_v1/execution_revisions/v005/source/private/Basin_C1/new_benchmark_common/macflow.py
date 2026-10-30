"""Official-MACFlow joint Stage-I conditional flow-matching policy.

This is intentionally a dimensional generalization of the validated
Double-Bottleneck implementation: official ``ActorVectorField``, ``ModuleDict``
and ``TrainState``; straight-line CFM; Adam; and ten Euler sampling steps.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax
import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


REPO_ROOT = Path(__file__).resolve().parents[1]
MACFLOW_OFFICIAL = REPO_ROOT.parent / "01_MACFlow_Baseline_Reproduction" / "MACFlow_Official"
if not MACFLOW_OFFICIAL.is_dir():
    raise FileNotFoundError(f"MACFlow_Official source tree not found: {MACFLOW_OFFICIAL}")
if str(MACFLOW_OFFICIAL) not in sys.path:
    sys.path.insert(0, str(MACFLOW_OFFICIAL))

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field  # noqa: E402
from utils.networks import ActorVectorField  # noqa: E402


CHECKPOINT_SCHEMA = "new_benchmark_joint_macflow_stage_i_v1"


class SetActorVectorField(nn.Module):
    """Permutation-equivariant joint Flow field for large agent sets.

    It consumes the same physical per-agent observation and joint velocity
    variables as the official flattened actor. Self-attention exposes nearby
    and distant traffic without an agent-index priority or a release rule.
    """

    num_agents: int
    obs_dim: int
    act_dim: int
    width: int = 128
    layers: int = 2

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        del is_encoded
        batch = observations.shape[0]
        observed = observations.reshape(batch, self.num_agents, self.obs_dim)
        action = actions.reshape(batch, self.num_agents, self.act_dim)
        if times is None:
            clock = jnp.zeros((batch, self.num_agents, 1), dtype=observed.dtype)
        else:
            clock = jnp.broadcast_to(times[:, None, :], (batch, self.num_agents, 1))
        hidden = nn.Dense(self.width)(jnp.concatenate((observed, action, clock), axis=-1))
        hidden = nn.gelu(hidden)
        for _ in range(self.layers):
            attended = nn.SelfAttention(num_heads=4, qkv_features=self.width,
                                        out_features=self.width)(hidden)
            hidden = nn.LayerNorm()(hidden + attended)
            residual = nn.Dense(self.width)(nn.gelu(nn.Dense(self.width)(hidden)))
            hidden = nn.LayerNorm()(hidden + residual)
        return nn.Dense(self.act_dim)(hidden).reshape(batch, self.num_agents * self.act_dim)


class JointMACFlowAgent(flax.struct.PyTreeNode):
    """A true joint ``p(u_1,...,u_N | x)`` policy with no mode input/state."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _flatten(self, observations: jnp.ndarray, actions: jnp.ndarray):
        batch = observations.shape[0]
        obs_flat = observations.reshape(batch, self.config["joint_obs_dim"])
        act_flat = actions.reshape(batch, self.config["joint_act_dim"])
        if self.config["normalize"]:
            obs_flat = (obs_flat - jnp.asarray(self.config["obs_mean"])) / jnp.asarray(
                self.config["obs_scale"]
            )
            act_flat = (act_flat - jnp.asarray(self.config["act_mean"])) / jnp.asarray(
                self.config["act_scale"]
            )
        return obs_flat, act_flat

    def flow_bc_samples(self, batch: Mapping[str, jnp.ndarray], rng):
        _, x_rng, t_rng = jax.random.split(rng, 3)
        observations, actions = self._flatten(
            jnp.asarray(batch["observations"]), jnp.asarray(batch["actions"])
        )
        x_0 = jax.random.normal(x_rng, actions.shape)
        time = jax.random.uniform(t_rng, (observations.shape[0], 1))
        x_t = (1.0 - time) * x_0 + time * actions
        return observations, x_t, time, actions - x_0

    def flow_bc_loss(self, batch, grad_params, rng):
        observations, x_t, time, velocity = self.flow_bc_samples(batch, rng)
        prediction = self.network.select("actor_bc_flow")(
            observations, x_t, time, params=grad_params
        )
        squared_error = (prediction - velocity) ** 2
        motion_weight = self.config.get("motion_loss_weight", 1.0)
        if motion_weight != 1.0:
            # Large-N one-way data contain many agents waiting safely while a
            # few cross the gate. Preserve the complete joint target, but do
            # not let the numerous zero-action rows dominate its CFM loss.
            moving = jnp.linalg.norm(jnp.asarray(batch["actions"]), axis=-1) > 0.1
            weights = 1.0 + (motion_weight - 1.0) * moving.astype(squared_error.dtype)
            per_agent = jnp.mean(squared_error.reshape(
                squared_error.shape[0], self.config["num_agents"], self.config["act_dim"]),
                axis=-1)
            loss = jnp.sum(per_agent * weights) / jnp.sum(weights)
        else:
            loss = jnp.mean(squared_error)
        return loss, {"bc_flow_loss": loss}

    def endpoint_bc_loss(self, batch, grad_params, rng):
        """Optional supervision on the action actually produced by 10 Flow steps.

        The conventional CFM velocity loss can be small while the sampled
        endpoint falsely moves held agents or undershoots final goals. This
        loss keeps the original Flow integration and penalizes that measured
        endpoint error. It is disabled for all existing checkpoints.
        """
        observations, targets = self._flatten(
            jnp.asarray(batch["observations"]), jnp.asarray(batch["actions"])
        )
        samples = jax.random.normal(rng, targets.shape)
        for index in range(self.config["flow_steps"]):
            time = jnp.full((observations.shape[0], 1), index / self.config["flow_steps"])
            samples = samples + self.network.select("actor_bc_flow")(
                observations, samples, time, params=grad_params
            ) / self.config["flow_steps"]
        squared_error = (samples - targets) ** 2
        endpoint_motion_weight = float(self.config.get("endpoint_motion_weight", 1.0))
        if endpoint_motion_weight != 1.0:
            moving = jnp.linalg.norm(jnp.asarray(batch["actions"]), axis=-1) > 0.1
            weights = 1.0 + (endpoint_motion_weight - 1.0) * moving.astype(squared_error.dtype)
            per_agent = jnp.mean(squared_error.reshape(
                squared_error.shape[0], self.config["num_agents"], self.config["act_dim"]),
                axis=-1)
            return jnp.sum(per_agent * weights) / jnp.sum(weights)
        return jnp.mean(squared_error)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        endpoint_weight = float(self.config.get("endpoint_action_loss_weight", 0.0))
        if endpoint_weight:
            _, flow_rng, endpoint_rng = jax.random.split(rng, 3)
            flow_loss, info = self.flow_bc_loss(batch, grad_params, flow_rng)
            endpoint_loss = self.endpoint_bc_loss(batch, grad_params, endpoint_rng)
            total = flow_loss + endpoint_weight * endpoint_loss
            return total, {**info, "endpoint_action_loss": endpoint_loss,
                           "total_loss": total}
        _, flow_rng = jax.random.split(rng)
        return self.flow_bc_loss(batch, grad_params, flow_rng)

    @jax.jit
    def update(self, batch, step: int):
        del step
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, rng=loss_rng)

        network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray, seed):
        batch = observations.shape[0]
        flat_observations = observations.reshape(batch, self.config["joint_obs_dim"])
        reflect_average = bool(self.config.get("reflect_average", False))
        if reflect_average:
            if self.config["obs_dim"] != 8 or self.config["act_dim"] != 2:
                raise ValueError("reflection averaging requires the eight-feature planar Gap observation")
            # The paired Gap1 data establish exact x-reflection for these
            # physical features and targets. This optional Flow-only adapter
            # symmetrizes the conditional action distribution; it supplies no
            # priority, map route, safety constraint or yielding rule.
            observation_sign = jnp.asarray([-1., 1., -1., 1., -1., 1., -1., 1.])
            action_sign = jnp.asarray([-1., 1.])
            reflected_observations = (observations.reshape(batch, self.config["num_agents"], 8)
                                      * observation_sign).reshape(batch, self.config["joint_obs_dim"])
        if self.config["normalize"]:
            flat_observations = (
                flat_observations - jnp.asarray(self.config["obs_mean"])
            ) / jnp.asarray(self.config["obs_scale"])
            if reflect_average:
                reflected_observations = (
                    reflected_observations - jnp.asarray(self.config["obs_mean"])
                ) / jnp.asarray(self.config["obs_scale"])
        actions = jax.random.normal(seed, (batch, self.config["joint_act_dim"]))
        if reflect_average:
            reflected_actions = (actions.reshape(batch, self.config["num_agents"], 2)
                                 * action_sign).reshape(batch, self.config["joint_act_dim"])
        # Fixed by protocol: conventional Stage-I, ten Euler integrations.
        for index in range(self.config["flow_steps"]):
            time = jnp.full((batch, 1), index / self.config["flow_steps"])
            actions = actions + self.network.select("actor_bc_flow")(
                flat_observations, actions, time
            ) / self.config["flow_steps"]
            if reflect_average:
                reflected_actions = reflected_actions + self.network.select("actor_bc_flow")(
                    reflected_observations, reflected_actions, time
                ) / self.config["flow_steps"]
        if self.config["normalize"]:
            actions = actions * jnp.asarray(self.config["act_scale"]) + jnp.asarray(
                self.config["act_mean"]
            )
            if reflect_average:
                reflected_actions = (reflected_actions * jnp.asarray(self.config["act_scale"])
                                     + jnp.asarray(self.config["act_mean"]))
        actions = jnp.clip(actions, -1.0, 1.0)
        if reflect_average:
            reflected_actions = jnp.clip(reflected_actions, -1.0, 1.0)
            reflected_actions = (reflected_actions.reshape(batch,self.config["num_agents"],2)
                                 * action_sign).reshape(batch,self.config["joint_act_dim"])
            actions = 0.5 * (actions + reflected_actions)
        return actions.reshape(batch, self.config["num_agents"], self.config["act_dim"])

    @classmethod
    def create(cls, seed: int, observations, actions, config: ml_collections.ConfigDict):
        observations = jnp.asarray(observations)
        actions = jnp.asarray(actions)
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("expected [batch, agents, feature] observations/actions")
        if observations.shape[:2] != actions.shape[:2]:
            raise ValueError("observation/action batch-agent dimensions differ")
        if config["flow_steps"] != 10:
            raise ValueError("new benchmark Stage-I protocol requires exactly 10 Euler steps")
        num_agents, obs_dim, act_dim = observations.shape[1], observations.shape[2], actions.shape[2]
        if config.get("num_agents", num_agents) != num_agents:
            raise ValueError("config/observation agent count mismatch")
        if config.get("obs_dim", obs_dim) != obs_dim or config.get("act_dim", act_dim) != act_dim:
            raise ValueError("config/tensor feature dimensions mismatch")
        config = config.copy_and_resolve_references()
        config.update(
            num_agents=int(num_agents), obs_dim=int(obs_dim), act_dim=int(act_dim),
            joint_obs_dim=int(num_agents * obs_dim), joint_act_dim=int(num_agents * act_dim),
        )
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        flat_observations = observations.reshape(len(observations), config["joint_obs_dim"])
        flat_actions = actions.reshape(len(actions), config["joint_act_dim"])
        architecture = config.get("architecture", "flat_mlp")
        if architecture == "flat_mlp":
            actor = ActorVectorField(
                hidden_dims=config["actor_hidden_dims"], action_dim=config["joint_act_dim"],
                layer_norm=config["actor_layer_norm"], encoder=None,
            )
        elif architecture == "set_attention":
            actor = SetActorVectorField(
                num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim,
                width=int(config["set_width"]), layers=int(config["set_layers"]),
            )
        else:
            raise ValueError(f"unsupported Flow architecture: {architecture}")
        definition = ModuleDict({"actor_bc_flow": actor})
        params = definition.init(
            init_rng, actor_bc_flow=(flat_observations, flat_actions, jnp.zeros((len(actions), 1)))
        )["params"]
        network = TrainState.create(definition, params, tx=optax.adam(config["lr"]))
        # Build this in two steps: configuration may already carry these
        # protocol fields, and FrozenDict rejects duplicate keyword names.
        frozen_payload = dict(config)
        frozen_payload.update(
            agent_order=tuple(config["agent_order"]),
            use_critic=False, use_q_guidance=False, use_distillation=False,
            critic_loss_weight=0.0, q_guidance_weight=0.0, distillation_weight=0.0,
        )
        frozen = flax.core.FrozenDict(frozen_payload)
        return cls(rng=rng, network=network, config=frozen)


def get_config(**overrides) -> ml_collections.ConfigDict:
    """The mature Toy/Double-Bottleneck Stage-I defaults, parameterized by shape."""
    config = ml_collections.ConfigDict(dict(
        lr=3e-4, actor_hidden_dims=(256, 256, 256), actor_layer_norm=False,
        flow_steps=10, normalize=True, num_agents=4, obs_dim=0, act_dim=2,
        joint_obs_dim=0, joint_act_dim=0, agent_order=(), environment_fingerprint="",
        max_speed=1.0, motion_loss_weight=1.0, reflect_average=False,
        endpoint_action_loss_weight=0.0, endpoint_motion_weight=1.0,
        architecture="flat_mlp", set_width=128, set_layers=2,
    ))
    config.update(overrides)
    return config


def speed_bound(actions, max_speed: float):
    actions = jnp.asarray(actions)
    norms = jnp.linalg.norm(actions, axis=-1, keepdims=True)
    return actions * jnp.minimum(1.0, max_speed / jnp.maximum(norms, 1e-12))


def sample_bounded_actions(agent: JointMACFlowAgent, observations, seed):
    return speed_bound(agent.sample_actions(observations, seed), agent.config["max_speed"])


def parameter_count(agent: JointMACFlowAgent) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(agent.network.params)))


def save_checkpoint(path: str | Path, agent: JointMACFlowAgent, metadata: Mapping[str, Any]) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "agent": flax.serialization.to_state_dict(agent),
        "config": dict(agent.config),
        "metadata": dict(metadata),
    }
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str | Path, *, expected_environment_fingerprint: str):
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("not a new-benchmark joint MACFlow checkpoint")
    config = get_config()
    config.update(payload["config"])
    if config["environment_fingerprint"] != expected_environment_fingerprint:
        raise ValueError("checkpoint/environment fingerprint mismatch")
    agent = JointMACFlowAgent.create(
        0,
        jnp.zeros((1, config["num_agents"], config["obs_dim"]), dtype=jnp.float32),
        jnp.zeros((1, config["num_agents"], config["act_dim"]), dtype=jnp.float32),
        config,
    )
    agent = flax.serialization.from_state_dict(agent, payload["agent"])
    return agent, dict(payload.get("metadata", {}))


__all__ = ("ActorVectorField", "CHECKPOINT_SCHEMA", "JointMACFlowAgent", "MACFLOW_OFFICIAL",
           "ModuleDict", "TrainState", "get_config", "load_checkpoint", "parameter_count",
           "sample_bounded_actions", "save_checkpoint", "speed_bound")

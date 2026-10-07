"""MACFlow-official Stage-I joint Flow-BC for Double-Bottleneck.

This is the four-agent dimensional extension of
``flowbc.giveway_flowbc_agent.GiveWayFlowBCAgent``. It reuses the exact
MACFlow_Official ``ActorVectorField``, ``ModuleDict`` and ``TrainState``
primitives, straight-line conditional flow-matching loss, Adam update path,
RNG splitting, and Euler sampler. Only the tensor contract changes from Toy
Give-Way's 20-condition/4-action policy to a 72-condition/8-action policy.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, Mapping, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax
import flax.serialization
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
MACFLOW_OFFICIAL = RESEARCH_ROOT / "01_MACFlow_Baseline_Reproduction" / "MACFlow_Official"
if not MACFLOW_OFFICIAL.is_dir():
    raise FileNotFoundError(f"MACFlow_Official source tree not found: {MACFLOW_OFFICIAL}")
if str(MACFLOW_OFFICIAL) not in sys.path:
    sys.path.insert(0, str(MACFLOW_OFFICIAL))

# These are the same classes imported by flowbc/giveway_flowbc_agent.py.
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField


CHECKPOINT_SCHEMA = "double_bottleneck_macflow_stage_i_joint_4a_v1"
AGENT_ORDER = ("A1", "A2", "B1", "B2")
NUM_AGENTS = 4
OBS_DIM = 18
ACT_DIM = 2
JOINT_OBS_DIM = NUM_AGENTS * OBS_DIM
JOINT_ACT_DIM = NUM_AGENTS * ACT_DIM


class DoubleBottleneckFlowBCAgent(flax.struct.PyTreeNode):
    """True joint 4-agent Stage-I policy using MACFlow official primitives."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _flatten(
        self, observations: jnp.ndarray, actions: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """``[B,4,18] -> [B,72]`` and ``[B,4,2] -> [B,8]``."""

        batch = observations.shape[0]
        obs_flat = observations.reshape(batch, JOINT_OBS_DIM)
        act_flat = actions.reshape(batch, JOINT_ACT_DIM)
        if self.config.get("normalize", False):
            obs_flat = (obs_flat - jnp.asarray(self.config["obs_mean"])) / jnp.asarray(
                self.config["obs_scale"]
            )
            act_flat = (act_flat - jnp.asarray(self.config["act_mean"])) / jnp.asarray(
                self.config["act_scale"]
            )
        return obs_flat, act_flat

    def flow_bc_samples(self, batch: Dict, rng) -> Dict:
        """Toy-identical CFM construction and RNG split in eight dimensions."""

        _, x_rng, t_rng = jax.random.split(rng, 3)
        obs_flat, act_flat = self._flatten(
            jnp.asarray(batch["observations"]), jnp.asarray(batch["actions"])
        )
        x_0 = jax.random.normal(x_rng, act_flat.shape)
        x_1 = act_flat
        time = jax.random.uniform(t_rng, (obs_flat.shape[0], 1))
        x_t = (1.0 - time) * x_0 + time * x_1
        velocity = x_1 - x_0
        return dict(
            X_t=obs_flat,
            h_0=x_0,
            h_1=x_1,
            s=time,
            h_s=x_t,
            v_fwd=velocity,
        )

    def flow_bc_loss(self, batch: Dict, grad_params, rng):
        samples = self.flow_bc_samples(batch, rng)
        obs_flat, x_t, time, velocity = (
            samples[key] for key in ("X_t", "h_s", "s", "v_fwd")
        )
        prediction = self.network.select("actor_bc_flow")(
            obs_flat, x_t, time, params=grad_params
        )
        loss = jnp.mean((prediction - velocity) ** 2)
        return loss, {"bc_flow_loss": loss}

    @jax.jit
    def total_loss(self, batch: Dict, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        _, flow_rng = jax.random.split(rng)
        return self.flow_bc_loss(batch, grad_params, flow_rng)

    @jax.jit
    def update(self, batch: Dict, step: int):
        del step
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray, seed) -> jnp.ndarray:
        """Toy-identical Euler sampler returning joint actions ``[B,4,2]``."""

        batch = observations.shape[0]
        obs_flat = observations.reshape(batch, JOINT_OBS_DIM)
        if self.config.get("normalize", False):
            obs_flat = (obs_flat - jnp.asarray(self.config["obs_mean"])) / jnp.asarray(
                self.config["obs_scale"]
            )
        actions = jax.random.normal(seed, (batch, JOINT_ACT_DIM))
        steps = self.config["flow_steps"]
        for index in range(steps):
            time = jnp.full((batch, 1), index / steps)
            velocity = self.network.select("actor_bc_flow")(obs_flat, actions, time)
            actions = actions + velocity / steps
        if self.config.get("normalize", False):
            actions = actions * jnp.asarray(self.config["act_scale"]) + jnp.asarray(
                self.config["act_mean"]
            )
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions.reshape(batch, NUM_AGENTS, ACT_DIM)

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: ml_collections.ConfigDict,
    ) -> "DoubleBottleneckFlowBCAgent":
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        batch = ex_observations.shape[0]
        ex_obs_flat = ex_observations.reshape(batch, JOINT_OBS_DIM)
        ex_act_flat = ex_actions.reshape(batch, JOINT_ACT_DIM)
        ex_times = jnp.zeros((batch, 1), dtype=jnp.float32)

        bc_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=JOINT_ACT_DIM,
            layer_norm=config["actor_layer_norm"],
            encoder=None,
        )
        network_info = {
            "actor_bc_flow": (bc_flow_def, (ex_obs_flat, ex_act_flat, ex_times))
        }
        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        normalization = {}
        if config.get("normalize", False):
            normalization = {
                key: tuple(config[key]) if key != "normalize" else config[key]
                for key in ("normalize", "obs_mean", "obs_scale", "act_mean", "act_scale")
            }
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(
                **normalization,
                actor_hidden_dims=tuple(config["actor_hidden_dims"]),
                actor_layer_norm=config["actor_layer_norm"],
                lr=config["lr"],
                flow_steps=config["flow_steps"],
                environment_fingerprint=config["environment_fingerprint"],
                max_speed=config["max_speed"],
                fixed_geometry=True,
                agent_order=AGENT_ORDER,
                use_critic=False,
                use_q_guidance=False,
                use_distillation=False,
                critic_loss_weight=0.0,
                q_guidance_weight=0.0,
                distillation_weight=0.0,
            ),
        )


def get_config() -> ml_collections.ConfigDict:
    """Toy-identical MACFlow network/training defaults plus scenario guards."""

    return ml_collections.ConfigDict(
        dict(
            lr=3e-4,
            actor_hidden_dims=(256, 256, 256),
            actor_layer_norm=False,
            flow_steps=10,
            use_critic=False,
            use_q_guidance=False,
            use_distillation=False,
            critic_loss_weight=0.0,
            q_guidance_weight=0.0,
            distillation_weight=0.0,
            environment_fingerprint="",
            max_speed=0.5,
        )
    )


def speed_bound(actions, max_speed: float):
    """Per-agent radial plant bound; not a collision-safety filter."""

    actions = jnp.asarray(actions)
    norms = jnp.linalg.norm(actions, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, float(max_speed) / jnp.maximum(norms, 1e-12))
    return actions * scale


def sample_bounded_actions(agent, observations, seed):
    return speed_bound(agent.sample_actions(observations, seed), agent.config["max_speed"])


def parameter_count(agent: DoubleBottleneckFlowBCAgent) -> int:
    return int(
        sum(
            np.asarray(leaf).size
            for leaf in jax.tree_util.tree_leaves(agent.network.params)
        )
    )


def save_checkpoint(
    path: Path,
    agent: DoubleBottleneckFlowBCAgent,
    metadata: Mapping[str, Any],
) -> None:
    """Persist the same Flax agent state format used by Toy Give-Way."""

    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "agent": flax.serialization.to_state_dict(agent),
        "config": dict(agent.config),
        "metadata": dict(metadata),
    }
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(
    path: Path,
    expected_environment_fingerprint: str,
) -> tuple[DoubleBottleneckFlowBCAgent, dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("not a MACFlow-official joint 4-agent checkpoint")
    config = get_config()
    config.update(payload["config"])
    if config["environment_fingerprint"] != expected_environment_fingerprint:
        raise ValueError(
            "checkpoint/environment fingerprint mismatch: "
            f"{config['environment_fingerprint']} != {expected_environment_fingerprint}"
        )
    agent = DoubleBottleneckFlowBCAgent.create(
        0,
        jnp.zeros((1, NUM_AGENTS, OBS_DIM), dtype=jnp.float32),
        jnp.zeros((1, NUM_AGENTS, ACT_DIM), dtype=jnp.float32),
        config,
    )
    agent = flax.serialization.from_state_dict(agent, payload["agent"])
    probe = np.asarray(
        agent.sample_actions(
            jnp.zeros((1, NUM_AGENTS, OBS_DIM), dtype=jnp.float32),
            jax.random.PRNGKey(0),
        )
    )
    if probe.shape != (1, NUM_AGENTS, ACT_DIM) or not np.isfinite(probe).all():
        raise ValueError("checkpoint failed joint action shape/numeric validation")
    return agent, dict(payload.get("metadata", {}))


__all__ = (
    "ACT_DIM",
    "AGENT_ORDER",
    "ActorVectorField",
    "CHECKPOINT_SCHEMA",
    "DoubleBottleneckFlowBCAgent",
    "JOINT_ACT_DIM",
    "JOINT_OBS_DIM",
    "MACFLOW_OFFICIAL",
    "ModuleDict",
    "NUM_AGENTS",
    "OBS_DIM",
    "TrainState",
    "get_config",
    "load_checkpoint",
    "parameter_count",
    "sample_bounded_actions",
    "save_checkpoint",
    "speed_bound",
)

"""GiveWay-v1 Stage-I Joint Flow-BC agent.

Reuses ActorVectorField / ModuleDict / TrainState from MACFlow_Official.
Only implements Stage-I Flow Matching (bc_flow_loss).
Critic, Q guidance, and distillation are truly absent (no network construction).

Joint obs/action convention:
  observations [B,2,10]  →  obs_flat  [B,20]
  actions      [B,2,2]   →  act_flat  [B, 4]
"""
from typing import Any, Dict, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

# Imported from MACFlow_Official (caller must add that directory to sys.path).
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField

NUM_AGENTS  = 2
OBS_DIM     = 10
ACT_DIM     = 2
JOINT_OBS   = NUM_AGENTS * OBS_DIM   # 20
JOINT_ACT   = NUM_AGENTS * ACT_DIM   # 4


class GiveWayFlowBCAgent(flax.struct.PyTreeNode):
    """Stage-I Flow-BC agent for joint GiveWay actions.

    Flags (all False → only bc_flow_loss is computed):
      use_critic         = False
      use_q_guidance     = False
      use_distillation   = False
    """
    rng:     Any
    network: Any
    config:  Any = nonpytree_field()

    # ------------------------------------------------------------------
    def _flatten(self, obs: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """[B,2,10] → [B,20]  and  [B,2,2] → [B,4]."""
        B = obs.shape[0]
        obs_flat = obs.reshape(B, JOINT_OBS)
        act_flat = actions.reshape(B, JOINT_ACT)
        if self.config.get("normalize", False):
            obs_flat = (obs_flat - jnp.asarray(self.config["obs_mean"])) / jnp.asarray(self.config["obs_scale"])
            act_flat = (act_flat - jnp.asarray(self.config["act_mean"])) / jnp.asarray(self.config["act_scale"])
        if self.config.get("position_only", False):
            obs_flat = obs_flat * jnp.asarray([1,1,0,0,1,1,1,1,0,0] * NUM_AGENTS)
        return obs_flat, act_flat

    # ------------------------------------------------------------------
    def flow_bc_samples(self, batch: Dict, rng) -> Dict:
        """Original CFM construction shared verbatim with C1 (same RNG splits)."""
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        obs_flat, act_flat = self._flatten(
            jnp.asarray(batch["observations"]),
            jnp.asarray(batch["actions"]),
        )

        x_0  = jax.random.normal(x_rng, act_flat.shape)             # [B,4] noise
        x_1  = act_flat                                               # [B,4] data
        t    = jax.random.uniform(t_rng, (obs_flat.shape[0], 1))    # [B,1]
        x_t  = (1.0 - t) * x_0 + t * x_1                            # [B,4]
        vel  = x_1 - x_0                                              # [B,4]

        return dict(X_t=obs_flat, h_0=x_0, h_1=x_1, s=t, h_s=x_t, v_fwd=vel)

    def flow_bc_loss(self, batch: Dict, grad_params, rng) -> Tuple[jnp.ndarray, Dict]:
        """Stage-I conditional Flow Matching loss (MAC eq. 3)."""
        samples = self.flow_bc_samples(batch, rng)
        obs_flat, x_t, t, vel = (samples[k] for k in ('X_t','h_s','s','v_fwd'))

        pred = self.network.select("actor_bc_flow")(
            obs_flat, x_t, t, params=grad_params
        )  # [B,4]
        loss = jnp.mean((pred - vel) ** 2)

        return loss, {"bc_flow_loss": loss}

    @jax.jit
    def total_loss(self, batch: Dict, grad_params, rng=None) -> Tuple[jnp.ndarray, Dict]:
        rng = self.rng if rng is None else rng
        rng, flow_rng = jax.random.split(rng)

        # Only Stage-I flow loss; critic / Q / distillation are skipped entirely.
        loss, info = self.flow_bc_loss(batch, grad_params, flow_rng)
        return loss, info

    @jax.jit
    def update(self, batch: Dict, step: int) -> Tuple["GiveWayFlowBCAgent", Dict]:
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray, seed) -> jnp.ndarray:
        """Euler integration over bc_flow model.  Returns actions [B,2,2]."""
        B = observations.shape[0]
        obs_flat = observations.reshape(B, JOINT_OBS)
        if self.config.get("normalize", False):
            obs_flat = (obs_flat - jnp.asarray(self.config["obs_mean"])) / jnp.asarray(self.config["obs_scale"])
        if self.config.get("position_only", False):
            obs_flat = obs_flat * jnp.asarray([1,1,0,0,1,1,1,1,0,0] * NUM_AGENTS)
        noise    = jax.random.normal(seed, (B, JOINT_ACT))
        actions  = noise
        steps    = self.config["flow_steps"]
        for i in range(steps):
            t    = jnp.full((B, 1), i / steps)
            vels = self.network.select("actor_bc_flow")(obs_flat, actions, t)
            actions = actions + vels / steps
        if self.config.get("normalize", False):
            actions = actions * jnp.asarray(self.config["act_scale"]) + jnp.asarray(self.config["act_mean"])
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions.reshape(B, NUM_AGENTS, ACT_DIM)   # [B,2,2]

    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        seed:            int,
        ex_observations: jnp.ndarray,   # [B,2,10]  example batch
        ex_actions:      jnp.ndarray,   # [B,2,2]   example batch
        config:          ml_collections.ConfigDict,
    ) -> "GiveWayFlowBCAgent":
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))

        B = ex_observations.shape[0]
        ex_obs_flat = ex_observations.reshape(B, JOINT_OBS)   # [B,20]
        ex_act_flat = ex_actions.reshape(B, JOINT_ACT)        # [B,4]
        ex_times    = jnp.zeros((B, 1), dtype=jnp.float32)   # [B,1]

        bc_flow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=JOINT_ACT,
            layer_norm=config["actor_layer_norm"],
            encoder=None,
        )

        # Only the bc_flow network is constructed; no critic / onestep_flow.
        network_info = {
            "actor_bc_flow": (bc_flow_def, (ex_obs_flat, ex_act_flat, ex_times)),
        }
        networks     = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def    = ModuleDict(networks)
        network_tx     = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network        = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(
                **({'position_only': True} if config.get('position_only', False) else {}),
                **({key: (tuple(config[key]) if key != "normalize" else config[key])
                    for key in ["normalize", "obs_mean", "obs_scale", "act_mean", "act_scale"]}
                   if config.get("normalize", False) else {}),
                actor_hidden_dims  = config["actor_hidden_dims"],
                actor_layer_norm   = config["actor_layer_norm"],
                lr                 = config["lr"],
                flow_steps         = config["flow_steps"],
                # Disabled modules (reserved for future stages)
                use_critic         = False,
                use_q_guidance     = False,
                use_distillation   = False,
                critic_loss_weight = 0.0,
                q_guidance_weight  = 0.0,
                distillation_weight= 0.0,
            ),
        )


def get_config() -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(dict(
        lr                 = 3e-4,
        actor_hidden_dims  = (256, 256, 256),
        actor_layer_norm   = False,
        flow_steps         = 10,
        # Stage-II / critic flags (all False → Stage-I only)
        use_critic         = False,
        use_q_guidance     = False,
        use_distillation   = False,
        critic_loss_weight = 0.0,
        q_guidance_weight  = 0.0,
        distillation_weight= 0.0,
    ))

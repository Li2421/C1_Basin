"""Alternate Ring MACFlow checkpoint under the unchanged native FF plant.

The checkpoint's environment fingerprint selects its observation/action chart.
Only the policy checkpoint and chart differ; physical dynamics, FF basis,
projection and evaluator are inherited from the archived Ring Runtime.
"""
import sys
from pathlib import Path

import numpy as np

FIELD = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
sys.path.insert(0, str(FIELD))

import jax
import jax.numpy as jnp
from poc.field_controller import Runtime
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import HardProjectionConfig
from ring_exchange.environment import Config, LocalFrameConfig, RingExchangeEnv


class RingFlowAdapter(Runtime):
    def __init__(self, checkpoint, manifest, frame):
        if frame not in ('local', 'global'):
            raise ValueError(frame)
        self.scenario = 'ring_exchange'
        self.frame = frame
        self.env_cls = RingExchangeEnv
        self.config_cls = LocalFrameConfig if frame == 'local' else Config
        self.manifest = manifest
        self.config = self.config_cls(**manifest['scenario_config'])
        self.checkpoint = Path(checkpoint)
        self.agent, _ = load_checkpoint(self.checkpoint, expected_environment_fingerprint=manifest['environment_fingerprint'])
        c = self.agent.config
        assert c['joint_obs_dim'] == 92 and c['joint_act_dim'] == 8
        assert c['flow_steps'] == 10 and c['normalize'] and c['num_agents'] == 4
        self.basis = get_basis_family('orthoflow3')
        self.cbf = HardProjectionConfig()
        self.steps = int(c['flow_steps'])
        self.scale = np.asarray(c['act_scale'], dtype=np.float64).reshape(-1, 2)
        self.mean = np.asarray(c['act_mean'], dtype=np.float64).reshape(-1, 2)
        if np.any(self.scale <= 0):
            raise ValueError('Invalid action normalization')

        @jax.jit
        def trace(obs, key):
            flat = obs.reshape(1, c['joint_obs_dim'])
            flat = (flat - jnp.asarray(c['obs_mean'])) / jnp.asarray(c['obs_scale'])
            z = jax.random.normal(key, (1, c['joint_act_dim']))
            path = [z[0]]
            for step in range(self.steps):
                t = jnp.full((1, 1), step/self.steps)
                z = z + self.agent.network.select('actor_bc_flow')(flat, z, t)/self.steps
                path.append(z[0])
            return jnp.stack(path)

        @jax.jit
        def velocity(obs, z, s):
            flat = obs.reshape(1, c['joint_obs_dim'])
            flat = (flat - jnp.asarray(c['obs_mean'])) / jnp.asarray(c['obs_scale'])
            return self.agent.network.select('actor_bc_flow')(flat, z.reshape(1, -1), s.reshape(1, 1))[0]

        self.trace_fn = trace
        self.velocity_fn = velocity
        self.native_fn = jax.jit(lambda obs, key: sample_bounded_actions(self.agent, obs[None], key)[0])

    def observation(self, env):
        if self.frame == 'global':
            return env.observation()
        from ring_exchange.local_frame import local_observation
        return local_observation(env.positions, env.velocities, env.goals, env.config)

    def to_world(self, actions, env):
        if self.frame == 'global':
            return np.asarray(actions)
        from ring_exchange.local_frame import local_actions_to_world
        return local_actions_to_world(actions, env.positions)

    def to_local(self, actions, env):
        if self.frame == 'global':
            return np.asarray(actions)
        from ring_exchange.local_frame import world_actions_to_local
        return world_actions_to_local(actions, env.positions)

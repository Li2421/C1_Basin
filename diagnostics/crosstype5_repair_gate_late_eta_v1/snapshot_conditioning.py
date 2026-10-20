"""Read-only conditioning of exact intermediate states via public restore."""
from common import *
from legacy_conditioning import ReadOnlyConditioning,LocalFrameConfig,RingExchangeEnv,load_checkpoint
class SnapshotConditioning(ReadOnlyConditioning):
    def __init__(self):
        self.name='ring_exchange';self.kind='ring';self.manifest=read(ROOT/'diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json');self.config_dict=self.manifest['scenario_config'];self.config=LocalFrameConfig(**self.config_dict);self.agent,self.checkpoint_metadata=load_checkpoint(ROOT/'diagnostics/ring_exchange_stage1/base_u_v10_local_macflow/best.pkl',expected_environment_fingerprint=self.manifest['environment_fingerprint']);self.make_env=lambda:RingExchangeEnv(self.config)
    def reset(self,env,state):
        p=state['physical']
        # This reference environment only supplies observations/Flow. Its
        # public restore API preserves the exact valid simulator snapshot,
        # including tiny post-action speed overshoot; never clip physical data.
        env.restore_augmented_state(dict(positions=p['positions'],last_applied_velocity=p['velocities'],goals=p['goals'],config=self.config.to_dict(),step=int(p.get('step',0)),done=False,first_success_step=None,first_obstacle_collision_step=None,first_outer_collision_step=None,first_agent_collision_step=None))
    def conditioning(self,state):
        # Same FrozenNewRuntime.conditioning computation after exact restore.
        from legacy_conditioning import state_token,CONDITIONING_FLOW_ROOT,_environment_descriptor
        import jax
        env=self.make_env();self.reset(env,state);obs=np.asarray(self.observation(env),float);key=jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT),state_token(state['uid']));flow=self.flow_world(env,key)
        return np.concatenate((obs.reshape(-1),flow.reshape(-1))).tolist(),_environment_descriptor(self)

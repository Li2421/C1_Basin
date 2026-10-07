"""Test whether critic's agent-relabeling symmetry holds for the frozen Flow.

Action-only, no task rollout. Permute the initial latent as well as agents to
avoid mistaking a changed random draw for a controller symmetry violation.
"""
import copy,hashlib,json,os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import jax,jax.numpy as jnp
import numpy as np
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from new_benchmark_common.macflow import load_checkpoint
from ring_exchange.local_frame import local_observation,local_actions_to_world
from ring_exchange.environment import Config

OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'orthoflow3_source_contrast_interaction_v1'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def main():
    jax.config.update('jax_enable_x64',False)
    proto=read(SRC/'protocol.json');states=read(SRC/'states.json');physical=read(SRC/'physical.json')
    indices=sorted(range(46),key=lambda i:hashlib.sha256(('symmetry_action_audit_v1'+states[i]['source_group']).encode()).hexdigest())[:12]
    cfg=Config(**json.loads((OUT.parent/'ring_exchange_stage1/base_u_v10_local_dataset/manifest.json').read_text())['scenario_config'])
    perms=[np.array([1,2,3,0]),np.array([3,2,1,0])]
    model=rep.Critic();x=rep.batch([rep.entities(physical[indices[0]]['physical'])]);params=model.init(jax.random.PRNGKey(17),x,jnp.zeros((1,3)))
    encoder_errors=[];rows=[]
    for profile in proto['profiles']:
        agent,_=load_checkpoint(profile['path'],expected_environment_fingerprint=proto['environment_fingerprint'])
        @jax.jit
        def sample(obs,latent):
            ob=obs.reshape(len(obs),agent.config['joint_obs_dim'])
            if agent.config['normalize']:ob=(ob-jnp.array(agent.config['obs_mean']))/jnp.array(agent.config['obs_scale'])
            actions=latent
            for k in range(agent.config['flow_steps']):
                tt=jnp.full((len(obs),1),k/agent.config['flow_steps'])
                actions=actions+agent.network.select('actor_bc_flow')(ob,actions,tt)/agent.config['flow_steps']
            if agent.config['normalize']:actions=actions*jnp.array(agent.config['act_scale'])+jnp.array(agent.config['act_mean'])
            actions=jnp.clip(actions,-1.,1.).reshape(-1,4,2)
            return actions*jnp.minimum(1.,cfg.max_speed/jnp.maximum(jnp.linalg.norm(actions,axis=-1,keepdims=True),1e-12))
        for index in indices:
            ph=physical[index]['physical'];p,v,g=[np.array(ph[k]) for k in ('positions','velocities','goals')]
            obs=local_observation(p,v,g,cfg);seed=jax.random.PRNGKey(55000+index)
            latent=jax.random.normal(seed,(128,8))
            a=np.array(sample(jnp.asarray(np.repeat(obs[None],128,0)),latent))
            # Same native sampler, latent, and observations must agree exactly.
            native=np.array(agent.sample_actions(jnp.asarray(np.repeat(obs[None],128,0)),seed))
            native=native*np.minimum(1,cfg.max_speed/np.maximum(np.linalg.norm(native,axis=-1,keepdims=True),1e-12))
            np.testing.assert_allclose(a,native,atol=2e-7,rtol=0)
            for perm in perms:
                changed=rep.transform(ph,permutation=perm)
                xa=rep.batch([rep.entities(ph)]);xb=rep.batch([rep.entities(changed)])
                za=np.array(model.apply(params,xa,jnp.zeros((1,3))));zb=np.array(model.apply(params,xb,jnp.zeros((1,3))))
                encoder_errors.append(float(np.max(abs(za-zb))))
                ob=local_observation(p[perm],v[perm],g[perm],cfg)
                lp=latent.reshape(128,4,2)[:,perm,:].reshape(128,8)
                b=np.array(sample(jnp.asarray(np.repeat(ob[None],128,0)),lp))[:,np.argsort(perm)]
                delta=a-b
                rows.append(dict(controller=profile['name'],state_uid=states[index]['uid'],permutation=perm.tolist(),
                    mean_action_difference_L2=float(np.linalg.norm(a.mean(0)-b.mean(0))),
                    per_agent_action_difference_mean=float(np.linalg.norm(delta,axis=-1).mean()),
                    matched_latent_RMS=float(np.sqrt(np.mean(delta**2))),
                    per_agent_mean_bias=float(np.linalg.norm(a.mean(0)-b.mean(0),axis=-1).mean())))
    result=dict(task_rollouts=0,states=12,controller_count=2,latent_samples=128,
        critic_permutation_max_error=max(encoder_errors),
        native_sampler_reproduction_passed=True,
        flow_permutation_RMS_median=float(np.median([r['matched_latent_RMS'] for r in rows])),
        flow_mean_action_bias_median=float(np.median([r['per_agent_mean_bias'] for r in rows])),
        max_speed=cfg.max_speed,
        verdict='FROZEN_FLOW_NOT_AGENT_PERMUTATION_EQUIVARIANT' if np.median([r['matched_latent_RMS'] for r in rows])>1e-4 else 'ACTION_SYMMETRY_APPROXIMATELY_SUPPORTED',
        caveat='Action symmetry violation is not yet proof that Q or full (h,C) alias; task-outcome and role-preserving representation controls are required.',rows=rows)
    write(OUT/'symmetry_action_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()

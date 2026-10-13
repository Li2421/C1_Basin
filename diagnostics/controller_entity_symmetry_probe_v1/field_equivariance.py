"""Label-free deterministic native field test; no random-noise coupling ambiguity."""
import argparse,itertools,sys
from pathlib import Path
import numpy as np
from diagnostics.controller_four_scene_mechanism_v1.study import read,write,freeze,sha
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'controller_ff_factorial_repair_v1'
SCENES=('toy_give_way','double_bottleneck','four_way_intersection','ring_exchange')


def prepare():
    p=read(SOURCE/'protocol.json')
    states=[s for s in p['states'] if s['metadata']['split']=='train']
    freeze(HERE/'field_equivariance_protocol.json',dict(
        prior_checked='orthoflow3_unified_representation_v1/active_metamorphic.py/json',
        reuse_gap='Old same-key stochastic action comparison does not couple agentwise noise and cannot alone distinguish distributional non-equivariance. No prior deterministic physical field permutation test found.',
        hypothesis='F(Pstate,Pphysical_pseudo_action,t)=P F(state,physical_pseudo_action,t)',
        states=states,permutations='All nonidentity permutations, not selected by labels',
        physical_action_probes='zero, +/-0.5vmax goalward, +/-0.5vmax lateral',
        pseudo_times=[0.,.5,.9],new_rollouts=0,simulator_steps=0,success_labels_used=False,
        source_protocol_sha256=sha(SOURCE/'protocol.json'),code_sha256=sha(__file__),
        boundary='Field non-equivariance is an input/architecture mechanism; by itself it doesnot prove long-horizon Qnoninvariance or explain allLOSO failures.'))


def run(scene):
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    import jax.numpy as jnp
    p=read(HERE/'field_equivariance_protocol.json');rt=native_runtime(scene)
    def forbidden(*args,**kw):raise RuntimeError('No simulator step allowed in static field diagnostic')
    rt.env_cls.step=forbidden
    states=[s for s in p['states'] if s['scenario']==scene]
    actual=[];expected=[];metadata=[]
    def field(env,h,t):
        z=(rt.to_local(h,env)-rt.mean)/rt.scale
        obs=rt.observation(env)
        f=np.asarray(rt.velocity_fn(jnp.asarray(obs),jnp.asarray(z),jnp.asarray(t))).reshape(-1,2)
        return rt.to_world(rt.scale*f,env)
    for si,s in enumerate(states):
        env=rt.make_env(s);pp,vv,gg=[getattr(env,k).copy() for k in ('positions','velocities','goals')]
        n=len(pp);e=gg-pp;e/=np.maximum(np.linalg.norm(e,axis=-1,keepdims=True),1e-12)
        lat=np.stack((-e[:,1],e[:,0]),-1);scale=.5*rt.config.max_speed
        probes=[np.zeros_like(pp),scale*e,-scale*e,scale*lat,-scale*lat]
        original={(h,t):field(env,hh,t) for h,hh in enumerate(probes) for t in p['pseudo_times']}
        for perm in itertools.permutations(range(n)):
            if perm==tuple(range(n)):continue
            perm=np.array(perm)
            for k,v in zip(('positions','velocities','goals'),(pp,vv,gg)):
                getattr(env,k)[:]=v[perm]
            for h,hh in enumerate(probes):
                for t in p['pseudo_times']:
                    expected.append(original[h,t][perm]);actual.append(field(env,hh[perm],t))
                    metadata.append(dict(state_uid=s['state_uid'],state_index=si,permutation=perm.tolist(),probe=h,pseudo_time=t))
    expected,actual=np.array(expected),np.array(actual);error=np.max(abs(expected-actual),(1,2))/rt.config.max_speed
    np.savez_compressed(HERE/f'field_permutation_{scene}.npz',expected=expected,actual=actual,error_over_max_speed=error)
    write(HERE/f'field_permutation_{scene}_cases.json',metadata)
    write(HERE/f'field_permutation_{scene}_summary.json',dict(scene=scene,physical_states=len(states),cases=len(error),
        mean_error_over_max_speed=float(error.mean()),median_error_over_max_speed=float(np.median(error)),
        max_error_over_max_speed=float(error.max()),fraction_above_1e_5=float(np.mean(error>1e-5)),
        checkpoint=str(rt.checkpoint),checkpoint_sha256=sha(rt.checkpoint),new_rollouts=0,simulator_steps=0,
        no_random_sampler_called=True,expected_and_actual_use_identical_physical_action_after_reindexing=True))
    print(read(HERE/f'field_permutation_{scene}_summary.json'),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--index',type=int);a=ap.parse_args()
    prepare() if a.stage=='prepare' else run(SCENES[a.index])

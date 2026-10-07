"""Independent orchestration of fresh continuations using frozen plant primitives."""
import json
import time
import numpy as np
import jax
import jax.numpy as jnp

from diagnostics.astra_true_q_audit.audit import OUT, OLD, read, write, sha, restore, comparison
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from single_integrator.environment import Config, bounded_nominal
from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from single_integrator.evaluate import load_policy

def main():
    jax.config.update('jax_enable_x64',True)
    assert jax.default_backend()=='gpu',jax.devices()
    plan=read(OUT/'replication_plan.json'); locked=sha(OUT/'replication_plan.json')
    cat={s['state_id']:s for s in read(OLD/'raw/q_map/state_catalog.json')}
    cfg=Config(**plan['environment']); cbf=CBFConfig()
    policy,_=load_policy(plan['source_checkpoint'])
    raw=OUT/'raw'; raw.mkdir(exist_ok=False)
    print(json.dumps({'states':plan['states'],'seed_count_per_arm':32,'proposed_rollouts':192,
        'expected_max_physical_steps':plan['max_steps'],'GPU_CPU_plan':plan['backend'],'devices':[str(d) for d in jax.devices()] }),flush=True)
    records=[]; begin=time.monotonic()
    for spec in plan['states']:
        sid=spec['state_id']; item=cat[sid]
        with np.load(OLD/'raw/q_map'/item['state_file']) as s: state=dict(s)
        for seed in plan['seeds']:
            for arm in ['phi_minus','phi_plus']:
                env=restore(state,cfg); corrector=DiagnosticCorrector(DiagnosticPhi(*spec[arm]))
                key0=jax.random.fold_in(jax.random.PRNGKey(seed),item['pair_id'])
                fields={k:[] for k in ['positions_before','positions_after','g','w','u_flow','u_safe','u_exec','event','second_active','flow_key_data','timer']}
                while not env.done:
                    obs=env.observation(); before=env.positions.copy(); key=jax.random.fold_in(key0,env.step_count)
                    nominal=bounded_nominal(np.asarray(policy.sample_actions(jnp.asarray(obs[None]),seed=key)[0]),cfg.max_speed)
                    A,b,_=barrier_constraints(env.snapshot(),cbf)
                    safe,_=project_velocity(nominal,A,b,cfg.max_speed,cbf)
                    correction=corrector(obs,safe,cfg.max_speed); w=safe+correction
                    executed,_=project_velocity(w,A,b,cfg.max_speed,cbf)
                    assert np.min(A@executed.ravel()-b)>-1e-8
                    _,_,_,info=env.step(executed)
                    active=np.r_[A@executed.ravel()-b<=1e-6,abs(np.linalg.norm(executed,axis=-1)-cfg.max_speed)<=1e-6]
                    values=[before,env.positions.copy(),correction,w,nominal,safe,executed,info['termination'],active,np.asarray(jax.random.key_data(key)),env.stuck_timer]
                    for name,v in zip(fields,values):fields[name].append(v)
                n=len(fields['event']); assert corrector.call_count==n
                trace={k:np.asarray(v) for k,v in fields.items()}
                trace.update(phi=np.asarray(spec[arm]),outcome=info['termination'],start_step=int(state['step']),terminal_step=env.step_count,flow_seed=seed)
                tid=f'{sid}__{arm}__{seed}'; path=raw/f'{tid}.npz'; np.savez_compressed(path,**trace)
                records.append({'state_id':sid,'arm':arm,'phi':spec[arm],'flow_seed':seed,'outcome':info['termination'],
                    'steps':n,'relative_path':str(path.relative_to(OUT)),'sha256':sha(path)})
                if len(records)%16==0:print(f'Completed {len(records)}/192, elapsed {time.monotonic()-begin:.1f}s',flush=True)
    assert sha(OUT/'replication_plan.json')==locked
    stats=[]
    for s in plan['states']:
        sid=s['state_id']; a=[r for r in records if r['state_id']==sid and r['arm']=='phi_minus']; b=[r for r in records if r['state_id']==sid and r['arm']=='phi_plus']
        stat=comparison(a,b); stat.update(state_id=sid,phi_low=s['phi_minus'],phi_high=s['phi_plus'])
        stat['replicates']=stat['conservative_95_difference_interval'][0]>0 and stat['paired_exact_p']<.05/3
        stats.append(stat)
    result={'plan_sha256':locked,'device':[str(d) for d in jax.devices()],'new_rollouts':len(records),
        'actual_physical_steps':sum(r['steps'] for r in records),'elapsed_seconds':time.monotonic()-begin,
        'comparisons':stats,'records':records,'all_three_replicate':all(r['replicates'] for r in stats),
        'uncertainty':'Clopper-Pearson marginal 95%; paired exact test; difference CI from Bonferroni 97.5% marginal intervals.'}
    write('fresh_replication.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)

if __name__=='__main__':main()

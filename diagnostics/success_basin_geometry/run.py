"""Batched Flow inference with frozen exact per-trajectory CPU plant/projections."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from diagnostics.astra_true_q_audit.audit import restore
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector,DiagnosticPhi
from single_integrator.environment import Config,bounded_nominal
from single_integrator.cbf import CBFConfig,CBFSolverError,barrier_constraints,project_velocity
from single_integrator.evaluate import load_policy
from diagnostics.success_basin_geometry.setup import HERE,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--stage',required=True);ap.add_argument('--device',choices=['cpu','gpu'],required=True);ap.add_argument('--batch',type=int,default=32);args=ap.parse_args()
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_platform_name',args.device)
    protocol=json.loads((HERE/'protocol.json').read_text());locked=sha(HERE/'protocol.json')
    jobs=json.loads((HERE/args.jobs).read_text());cat={s['state_id']:s for s in protocol['state_catalog']}
    cfg=Config(**protocol['environment']);cbf=CBFConfig();policy,_=load_policy(Path(protocol['checkpoint']))
    raw=HERE/'raw'/args.stage;raw.mkdir(parents=True,exist_ok=False)
    print(json.dumps({'stage':args.stage,'proposed_new_rollouts':len(jobs),'expected_max_physical_steps':sum(cfg.max_steps-cat[j['state_id']]['start_step'] for j in jobs),'device':args.device,'batch':args.batch,'purpose':'Full-horizon success/outcome geometry; same state-feedback phi active until first event'}),flush=True)
    single=lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]
    sample=jax.jit(jax.vmap(single))
    fold=jax.jit(jax.vmap(jax.random.fold_in))
    # Numerical-only batch-vs-scalar check; no rollout and no threshold tuning.
    probe=np.zeros((args.batch,2,10),dtype=np.float32);keys=jax.random.split(jax.random.PRNGKey(901),args.batch)
    bp=np.asarray(sample(jnp.asarray(probe),keys));sp=np.asarray(single(jnp.asarray(probe[0]),keys[0]))
    batch_error=float(np.max(abs(bp[0]-sp)));assert batch_error<1e-6,batch_error
    records=[];total_steps=0;started=time.monotonic();last=time.monotonic()
    for chunk in range(0,len(jobs),args.batch):
        tasks=jobs[chunk:chunk+args.batch];envs=[];correctors=[];hist=[];key0=[];failures=[None]*len(tasks)
        for task in tasks:
            entry=cat[task['state_id']]
            with np.load(HERE/entry['state_file']) as state:envs.append(restore(dict(state),cfg))
            correctors.append(DiagnosticCorrector(DiagnosticPhi(*task['phi'])))
            hist.append({k:[] for k in ['positions_before','positions_after','u_flow','u_safe','g','w','u_exec','active','timer','event']})
            key0.append(np.asarray(jax.random.fold_in(jax.random.PRNGKey(task['seed']),entry['pair_id'])))
        while len(key0)<args.batch:key0.append(key0[-1])
        keys0=jnp.asarray(np.array(key0));obs=np.zeros((args.batch,2,10),dtype=np.float32);steps=np.zeros(args.batch,dtype=np.uint32)
        while any(not e.done and failures[i] is None for i,e in enumerate(envs)):
            for i,env in enumerate(envs):obs[i]=env.observation();steps[i]=env.step_count
            acts=np.asarray(sample(jnp.asarray(obs),fold(keys0,jnp.asarray(steps))))
            for i,env in enumerate(envs):
                if env.done or failures[i] is not None:continue
                before=env.positions.copy();flow=bounded_nominal(acts[i],cfg.max_speed)
                try:
                    A,b,_=barrier_constraints(env.snapshot(),cbf);safe,_=project_velocity(flow,A,b,cfg.max_speed,cbf)
                    g=correctors[i](obs[i],safe,cfg.max_speed);w=safe+g;u,_=project_velocity(w,A,b,cfg.max_speed,cbf)
                except CBFSolverError as exc:
                    failures[i]={'type':'CBFSolverError','message':str(exc),'absolute_step':env.step_count}
                    continue
                _,_,_,info=env.step(u)
                active=np.r_[A@u.ravel()-b<=1e-6,abs(np.linalg.norm(u,axis=-1)-cfg.max_speed)<=1e-6]
                vals=[before,env.positions.copy(),flow,safe,g,w,u,active,env.stuck_timer,info['termination']]
                for k,v in zip(hist[i],vals):hist[i][k].append(v)
        for i,task in enumerate(tasks):
            h={k:np.asarray(v) for k,v in hist[i].items()};n=len(h['event'])
            if failures[i] is None:assert n==correctors[i].call_count
            filename=f'{chunk+i:05d}.npz';np.savez_compressed(raw/filename,**h,phi=np.asarray(task['phi']),seed=task['seed'])
            rec={**task,'outcome':str(h['event'][-1]) if failures[i] is None else None,'execution_error':failures[i],
                 'steps':n,'terminal_step':envs[i].step_count,
                 'file':f'raw/{args.stage}/{filename}','sha256':sha(raw/filename)}
            records.append(rec);total_steps+=n
        # Durable partial index, permits audit of interrupted jobs without overwriting results.
        (raw/'partial.json').write_text(json.dumps(records))
        if time.monotonic()-last>20 or len(records)==len(jobs):
            totals={e:sum(r['outcome']==e for r in records) for e in ['success','deadlock','timeout','collision']}
            totals['unresolved_solver_failures']=sum(r['outcome'] is None for r in records)
            print(json.dumps({'completed':len(records),'total':len(jobs),'elapsed_s':round(time.monotonic()-started,1),'outcomes':totals}),flush=True);last=time.monotonic()
    assert sha(HERE/'protocol.json')==locked
    result={'stage':args.stage,'records':records,'new_rollouts':len(records),'physical_steps':total_steps,'elapsed_s':time.monotonic()-started,
        'device':[str(d) for d in jax.devices()],'batch_size':args.batch,'batch_scalar_max_error':batch_error,'protocol_sha256':locked,'jobs_sha256':sha(HERE/args.jobs)}
    (raw/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

if __name__=='__main__':main()

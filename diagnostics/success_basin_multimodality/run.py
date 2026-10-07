"""Batched Flow, exact frozen SOCP, and authoritative serial environment rollout."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np

SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay')
sys.path.insert(0,str(SYSROOT))
import jax
import jax.numpy as jnp
from single_integrator.environment import Config,bounded_nominal
from single_integrator.cbf import CBFConfig,CBFSolverError,barrier_constraints,project_velocity
from single_integrator.evaluate import load_policy
from diagnostics.astra_true_q_audit.audit import restore
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector,DiagnosticPhi
from diagnostics.success_basin_multimodality.setup import HERE,sha
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--stage',required=True);ap.add_argument('--device',choices=['cpu','gpu'],default='cpu');ap.add_argument('--batch',type=int,default=16);args=ap.parse_args()
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_platform_name',args.device)
    p=json.loads((HERE/'protocol.json').read_text());locked=sha(HERE/'protocol.json');jobs=json.loads((HERE/args.jobs).read_text())
    cat={x['state_id']:x for x in p['primary_states']};cfg=Config(**p['environment']);cbf=CBFConfig();policy,_=load_policy(Path(p['checkpoint']))
    import inspect
    assert Path(inspect.getfile(project_velocity)).resolve()==Path(p['solver']['current_source']).resolve()
    assert sha(Path(inspect.getfile(project_velocity)))==p['solver']['current_source_sha256']
    raw=HERE/'raw'/args.stage;raw.mkdir(parents=True,exist_ok=False)
    print(json.dumps({'stage':args.stage,'rollout_count':len(jobs),
      'expected_max_physical_steps':sum(cfg.max_steps-cat[x['state_id']]['start_step'] for x in jobs),
      'CPU_GPU':args.device,'purpose':'3D eta success topology; same G_eta until first frozen terminal event',
      'projector':'identical exact SOCP, Clarabel'}),flush=True)
    single=lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]
    sample=jax.jit(jax.vmap(single));fold=jax.jit(jax.vmap(jax.random.fold_in))
    records=[];total_steps=0;started=time.monotonic();last=started
    for chunk in range(0,len(jobs),args.batch):
        tasks=jobs[chunk:chunk+args.batch];envs=[];correctors=[];hist=[];key0=[];errors=[None]*len(tasks)
        for task in tasks:
            entry=cat[task['state_id']]
            with np.load(Path('/home/zhihan/research/CL-FHCB/diagnostics/success_basin_geometry')/entry['state_file']) as s:envs.append(restore(dict(s),cfg))
            correctors.append(DiagnosticCorrector(DiagnosticPhi(*task['eta'])))
            hist.append({k:[] for k in ['positions_before','positions_after','u_flow','u_safe','g','w','u_exec','first_active','second_active','first_status','second_status','first_retry','second_retry','timer','event']})
            key0.append(np.asarray(jax.random.fold_in(jax.random.PRNGKey(task['seed']),entry['pair_id'])))
        while len(key0)<args.batch:key0.append(key0[-1])
        keys0=jnp.asarray(np.asarray(key0));obs=np.zeros((args.batch,2,10),np.float32);steps=np.zeros(args.batch,np.uint32)
        while any(not e.done and errors[i] is None for i,e in enumerate(envs)):
            for i,e in enumerate(envs):obs[i]=e.observation();steps[i]=e.step_count
            actions=np.asarray(sample(jnp.asarray(obs),fold(keys0,jnp.asarray(steps))))
            for i,e in enumerate(envs):
                if e.done or errors[i] is not None:continue
                before=e.positions.copy();flow=bounded_nominal(actions[i],cfg.max_speed);A,b,_=barrier_constraints(e.snapshot(),cbf)
                try:
                    safe,s1,r1,_=project_velocity_with_retry(flow,A,b,cfg.max_speed,cbf)
                    g=correctors[i](obs[i],safe,cfg.max_speed);w=safe+g;u,s2,r2,_=project_velocity_with_retry(w,A,b,cfg.max_speed,cbf)
                except Exception as exc:
                    errors[i]={'type':type(exc).__name__,'message':str(exc),'absolute_step':e.step_count};continue
                _,_,_,info=e.step(u);flat1=safe.ravel();flat2=u.ravel()
                vals=[before,e.positions.copy(),flow,safe,g,w,u,
                      np.r_[A@flat1-b<=1e-6,abs(np.linalg.norm(safe,axis=-1)-cfg.max_speed)<=1e-6],
                      np.r_[A@flat2-b<=1e-6,abs(np.linalg.norm(u,axis=-1)-cfg.max_speed)<=1e-6],
                      s1,s2,r1,r2,e.stuck_timer,info['termination']]
                for key,val in zip(hist[i],vals):hist[i][key].append(val)
        for i,task in enumerate(tasks):
            h={k:np.asarray(v) for k,v in hist[i].items()};n=len(h['event']);name=f'{chunk+i:05d}.npz'
            np.savez_compressed(raw/name,**h,eta=np.asarray(task['eta']),seed=task['seed'])
            records.append({**task,'outcome':str(h['event'][-1]) if errors[i] is None else None,
                'execution_error':errors[i],'steps':n,'terminal_step':envs[i].step_count,
                'file':f'raw/{args.stage}/{name}','sha256':sha(raw/name)})
            total_steps+=n
        (raw/'partial.json').write_text(json.dumps(records))
        if time.monotonic()-last>20 or len(records)==len(jobs):
            print(json.dumps({'completed':len(records),'total':len(jobs),'elapsed_s':round(time.monotonic()-started,1),
              'outcomes':dict((x,sum(r['outcome']==x for r in records)) for x in ['success','deadlock','timeout','collision',None])}),flush=True);last=time.monotonic()
    assert sha(HERE/'protocol.json')==locked
    result={'stage':args.stage,'records':records,'new_rollouts':len(records),'physical_steps':total_steps,
      'elapsed_s':time.monotonic()-started,'device':[str(x) for x in jax.devices()],
      'batch_size':args.batch,'protocol_sha256':locked,'jobs_sha256':sha(HERE/args.jobs)}
    (raw/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

if __name__=='__main__':main()

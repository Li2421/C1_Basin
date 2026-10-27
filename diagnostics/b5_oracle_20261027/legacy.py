"""Frozen four-scene checkpoint backtest, using the preserved TT/FF adapters.

No reference-package projector; these are the archived production solvers.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
from pathlib import Path
import sys,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src import rollout_db as db,planner,cache_writer
assert db.ROOT==ROOT/'shared_rollout_db'
from shared_control.correction_b5 import FrozenYield,YieldConfig,rotate,target
sys.path.insert(0,str(OLD))
from validation_v2.adapters import OtherRuntime,FourRingRuntime
import jax
import numpy as np

EXP='exp_b5_legacy_20261027'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v)for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,float)and not np.isfinite(x):return None
    return x
def dump(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

@lru_cache(None)
def runtime(sc):
    jax.config.update('jax_enable_x64',sc=='toy_give_way')
    return OtherRuntime(sc)if sc in ('toy_give_way','double_bottleneck')else FourRingRuntime(sc)


def freeze():
    files=['shared_control/hard_projection.py','shared_control/basis_families.py','single_integrator/environment.py',
        'single_integrator/cbf.py','ring_exchange/environment.py','four_way_intersection/environment.py','double_bottleneck/environment.py']
    for name in files:assert (OLD/'source'/name).read_bytes()==(ROOT/name).read_bytes(),name
    pp=json.load(open(OLD/'validation_v2/protocol.json'))
    unique={c['state']['uid']:c['state']for c in pp['cases']};selected=[]
    for sc in ('toy_give_way','double_bottleneck','ring_exchange','four_way_intersection'):
        selected.extend([s for s in unique.values()if s['scenario']==sc][:2])
    # Frozen from the initial pilot before any legacy rollout or independent test.
    methods={
        'safety':dict(mode='b3',eta=[0.]*5),
        'B3':dict(mode='b3',eta=[.875,0.,.375,0.,0.]),
        'T_only':dict(mode='add',eta=[.875,0.,.375,-.75,0.]),
        'ADD5':dict(mode='add',eta=[.875,0.,.375,-.75,1.],role='reverse_nearest_hold'),
        'BLEND5':dict(mode='blend',eta=[1.183902831049636,-.4170497842133045,.32363618491217494,-.8958365190774202,.21004703268408775],role='nearest_back2'),
        'RHR_A':dict(mode='rhr_goal',eta=[0.]*5),
        'RHR_B':dict(mode='rhr_centre',eta=[0.]*5),
        'FF_B3':dict(mode='ff_b3',eta=[.875,0.,.375,0.,0.]),
        'FF_ADD5':dict(mode='ff_add',eta=[.875,0.,.375,-.75,1.],role='reverse_nearest_hold')}
    # One physical constant lifting in FF. FF-BLEND is deliberately not relabeled
    # as the terminal-input BLEND formula because those execution semantics differ.
    tasks=[]
    for state in selected:
        rt=runtime(state['scenario']);cfg=rt.config.to_dict();scid=db.uid('scenario',dict(legacy_scene=state['scenario'],config=cfg))
        physical={k:state['physical'][k]for k in ('positions','velocities','goals')}
        sid=db.uid('state',dict(scenario=scid,physical=physical,legacy_anchor=state['uid']))
        for name,spec in methods.items():
            payload=dict(nominal_checkpoint_sha256=sha(rt.checkpoint),config=cfg,spec=spec,
                method=name,implementation_sha256=sha(__file__),new_basis_sha256=sha(ROOT/'shared_control/correction_b5.py'),
                frozen_source=sha(OLD/'source_snapshot.json'),adapter=sha(OLD/'validation_v2/adapters.py'),
                controller=sha(OLD/'poc/field_controller.py'),cbf=rt.cbf.to_dict(),
                rng='fold(2026100402,sha256(legacy_state_uid)[:8],future_index,physical_step)',
                precision='x64'if state['scenario']=='toy_give_way'else'x32')
            cid=db.uid('ctl',payload);eid=db.eta_identity(spec['eta'][:3])[0]
            for seed in (16,17):
                sk=db.canonical(dict(future_index=seed));rid=db.uid('roll',dict(state=sid,eta=eid,controller=cid,seed=sk))
                tasks.append(dict(state=state,physical=physical,scenario_uid=scid,state_uid=sid,controller_uid=cid,eta_uid=eid,
                    seed_key=sk,rollout_uid=rid,seed=seed,method=name,spec=spec,payload=payload,config=cfg,
                    checkpoint=str(rt.checkpoint),checkpoint_sha256=sha(rt.checkpoint)))
    protocol=dict(tasks=tasks,methods=methods,states=selected,source_production_files_identical=files,
        evidence='Original validation_v2 cases use seeds 0/1 and do not contain B5; seeds16/17 are new matched holdout streams.',
        frozen=True,no_training=True,local_progress='sum-distance reduction .1 m, proxy only')
    path=OUT/'LEGACY_FROZEN.json'
    if path.exists():assert json.load(path.open())==clean(protocol)
    else:dump(path,protocol)
    return tasks


def register(tasks):
    with db.connect()as con,db.transaction(con):
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,metadata_json) VALUES(?,?,?,?)',(EXP,EXP,str(OUT/'legacy_traces'),db.canonical(dict(legacy_checkpoint_backtest=True))))
        for c in tasks:
            con.execute('INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json)VALUES(?,?,?,?)',
                (c['scenario_uid'],c['state']['scenario'],c['scenario_uid'],db.canonical(c['config'])))
            con.execute('INSERT INTO state(state_uid,scenario_uid,content_hash,physical_state_json,provenance_json,identity_quality)VALUES(?,?,?,?,?,?) ON CONFLICT(state_uid) DO NOTHING',
                (c['state_uid'],c['scenario_uid'],db.uid('physical',c['physical']),db.canonical(c['physical']),db.canonical(dict(legacy_state=c['state']['uid'],experiment=EXP)),'CONTENT_EXACT'))
            assert con.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone()[0]==db.canonical(c['physical'])
            con.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,config_json,compatibility_quality)VALUES(?,?,?,?,?)',
                (c['controller_uid'],c['scenario_uid'],c['checkpoint_sha256'],db.canonical(c['payload']),'EXACT_PROFILE'))
            eid,e,hx=db.eta_identity(c['spec']['eta'][:3]);con.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex)VALUES(?,?,?,?,?)',(eid,*e,hx))
    path=OUT/'legacy_planned_rollouts.json';dump(path,dict(requests=[{k:c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key')}for c in tasks]))
    pf=planner.preflight(path);dump(OUT/'legacy_cache_preflight.json',pf)
    assert pf['summary']['ambiguous']==0
    return {(r['state_uid'],r['eta_uid'],r['controller_uid'],sk)for r in pf['details']for sk in r['missing_seeds']}


def init_worker():
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[cpus[(mp.current_process()._identity[0]-1)%len(cpus)]])


def work(c):
    sc=c['state']['scenario'];jax.config.update('jax_enable_x64',sc=='toy_give_way');rt=runtime(sc)
    env=rt.make_env(c['state']);spec=c['spec'];e=spec['eta'];cfg=rt.config
    p=env.positions;d=np.linalg.norm(p[:,None]-p[None,:],axis=-1);np.fill_diagonal(d,np.inf);i,j=np.unravel_index(np.argmin(d),d.shape)
    role=None
    if spec.get('role'):
        rc=YieldConfig(int(j),int(i))if spec['role']=='reverse_nearest_hold'else YieldConfig(int(i),int(j),'back',2.)
        role=FrozenYield(rc,p,env.goals,2*cfg.agent_radius+cfg.agent_collision_margin+rt.cbf.separation_buffer)
    token=int(hashlib.sha256(c['state']['uid'].encode()).hexdigest()[:8],16)
    root=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026100402),token),c['seed'])
    pos=[p.copy()];actions=[];safe_actions=[];certs=[];keys=[];history={};term='timeout';error=None;zero_checks=0;t0=time.monotonic()
    for t in range(cfg.max_steps):
        try:
            key=jax.random.fold_in(root,t)
            obs,safety,flow,safe,delta,_=rt.prepare(env,key,np.asarray(e[:3]))
            fields=rt.basis.compute(env.positions,env.goals,safe,cfg.max_speed)
            b3=safe+delta
            # Preserve exact old G/P/R calculation. No contact gate is backported.
            test=target(safe,np.stack(fields.values,axis=1),rotate(fields.values[0]),role,env.positions,cfg.max_speed,e[:3]+[0.,0.])
            assert np.array_equal(test,b3);zero_checks+=1
            if spec['mode'].startswith('rhr'):
                act,_=safety.project(b3)
                dist=np.linalg.norm(env.goals-env.positions,axis=1);history[t]=dist.copy();history.pop(t-41,None)
                active=(history[t-40]-dist<.05)&(dist>cfg.goal_tolerance)if t>=40 else np.zeros(len(p),bool)
                if active.any():
                    q=env.goals-env.positions if spec['mode']=='rhr_goal'else env.positions
                    out=act.copy()
                    for ii in np.flatnonzero(active):out[ii]+=.35*np.array([q[ii,1],-q[ii,0]])/max(np.linalg.norm(q[ii]),1e-30)
                    act,_=safety.project(out)
            elif spec['mode'].startswith('ff'):
                dd=delta.copy()
                if spec['mode']=='ff_add':
                    dd+=e[3]*rotate(fields.values[0])
                    if role is not None:ii=role.config.yielder;dd[ii]+=e[4]*(role.clear(env.positions,cfg.max_speed)[ii]-safe[ii])
                act,_=rt.integrate(env,obs,key,dd,safety)
            elif spec['mode']=='b3':act,_=safety.project(b3)
            else:
                out=target(safe,np.stack(fields.values,axis=1),rotate(fields.values[0]),role,env.positions,cfg.max_speed,e,spec['mode'])
                act,_=safety.project(out)
            cert=safety.assert_safe(act);_,_,done,info=env.step(act)
            actions.append(act);safe_actions.append(safe);pos.append(env.positions.copy());keys.append(np.asarray(key));certs.append(cert)
            term=info['termination']
            if done:break
        except Exception as exc:
            import traceback
            error=dict(step=t,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());term='numerical_failure';break
    summary=env.summary();collision=any(summary.get(k,False)for k in ('wall_collision','agent_collision','obstacle_collision','outer_collision'))
    success=bool(term=='success' and summary['collision_free_success']and not error)
    distances=np.linalg.norm(env.goals[None]-np.asarray(pos),axis=-1).sum(-1);reached=np.flatnonzero(distances<=distances[0]-.1)
    path=OUT/'legacy_traces'/f"{c['rollout_uid']}.npz";path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,positions=pos,actions=actions,safe=safe_actions,goals=env.goals,keys=keys)
    row=dict(**{k:c[k]for k in ('scenario_uid','state_uid','controller_uid','eta_uid','seed_key','rollout_uid','seed','method','spec')},
        scenario=sc,eta=e[:3],future_index=c['seed'],success=int(success),outcome=term,timeout=int(term=='timeout'),
        collision=int(collision),deadlock=int(term=='deadlock'),numerical_failure=int(error is not None),error=error,
        completion_time=len(actions)*cfg.dt if success else None,truncated_time=len(actions)*cfg.dt if success else cfg.max_steps*cfg.dt,
        episode_length=len(actions),elapsed_seconds=time.monotonic()-t0,source='legacy_new_rollout',trace=str(path),
        local_progress_time=float(reached[0]*cfg.dt)if len(reached)else None,final_goal_distances=np.linalg.norm(env.goals-env.positions,axis=-1),
        bitwise_zero_extension_checks=zero_checks,max_action_violation=max((d['max_violation']for d in certs),default=None),
        flow_checkpoint_sha256=c['checkpoint_sha256'],cache_identity_schema='registered_exact_uid_v1',
        initial_positions=c['physical']['positions'],initial_velocities=c['physical']['velocities'],goals=c['physical']['goals'],
        legacy_state_uid=c['state']['uid'])
    row=clean(row);cache_writer.append_journal([row],EXP);return row


def main():
    # Resume the recorded execution identities. Subsequent registration-only
    # fixes must not invent new rollout identities for completed trajectories.
    frozen_path=OUT/'LEGACY_FROZEN.json'
    tasks=json.load(frozen_path.open())['tasks']if frozen_path.exists()else freeze()
    missing=register(tasks);p=OUT/'legacy_candidates.jsonl'
    done={x['rollout_uid']for x in map(json.loads,p.open())}if p.exists()else set()
    todo=[c for c in tasks if c['rollout_uid']not in done]
    assert all(tuple(c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key'))in missing for c in todo)
    print('legacy new',len(todo),flush=True)
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=init_worker)as pool,p.open('a')as f:
        for n,fu in enumerate(as_completed([pool.submit(work,c)for c in todo]),1):
            r=fu.result();f.write(json.dumps(r)+'\n');f.flush();print(n,r['scenario'],r['method'],r['outcome'],round(r['elapsed_seconds'],2),flush=True)

if __name__=='__main__':main()

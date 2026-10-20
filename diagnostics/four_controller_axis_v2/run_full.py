"""Fixed Four FF rollouts for Round19b; schedule support added after headroom gate."""
from __future__ import annotations
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from adapter import FourFlowAdapter
from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src import cache_writer
cache_writer.ROOT=HERE
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def initial(env):
    if hasattr(env,'augmented_state'):return env.augmented_state()
    keys=['distance_history','candidate_since','stuck_timer','max_stuck_timer','ever_candidate_deadlock','first_success_step','first_deadlock_step','first_wall_collision_step','first_agent_collision_step','done']
    return dict(env.snapshot(),**{k:getattr(env,k) for k in keys if hasattr(env,k)})
def execute(phase,index,workers):
    assert phase=='fixed'
    receipt=read(HERE/'R5_FREEZE.receipt.json');assert receipt['features_sha256']==sha(HERE/'R5_FROZEN.npz')
    ppath=HERE/'FIXED_PROTOCOL.json';p=read(ppath);assert p['runner_sha256']==sha(__file__) and p['adapter_sha256']==sha(HERE/'adapter.py')
    assert read(HERE/'FIXED_CACHE_PREFLIGHT.json')['summary']['ambiguous']==0
    names=p['controller_order'];rows=read(HERE/'FIXED_MISSING_CASES.json');assigned=[c for c in rows if names.index(c['base_controller_name'])%workers==index]
    states={s['state_uid']:s for s in p['states']};runtimes={};buffer=[];new=0;elapsed=0.
    out=HERE/'stage2';write(out/f'START_AUTHORIZATION_{index:02d}.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        worker=index,protocol_sha256=sha(ppath),r5_receipt_sha256=sha(HERE/'R5_FREEZE.receipt.json'),assigned=len(assigned)))
    for c in assigned:
        dest=out/'raw'/f"{c['rollout_uid']}.json"
        if dest.exists():buffer.append(read(dest));continue
        name=c['base_controller_name'];ctl=p['profiles'][name];v=ctl['payload'];assert sha(v['flow_checkpoint_path'])==v['flow_checkpoint_sha256']
        if name not in runtimes:runtimes[name]=FlowEndpointMatchedField(FourFlowAdapter(v['flow_checkpoint_path']),'isotropic')
        rt=runtimes[name];s=states[c['state_uid']];env=rt.make_env(s);aug=clean(initial(env));augbytes=canonical(aug).encode();augsha=hashlib.sha256(augbytes).hexdigest()
        initialpath=out/'initial_states'/f'{augsha}_{index:02d}.json'
        if not initialpath.exists():write(initialpath,aug)
        positions=[env.positions.copy()];velocities=[env.velocities.copy()];actions=[];clearance=[]
        error=None;term='timeout';steps=0;calls=0;first=None;t0=time.monotonic()
        for t in range(rt.config.max_steps):
            try:
                eta=c['eta_a'] if t<c['switch_step'] else c['eta_b']
                a,_,n=rt.action(env,key_for(s,c['seed'],t),np.asarray(eta),'field');steps+=1;calls+=n
                if first is None:first=np.asarray(a).tolist()
                _,_,done,info=env.step(a);term=info['termination']
                positions.append(env.positions.copy());velocities.append(env.velocities.copy());actions.append(np.asarray(a).copy())
                cfg=rt.config
                wall=float(info.get('min_swept_wall_distance',np.inf))-getattr(cfg,'wall_collision_margin',getattr(cfg,'collision_margin',0.))
                pair=float(info.get('min_swept_agent_distance',np.min(info.get('swept_agent_distances',[np.inf]))))-cfg.agent_collision_margin
                clearance.append(min(wall,pair))
                if done:break
            except Exception as exc:
                error=dict(step=t,type=type(exc).__name__,message=str(exc)[:500]);term='numerical_failure';break
        summary=env.summary();collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
        success=int(bool(summary['collision_free_success'] and term=='success' and error is None));physical=len(actions)
        metadata=dict(batch_id=p['experiment_uid'],scenario='four_way_intersection',rollout_id=c['rollout_uid'],
            controller=c['controller_uid'],checkpoint=dict(path=v['flow_checkpoint_path'],sha256=v['flow_checkpoint_sha256']),
            seed=c['seed'],source_record=str(dest),initial_state_sha256=augsha,augmented_initial_state_path=str(initialpath),
            augmented_initial_state_file_sha256=sha(initialpath),config=rt.config.to_dict(),start_step=0,
            episode_steps=physical,complete=bool(error is None and env.done and physical==env.step_count),
            termination=term,collision=bool(collision),safety_enabled=True,deadlock_detected=False,
            deadlock_criterion=None,arm=c['arm'],eta_a=c['eta_a'],eta_b=c['eta_b'],switch_step=c['switch_step'])
        trace=out/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace,positions=np.asarray(positions),velocities=np.asarray(velocities),actions=np.asarray(actions),
            steps=np.arange(physical+1),swept_clearance=np.asarray(clearance),goals=env.goals,
            walls=np.asarray(getattr(env,'walls',np.empty((0,2,2)))),metadata_json=np.array(json.dumps(clean(metadata))))
        r=dict(schema='round19b_four_fixed_seed_v1',experiment_uid=p['experiment_uid'],protocol_sha256=sha(ppath),case=c,
            **{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')},
            success=success,deadlock=0,timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(error is not None),
            outcome=term,episode_length=steps,physical_steps=physical,elapsed_seconds=time.monotonic()-t0,
            first_action=first,projection_calls=calls,error=error,summary=summary,jax_enable_x64=False,
            execution_code_sha256=sha(__file__),trace_path=str(trace),trace_sha256=sha(trace),
            checkpoint_sha256=v['flow_checkpoint_sha256'],minimum_swept_clearance=min(clearance) if clearance else None)
        write(dest,clean(r));buffer.append(clean(r));new+=1;elapsed+=r['elapsed_seconds']
        if len(buffer)>=8:append_journal(buffer,p['experiment_uid'],f'fixed_{index}');buffer=[]
        if new%16==0:print(json.dumps(dict(worker=index,new=new,assigned=len(assigned),mean_seconds=elapsed/new)),flush=True)
    if buffer:append_journal(buffer,p['experiment_uid'],f'fixed_{index}')
    write(out/f'worker_{index:02d}.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker=index,assigned=len(assigned),new=new,complete=True,elapsed_seconds=elapsed))
    print(json.dumps(dict(worker=index,new=new,assigned=len(assigned),complete=True)),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['fixed']);ap.add_argument('index',type=int);ap.add_argument('workers',type=int);a=ap.parse_args()
    execute(a.phase,a.index,a.workers)

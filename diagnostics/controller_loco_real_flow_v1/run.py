"""Cache-first real Flow rollout execution; imports Round 3 exact runtime."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('JAX_PLATFORMS','cpu')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
EXP='exp_controller_loco_real_flow_v1_stage_a'

import numpy as np
from diagnostics.controller_q_diversity_v1 import experiment as old
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import connect

def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def execute(rows,tag):
    import jax
    from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(HERE/'protocol.json')
    assert old.sha(HERE/'protocol.json')==read(HERE/'PILOT_OR_FULL_AUTHORITY.json')['protocol_sha256']
    assert old.sha(ROOT/'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py')==p['flow_adapter_sha256']
    assert read(HERE/'CACHE_INVENTORY.json')['missing']<=90000
    states={s['state_uid']:s for s in p['states']};runtimes={};buffer=[];times=[];new=0
    for r in rows:
        dest=HERE/'raw'/(r['rollout_uid']+'.json')
        if dest.exists():
            previous=read(dest);assert previous['case']==r and previous['protocol_sha256']==old.sha(HERE/'protocol.json')
            buffer.append(previous)
            if len(buffer)>=16:append_journal(buffer,EXP,tag);buffer=[]
            continue
        name=r['controller_name'];profile=p['controllers']['ring_exchange'][name]['payload']
        jax.config.update('jax_enable_x64',bool(profile['jax_enable_x64']))
        assert not jax.config.x64_enabled
        if name not in runtimes:
            if name=='base_u_v10_local_macflow':
                # P1 reused all 1,536 native v10 labels, so its profile has no adapter manifest.
                # The historical exact execution is the native FourRingRuntime.
                runtimes[name]=native_runtime('ring_exchange')
            else:
                manifest=read(profile['adapter_manifest_path'])
                runtimes[name]=RingFlowAdapter(profile['flow_checkpoint_path'],manifest,profile['observation_frame'])
        rt=runtimes[name];s=states[r['state_uid']];env=rt.make_env(s)
        term='timeout';error=None;steps=0;calls=0;first=None;t0=time.monotonic()
        for step in range(rt.config.max_steps):
            try:
                action,_,n=rt.action(env,key_for(s,r['seed'],step),np.asarray(r['eta']),'field')
                calls+=n;steps+=1
                if first is None:first=np.asarray(action).tolist()
                _,_,done,info=env.step(action);term=info['termination']
                if done:break
            except Exception as exc:
                error=dict(step=step,type=type(exc).__name__,message=str(exc)[:500]);term='numerical_failure';break
        summary=env.summary();collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
        record=dict(schema='controller_loco_real_flow_stage_a_seed_v1',experiment_uid=EXP,protocol_sha256=old.sha(HERE/'protocol.json'),case=r,
                    **{k:r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')},
                    success=int(summary['collision_free_success'] and term=='success' and error is None),
                    deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(error is not None),
                    episode_length=steps,outcome=term,elapsed_seconds=time.monotonic()-t0,first_action=first,projection_calls=calls,error=error,summary=summary,
                    execution_semantics_version='scene_precision_guard_v1',jax_enable_x64=bool(jax.config.x64_enabled),execution_code_sha256=old.sha(__file__),
                    flow_adapter_sha256=p['flow_adapter_sha256'],checkpoint_sha256=profile['flow_checkpoint_sha256'])
        assert not(record['success'] and (record['collision'] or record['numerical_failure']))
        write(dest,record);buffer.append(record);new+=1;times.append(record['elapsed_seconds'])
        if len(buffer)>=16:append_journal(buffer,EXP,tag);buffer=[]
        if new%64==0:print(json.dumps(dict(tag=tag,new=new,last_seconds=times[-1])),flush=True)
    if buffer:append_journal(buffer,EXP,tag)
    return dict(tag=tag,assigned=len(rows),new=new,episode_cpu_seconds=sum(times),mean_seconds=float(np.mean(times)) if times else None,
                complete=all((HERE/'raw'/(r['rollout_uid']+'.json')).exists() for r in rows))

def pilot():
    rows=read(HERE/'MISSING_CASES.json')
    indices=np.linspace(0,len(rows)-1,98,dtype=int)
    selected=[rows[int(i)] for i in indices]
    write(HERE/'PILOT_CASE_UIDS.json',[r['rollout_uid'] for r in selected])
    rec=execute(selected,'pilot')
    write(HERE/'PILOT_TIMING.json',rec)
    print(json.dumps(rec),flush=True)

def worker(index,workers):
    rows=read(HERE/'MISSING_CASES.json')
    # Round-robin frozen UIDs across heterogeneous Flow runtimes. The first
    # contiguous scheduling attempt was cancelled because controller-dependent
    # episode time created a 2.2h–8.1h worker tail. Existing raw is exact-UID
    # skipped, never rerun; only scheduling changes.
    selected=rows[index::workers]
    rec=execute(selected,f'worker_{index}')
    write(HERE/f'WORKER_{index}.json',rec)
    print(json.dumps(rec),flush=True)

def merge():
    old.HERE=HERE;old.EXPERIMENT=EXP;old.merge()

def postflight():
    cases=read(HERE/'CASES.json');valid=unknown=missing=quarantine=0
    with connect(True) as db:
        for r in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if row is None:missing+=1
            elif row['conflict_quarantined']:quarantine+=1
            elif row['numerical_failure']:unknown+=1
            else:valid+=1
    rec=dict(total=len(cases),valid=valid,unknown=unknown,missing=missing,quarantined=quarantine,
             raw=len(list((HERE/'raw').glob('*.json'))),new_expected=len(read(HERE/'MISSING_CASES.json')),
             workers_complete=all(read(HERE/f'WORKER_{i}.json')['complete'] for i in range(19)))
    write(HERE/'POSTFLIGHT.json',rec)
    assert missing==quarantine==0 and rec['raw']==rec['new_expected'] and rec['workers_complete'],rec
    print(json.dumps(rec))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','worker','merge','postflight']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=19)
    a=ap.parse_args()
    if a.stage=='worker':worker(a.index,a.workers)
    else:globals()[a.stage]()

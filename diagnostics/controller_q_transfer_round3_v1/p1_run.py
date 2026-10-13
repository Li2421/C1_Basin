"""P1 cache-first alternate real-Flow rollout collection; single merger."""
import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('JAX_PLATFORMS','cpu')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
EXP='exp_controller_q_transfer_round3_p1_v1'

import numpy as np
from diagnostics.controller_q_diversity_v1 import experiment as old
from shared_rollout_db.src.rollout_db import connect,canonical
from shared_rollout_db.src.cache_writer import append_journal


def read(name):return json.loads((HERE/name).read_text())
def write(name,content):
    path=HERE/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(content,indent=2,allow_nan=False)+'\n')
def freeze(name,content):
    path=HERE/name
    if path.exists():assert json.loads(path.read_text())==content,name
    else:write(name,content)


def prepare():
    assert read('P1_STATIC_AUDIT.json')['status']=='PASS'
    shutil.copyfile(HERE/'P1_EXECUTION_PROTOCOL.json',HERE/'protocol.json') if not (HERE/'protocol.json').exists() else None
    assert old.sha(HERE/'protocol.json')==old.sha(HERE/'P1_EXECUTION_PROTOCOL.json')
    p=read('protocol.json');cases=[old.case(p,s,name,j,k) for name in p['controller_order'] for s in p['states'] for j in p['eta_indices'] for k in p['seeds']]
    with connect() as db:
        db.execute('UPDATE experiment SET protocol_hash=? WHERE experiment_uid=?',(old.sha(HERE/'protocol.json'),EXP))
        db.commit()
    cases.sort(key=lambda r:(r['controller_name'],r['state_index'],r['eta_index'],r['seed']))
    groups=defaultdict(list);missing=[];cached=[];unknown=[];quarantined=[]
    with connect(True) as db:
        for r in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if row is None:missing.append(r)
            else:
                assert all(row[key]==r[key] for key in ('state_uid','eta_uid','controller_uid','seed_key'))
                if row['conflict_quarantined']:quarantined.append(r['rollout_uid'])
                elif row['numerical_failure']:unknown.append(r['rollout_uid'])
                else:cached.append(r['rollout_uid'])
            groups[(r['state_uid'],r['eta_uid'],r['controller_uid'])].append(r['seed_key'])
    assert not quarantined, 'Quarantined cases require explicit resolution; no rollout'
    assert len(missing)<=25000
    freeze('P1_CASES.json',cases)
    freeze('P1_PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=v) for (s,e,c),v in groups.items()]))
    freeze('P1_MISSING_CASES.json',missing)
    write('P1_CACHED_UIDS.json',dict(valid=cached,numerical_unknown=unknown,quarantined=quarantined))
    cost=dict(total=len(cases),genuinely_missing=len(missing),cached_valid=len(cached),cached_unknown=len(unknown),
              historic_mean_seconds=2.118,predicted_episode_cpu_hours=len(missing)*2.118/3600,
              predicted_wall_minutes_at24=len(missing)*2.118/(24*60),max_new_rollouts=25000,max_workers=24)
    write('P1_COST.json',cost)
    print(json.dumps(cost))


def worker(index,workers):
    import jax
    from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read('protocol.json');assert old.sha(HERE/'protocol.json')==old.sha(HERE/'P1_EXECUTION_PROTOCOL.json')
    assert old.sha(HERE/'flow_adapter.py')==p['adapter_sha256']
    assert read('P1_STATIC_AUDIT.json')['status']=='PASS'
    standard=read('P1_CACHE_PREFLIGHT.json')['summary']
    local=read('P1_COST.json')
    assert standard['ambiguous']==0 and standard['genuinely_missing']==local['genuinely_missing']+local['cached_unknown']
    rows=read('P1_MISSING_CASES.json')
    start=len(rows)*index//workers;end=len(rows)*(index+1)//workers
    assigned=rows[start:end]
    states={s['state_uid']:s for s in p['states']};runtimes={};buffer=[];new=0;elapsed=[]
    for r in assigned:
        dest=HERE/'p1_raw'/(r['rollout_uid']+'.json')
        if dest.exists():
            oldrow=json.loads(dest.read_text());assert oldrow['case']==r
            buffer.append(oldrow)
            if len(buffer)>=8:append_journal(buffer,EXP,f'p1_{index}');buffer=[]
            continue
        name=r['controller_name'];profile=p['controllers']['ring_exchange'][name]['payload'];jax.config.update('jax_enable_x64',bool(profile['jax_enable_x64']))
        assert not jax.config.x64_enabled
        if name not in runtimes:
            manifest=json.loads(Path(profile['adapter_manifest_path']).read_text())
            runtimes[name]=RingFlowAdapter(profile['flow_checkpoint_path'],manifest,profile['observation_frame'])
        rt=runtimes[name];s=states[r['state_uid']];env=rt.make_env(s);term='timeout';error=None;steps=0;calls=0;first=None;t0=time.monotonic()
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
        record=dict(schema='controller_q_round3_p1_seed_v1',experiment_uid=EXP,protocol_sha256=old.sha(HERE/'protocol.json'),case=r,
                    **{k:r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')},
                    success=int(summary['collision_free_success'] and term=='success' and error is None),deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(error is not None),
                    episode_length=steps,outcome=term,elapsed_seconds=time.monotonic()-t0,first_action=first,projection_calls=calls,error=error,summary=summary,
                    execution_semantics_version='scene_precision_guard_v1',jax_enable_x64=bool(jax.config.x64_enabled),execution_code_sha256=old.sha(__file__),
                    flow_adapter_sha256=old.sha(HERE/'flow_adapter.py'),checkpoint_sha256=profile['flow_checkpoint_sha256'])
        assert not(record['success'] and (record['collision'] or record['numerical_failure']))
        old.write(dest,record);buffer.append(record);new+=1;elapsed.append(record['elapsed_seconds'])
        if len(buffer)>=8:append_journal(buffer,EXP,f'p1_{index}');buffer=[]
        if new%32==0:print(json.dumps(dict(index=index,new=new,last_seconds=elapsed[-1])),flush=True)
    if buffer:append_journal(buffer,EXP,f'p1_{index}')
    write(f'P1_WORKER_{index}.json',dict(index=index,assigned=len(assigned),new=new,
                                       completed=all((HERE/'p1_raw'/(r['rollout_uid']+'.json')).exists() for r in assigned),
                                       episode_cpu_seconds=sum(elapsed)))


def merge():
    old.HERE=HERE;old.EXPERIMENT=EXP
    old.merge()


def postflight():
    p=read('protocol.json');cases=read('P1_CASES.json');valid=unknown=missing=quarantine=0
    with connect(True) as db:
        for r in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if row is None:missing+=1;continue
            if row['conflict_quarantined']:quarantine+=1
            elif row['numerical_failure']:unknown+=1
            else:valid+=1
    result=dict(total=len(cases),valid=valid,unknown=unknown,missing=missing,quarantined=quarantine,
                new_expected=len(read('P1_MISSING_CASES.json')),new_raw=len(list((HERE/'p1_raw').glob('*.json'))),
                all_workers_completed=all(read(f'P1_WORKER_{i}.json')['completed'] for i in range(24)))
    assert missing==quarantine==0 and result['all_workers_completed'] and result['new_raw']==result['new_expected'],result
    write('P1_POSTFLIGHT.json',result);print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','worker','merge','postflight']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=24)
    a=ap.parse_args()
    globals()[a.stage](a.index,a.workers) if a.stage=='worker' else globals()[a.stage]()

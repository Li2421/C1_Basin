"""Cache-first, frozen P1-compatible Ring FF Q16 confirmation and single merger."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ['JAX_PLATFORMS']='cpu'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
P1=ROOT/'diagnostics/controller_q_transfer_round3_v1'
from diagnostics.controller_q_diversity_v1.experiment import case
from shared_rollout_db.src.rollout_db import connect,transaction,uid
from shared_rollout_db.src.cache_writer import append_journal

def read(name):return json.loads((HERE/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)

def guard():
    p=read('protocol.json');cases=read('cases.json');f=read('cache_preflight.json')
    assert p['max_new_continuations']==320 and len(cases)==320
    assert p['source_p1_protocol_sha256']==sha(P1/'P1_PROTOCOL.json')
    assert p['adapter_sha256']==sha(P1/'flow_adapter.py')
    assert f['summary']['total_requested']==320 and f['summary']['ambiguous']==0
    assert f['summary']['genuinely_missing']<=320
    for name in p['controller_order']:
        profile=p['controllers']['ring_exchange'][name]['payload']
        assert sha(profile['flow_checkpoint_path'])==profile['flow_checkpoint_sha256']
        assert sha(profile['adapter_manifest_path'])==profile['adapter_manifest_sha256']
    return p,cases,f

def missing_cases(cases,preflight):
    missing={(d['state_uid'],d['eta_uid'],d['controller_uid'],s)
             for d in preflight['details'] for s in d['missing_seeds']}
    selected=[c for c in cases if (c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key']) in missing]
    assert len(selected)==preflight['summary']['genuinely_missing']
    return selected

def worker(index,workers):
    import jax
    import numpy as np
    from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p,cases,pre=guard();all_missing=missing_cases(cases,pre)
    assigned=all_missing[len(all_missing)*index//workers:len(all_missing)*(index+1)//workers]
    states={s['state_uid']:s for s in p['states']}
    source=json.loads((P1/'P1_PROTOCOL.json').read_text())
    runtimes={};buffer=[];new=0;reuse_raw=0;cpu_seconds=0.
    for c in assigned:
        s=states[c['state_uid']];name=c['controller_name']
        assert c==case(source|{'states':p['states']},s,name,c['eta_index'],c['seed'])
        path=HERE/'raw'/(c['rollout_uid']+'.json')
        if path.exists():
            rec=json.loads(path.read_text())
            assert rec['case']==c and rec['protocol_sha256']==sha(HERE/'protocol.json')
            reuse_raw+=1;buffer.append(rec)
            if len(buffer)>=8:append_journal(buffer,p['experiment_uid'],f'part3r2_{index}');buffer=[]
            continue
        profile=p['controllers']['ring_exchange'][name]['payload']
        jax.config.update('jax_enable_x64',False)
        assert not jax.config.x64_enabled
        if name not in runtimes:
            manifest=json.loads(Path(profile['adapter_manifest_path']).read_text())
            runtimes[name]=RingFlowAdapter(profile['flow_checkpoint_path'],manifest,profile['observation_frame'])
        rt=runtimes[name];env=rt.make_env(s)
        term='timeout';error=None;steps=0;calls=0;first=None;t0=time.monotonic()
        for step in range(rt.config.max_steps):
            try:
                act,_,n=rt.action(env,key_for(s,c['seed'],step),np.asarray(c['eta']),'field')
                calls+=n;steps+=1
                if first is None:first=np.asarray(act).tolist()
                _,_,done,info=env.step(act);term=info['termination']
                if done:break
            except Exception as exc:
                error=dict(step=step,type=type(exc).__name__,message=str(exc)[:500]);term='numerical_failure';break
        summary=env.summary()
        collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
        elapsed=time.monotonic()-t0
        rec=dict(schema='part3_round2_real_flow_q16_seed_v1',experiment_uid=p['experiment_uid'],
            protocol_sha256=sha(HERE/'protocol.json'),case=c,
            **{k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')},
            success=int(summary['collision_free_success'] and term=='success' and error is None),
            deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),
            numerical_failure=int(error is not None),episode_length=steps,outcome=term,
            elapsed_seconds=elapsed,first_action=first,projection_calls=calls,error=error,summary=summary,
            execution_semantics_version='scene_precision_guard_v1',jax_enable_x64=False,
            execution_code_sha256=sha(__file__),flow_adapter_sha256=sha(P1/'flow_adapter.py'),
            checkpoint_sha256=profile['flow_checkpoint_sha256'])
        assert not(rec['success'] and (rec['collision'] or rec['numerical_failure']))
        save(path,rec);buffer.append(rec);new+=1;cpu_seconds+=elapsed
        if len(buffer)>=8:append_journal(buffer,p['experiment_uid'],f'part3r2_{index}');buffer=[]
        if new%8==0:print(json.dumps(dict(worker=index,new=new,last_seconds=elapsed)),flush=True)
    if buffer:append_journal(buffer,p['experiment_uid'],f'part3r2_{index}')
    save(HERE/f'worker_{index}.json',dict(index=index,assigned=len(assigned),new=new,reused_raw=reuse_raw,
        cpu_seconds=cpu_seconds,complete=all((HERE/'raw'/(c['rollout_uid']+'.json')).exists() for c in assigned)))

def merge():
    p,cases,pre=guard();expected=sha(HERE/'protocol.json');source=json.loads((P1/'P1_PROTOCOL.json').read_text())
    by_uid={c['rollout_uid']:c for c in cases};states={s['state_uid']:s for s in p['states']}
    journal_dir=ROOT/'shared_rollout_db/journals'/p['experiment_uid']
    stats=dict(journals=0,inserted=0,duplicates=0)
    with connect() as db:
        for path in sorted(journal_dir.glob('*.jsonl')):
            records=[json.loads(line) for line in path.read_text().splitlines()]
            source_uid=uid('source',dict(path=str(path),sha256=sha(path)))
            with transaction(db):
                db.execute('INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                    (source_uid,p['experiment_uid'],str(path),sha(path),'.jsonl','SEED_EXACT',len(records)))
                added=duplicates=0
                for line,r in enumerate(records,1):
                    c=r['case'];s=states[c['state_uid']]
                    assert c==by_uid[c['rollout_uid']]
                    assert c==case(source|{'states':p['states']},s,c['controller_name'],c['eta_index'],c['seed'])
                    assert r['protocol_sha256']==expected and r['experiment_uid']==p['experiment_uid']
                    assert r['checkpoint_sha256']==p['controllers']['ring_exchange'][c['controller_name']]['payload']['flow_checkpoint_sha256']
                    assert r['execution_semantics_version']=='scene_precision_guard_v1' and not r['jax_enable_x64']
                    core=tuple(r[k] for k in ('success','deadlock','timeout','collision','numerical_failure'))
                    old=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                    if old:
                        assert core==tuple(old[k] for k in ('success','deadlock','timeout','collision','numerical_failure'))
                        duplicates+=1
                    else:
                        db.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*core,
                             r['episode_length'],r['outcome'],p['experiment_uid'],str(path),'EXACT_REUSE',uid('record',r)))
                        added+=1
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source_uid,line))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?',(added,duplicates,source_uid))
                stats['journals']+=1;stats['inserted']+=added;stats['duplicates']+=duplicates
            done=path.with_suffix('.merged');path.rename(done)
            db.execute('UPDATE source_file SET path=? WHERE source_uid=?',(str(done),source_uid));db.commit()
    save(HERE/'merge_result.json',stats);print(json.dumps(stats))

def postflight():
    p,cases,_=guard();counts=dict(total=len(cases),valid=0,unknown=0,missing=0,quarantined=0)
    with connect(True) as db:
        for c in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if row is None:counts['missing']+=1
            elif row['conflict_quarantined']:counts['quarantined']+=1
            elif row['numerical_failure']:counts['unknown']+=1
            else:counts['valid']+=1
    counts['workers']=[read(f'worker_{i}.json') for i in range(4)]
    assert counts['missing']==counts['quarantined']==0 and all(w['complete'] for w in counts['workers'])
    assert sum(w['new'] for w in counts['workers'])<=320
    save(HERE/'postflight.json',counts);print(json.dumps({k:v for k,v in counts.items() if k!='workers'}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['worker','merge','postflight'])
    ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    if args.stage=='worker':worker(args.index,args.workers)
    elif args.stage=='merge':merge()
    else:postflight()

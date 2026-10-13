"""Conditional P2′ rollout worker; must pass P1 gate, preflight, and budget."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('JAX_PLATFORMS','cpu')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BASE=Path(__file__).resolve().parent
HERE=BASE/'p2prime'
EXP='exp_controller_q_transfer_round3_p2prime_v1'

import numpy as np
from diagnostics.controller_q_diversity_v1 import experiment as old
from diagnostics.controller_q_diversity_v1.runtime import native_runtime
from diagnostics.controller_q_transfer_round3_v1.placement_runtime import make_program
from shared_rollout_db.src.rollout_db import connect
from shared_rollout_db.src.cache_writer import append_journal


def read(path):return json.loads(Path(path).read_text())
def write(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def setup():
    assert not read(BASE/'P1_RESULTS.json')['gate']['p2_qualified']
    assert read(BASE/'P2PRIME_COST.json')['new']<=70000
    HERE.mkdir(exist_ok=True)
    cp=HERE/'protocol.json'
    if not cp.exists():shutil.copyfile(BASE/'P2PRIME_PROTOCOL.json',cp)
    assert old.sha(cp)==old.sha(BASE/'P2PRIME_PROTOCOL.json')
    print(dict(status='FROZEN',protocol_sha=old.sha(cp)))


def worker(index,workers):
    import jax
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    p=read(HERE/'protocol.json')
    assert old.sha(HERE/'protocol.json')==old.sha(BASE/'P2PRIME_PROTOCOL.json')
    assert old.sha(BASE/'placement_runtime.py')==p['placement_runtime_sha256']
    assert read(BASE/'P2PRIME_STATIC_AUDIT.json')['status']=='PASS'
    assert read(BASE/'P2PRIME_COST.json')['new']<=70000
    std=read(BASE/'P2PRIME_CACHE_PREFLIGHT.json')['summary']
    cost=read(BASE/'P2PRIME_COST.json')
    assert std['ambiguous']==0 and std['genuinely_missing']==cost['new']+cost['cached_unknown']
    rows=read(BASE/'P2PRIME_MISSING_CASES.json')
    assigned=rows[len(rows)*index//workers:len(rows)*(index+1)//workers]
    states={s['state_uid']:s for s in p['states']};rt=None;programs={};buffer=[];new=0;elapsed=[]
    for r in assigned:
        dest=HERE/'raw'/(r['rollout_uid']+'.json')
        if dest.exists():
            previous=read(dest);assert previous['case']==r
            buffer.append(previous)
            if len(buffer)>=8:append_journal(buffer,EXP,f'p2prime_{index}');buffer=[]
            continue
        if rt is None:
            jax.config.update('jax_enable_x64',False)
            rt=native_runtime('ring_exchange')
        name=r['controller_name']
        if name not in programs:programs[name]=make_program(rt,name)
        profile=p['controllers']['ring_exchange'][name]['payload']
        jax.config.update('jax_enable_x64',bool(profile['jax_enable_x64']))
        assert not jax.config.x64_enabled
        program=programs[name];s=states[r['state_uid']];env=program.make_env(s)
        term='timeout';error=None;steps=0;calls=0;first=None;t0=time.monotonic()
        for step in range(program.config.max_steps):
            try:
                a,_,n=program.action(env,key_for(s,r['seed'],step),np.asarray(r['eta']),'field')
                calls+=n;steps+=1
                if first is None:first=np.asarray(a).tolist()
                _,_,done,info=env.step(a);term=info['termination']
                if done:break
            except Exception as exc:
                error=dict(step=step,type=type(exc).__name__,message=str(exc)[:500]);term='numerical_failure';break
        summary=env.summary();collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
        rec=dict(schema='controller_q_round3_p2prime_seed_v1',experiment_uid=EXP,protocol_sha256=old.sha(HERE/'protocol.json'),case=r,
                 **{k:r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')},
                 success=int(summary['collision_free_success'] and term=='success' and error is None),deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(error is not None),
                 episode_length=steps,outcome=term,elapsed_seconds=time.monotonic()-t0,first_action=first,projection_calls=calls,error=error,summary=summary,
                 execution_semantics_version='scene_precision_guard_v1',jax_enable_x64=bool(jax.config.x64_enabled),execution_code_sha256=old.sha(__file__),
                 placement_runtime_sha256=p['placement_runtime_sha256'])
        assert not(rec['success'] and (rec['collision'] or rec['numerical_failure']))
        old.write(dest,rec);buffer.append(rec);new+=1;elapsed.append(rec['elapsed_seconds'])
        if len(buffer)>=8:append_journal(buffer,EXP,f'p2prime_{index}');buffer=[]
        if new%64==0:print(json.dumps(dict(index=index,new=new,last_seconds=elapsed[-1])),flush=True)
    if buffer:append_journal(buffer,EXP,f'p2prime_{index}')
    write(HERE/f'worker_{index}.json',dict(index=index,assigned=len(assigned),new=new,
                                          completed=all((HERE/'raw'/(r['rollout_uid']+'.json')).exists() for r in assigned),
                                          episode_cpu_seconds=sum(elapsed)))


def merge():
    old.HERE=HERE;old.EXPERIMENT=EXP
    old.merge()


def postflight():
    cases=read(BASE/'P2PRIME_CASES.json');valid=unknown=missing=quarantine=0
    with connect(True) as db:
        for r in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if row is None:missing+=1;continue
            if row['conflict_quarantined']:quarantine+=1
            elif row['numerical_failure']:unknown+=1
            else:valid+=1
    stats=dict(total=len(cases),valid=valid,unknown=unknown,missing=missing,quarantined=quarantine,
               new_raw=len(list((HERE/'raw').glob('*.json'))),new_expected=len(read(BASE/'P2PRIME_MISSING_CASES.json')),
               workers_done=all(read(HERE/f'worker_{i}.json')['completed'] for i in range(22)))
    assert missing==quarantine==0 and stats['workers_done'] and stats['new_raw']==stats['new_expected'],stats
    write(BASE/'P2PRIME_POSTFLIGHT.json',stats);print(json.dumps(stats))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['setup','worker','merge','postflight']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=24)
    args=ap.parse_args()
    globals()[args.stage](args.index,args.workers) if args.stage=='worker' else globals()[args.stage]()

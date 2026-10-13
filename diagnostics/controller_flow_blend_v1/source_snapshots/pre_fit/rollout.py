"""New blended execution only; source runtimes reused unchanged."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from common import *
import time,argparse
import numpy as np
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import connect

def execute(folder,rows,tag,extra=False):
    from blend_runtime import make
    from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
    import jax
    p=read(folder/'protocol.json');exp=p['experiment_uid']
    assert sha(HERE/'blend_runtime.py')==p['runtime_sha256']
    assert sha(ROOT/'diagnostics/controller_q_transfer_round3_v1/flow_adapter.py')==p['flow_adapter_sha256']
    assert read(folder/'CACHE_INVENTORY.json')['missing']<=read(folder/'CACHE_INVENTORY.json')['new_cap']
    states={s['state_uid']:s for s in p['states']};cache={};new=0;secs=[];initialization=[];start=time.monotonic()
    cutoff=datetime.fromisoformat(read(HERE/'RESOURCE_AUTHORIZATION.json')['extra_capacity_cutoff_utc'])
    for r in rows:
        if extra and (cutoff-datetime.now(timezone.utc)).total_seconds()<120:break
        dest=folder/'raw'/(r['rollout_uid']+'.json')
        if dest.exists():
            previous=read(dest);assert previous['case']==r and previous['protocol_sha256']==sha(folder/'protocol.json')
            # An interrupted worker may have published raw before its journal.
            append_journal([previous],exp,tag);continue
        name=r['controller_name'];t0=time.monotonic();rt=make(name,p,cache);initialization.append(time.monotonic()-t0)
        jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled
        s=states[r['state_uid']];env=rt.make_env(s);term='timeout';error=None;steps=0;calls=0;first=None;t0=time.monotonic()
        for step in range(rt.config.max_steps):
            try:
                action,cert,n=rt.action(env,key_for(s,r['seed'],step),np.asarray(r['eta']),'field')
                calls+=n;steps+=1
                if first is None:first=np.asarray(action).tolist()
                _,_,done,info=env.step(action);term=info['termination']
                if done:break
            except Exception as exc:
                error=dict(step=step,type=type(exc).__name__,message=str(exc)[:500]);term='numerical_failure';break
        summary=env.summary();collision=any(summary.get(k,False) for k in ('wall_collision','obstacle_collision','outer_collision','agent_collision'))
        record=dict(schema='controller_flow_blend_seed_v1',experiment_uid=exp,protocol_sha256=sha(folder/'protocol.json'),case=r,
            **{k:r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')},
            success=int(summary['collision_free_success'] and term=='success' and error is None),deadlock=int(term=='deadlock'),timeout=int(term=='timeout'),collision=int(collision),numerical_failure=int(error is not None),
            episode_length=steps,outcome=term,elapsed_seconds=time.monotonic()-t0,first_action=first,projection_calls=calls,error=error,summary=summary,
            execution_semantics_version='scene_precision_guard_v1',jax_enable_x64=False,execution_code_sha256=sha(__file__),runtime_sha256=p['runtime_sha256'])
        assert not(record['success'] and (record['collision'] or record['numerical_failure']))
        write(dest,record);append_journal([record],exp,tag);new+=1;secs.append(record['elapsed_seconds'])
        if new%8==0:print(json.dumps(dict(tag=tag,new=new,last_seconds=secs[-1],wall_seconds=time.monotonic()-start)),flush=True)
    result=dict(tag=tag,assigned=len(rows),new=new,complete=all((folder/'raw'/(r['rollout_uid']+'.json')).exists() for r in rows),elapsed_episode_seconds=secs,
        episode_cpu_seconds=sum(secs),mean_seconds=float(np.mean(secs)) if secs else None,initialization_seconds=sum(initialization),wall_seconds=time.monotonic()-start,jax_enable_x64=False)
    write(folder/f'{tag}.json',result);print(json.dumps({k:v for k,v in result.items() if k!='elapsed_episode_seconds'}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['pilot','worker','merge','postflight','timing']);ap.add_argument('--stage',default='stage_1');ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=24);ap.add_argument('--extra',action='store_true');args=ap.parse_args();folder=HERE/args.stage
    if args.command in ('pilot','worker'):
        rows=read(folder/'MISSING_CASES.json')
        if args.command=='pilot':rows=[rows[int(i)] for i in np.linspace(0,len(rows)-1,min(98,len(rows)),dtype=int)]
        execute(folder,rows[args.index::args.workers],f'{args.command}_{args.index}',args.extra)
    elif args.command=='timing':
        rr=[read(folder/f'pilot_{i}.json') for i in range(args.workers)];assert all(r['complete'] for r in rr)
        secs=np.concatenate([r['elapsed_episode_seconds'] for r in rr]);assert len(secs)==98
        record=dict(utc=utc(),requests=len(secs),mean_seconds=float(secs.mean()),median_seconds=float(np.median(secs)),p90_seconds=float(np.quantile(secs,.9)),initialization_cpu_seconds=sum(r['initialization_seconds'] for r in rr),
            stage1_new_cap=5000,stage2_full_requests=61440,stage2_short_probes=832,stage2_max_total=62272,
            stage1_cpu_hours=float(4608*secs.mean()/3600),stage2_full_cpu_hours=float(61440*secs.mean()/3600),
            stage1_wall_hours_24=float(4608*secs.mean()/86400),stage2_wall_hours_24=float(61440*secs.mean()/86400),stage2_wall_hours_18=float(61440*secs.mean()/64800),
            caveat='Initialization, tails, static features and training are additional; timing pilot is included in requested labels')
        write(folder/'PILOT_TIMING.json',record);print(json.dumps(record))
    elif args.command=='merge':
        from diagnostics.controller_q_diversity_v1 import experiment
        p=read(folder/'protocol.json')
        with connect(True) as db:assert db.execute('SELECT protocol_hash FROM experiment WHERE experiment_uid=?',(p['experiment_uid'],)).fetchone()[0]==sha(folder/'protocol.json')
        experiment.HERE=folder;experiment.EXPERIMENT=p['experiment_uid'];experiment.merge()
    elif args.command=='postflight':
        cases=read(folder/'CASES.json');counts=dict(total=len(cases),valid=0,unknown=0,missing=0,quarantined=0)
        with connect(True) as db:
            for r in cases:
                x=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
                counts['missing' if x is None else 'quarantined' if x['conflict_quarantined'] else 'unknown' if x['numerical_failure'] else 'valid']+=1
        counts.update(raw=len(list((folder/'raw').glob('*.json'))),utc=utc())
        write(folder/'POSTFLIGHT.json',counts);assert counts['missing']==counts['quarantined']==0,counts;print(json.dumps(counts))
if __name__=='__main__':main()

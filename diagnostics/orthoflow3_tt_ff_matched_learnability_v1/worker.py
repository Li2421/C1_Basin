"""Missing-seed-only task workers, immutable journals, frozen TT/FF execution."""
from __future__ import annotations
import argparse,hashlib,json,time
import numpy as np
from .design import ROOT,EXPERIMENT,read,write,sha,canonical,FIELD
from .cache import protocol,raw_record,append_journal
from .execution import rollout,key_for


def run(batch,worker,workers):
    p=protocol();folder=ROOT/'batches'/batch;pre=read(folder/'identity_preflight.json')
    assert pre['cases_sha256']==sha(folder/'cases.json') and pre['protocol_sha256']==sha(ROOT/'protocol.json')
    assert (ROOT/'execution_smoke.json').exists() and read(ROOT/'execution_smoke.json')['success']
    rows=read(folder/'cases.json');selected=[r for i,r in enumerate(rows) if i%workers==worker]
    states={s['state_uid']:s for s in p['states']};dest=folder/'results';dest.mkdir(exist_ok=True)
    from validation_stage2.run_rollouts import make_runtime
    import jax
    runtime=None;scene=None;buf=[];attempts=0;recovered=0;journals=[]
    def flush():
        if buf:
            journals.append(str(append_journal(buf,EXPERIMENT,f'{batch}_worker{worker}')));buf.clear()
    for r in selected:
        if r['split']=='test':assert (ROOT/'models_frozen.json').exists()
        rid=hashlib.sha256(canonical(r).encode()).hexdigest();path=dest/(rid+'.json')
        if path.exists():record=read(path);recovered+=1
        else:
            if scene!=r['scene']:runtime=make_runtime(r['scene']);scene=r['scene']
            ctl=p['controllers'][scene][r['chain']]
            assert sha(runtime.checkpoint)==ctl['payload']['flow_checkpoint_sha256']
            assert bool(jax.config.jax_enable_x64)==ctl['payload']['jax_enable_x64'] and runtime.steps==10
            c=dict(id=rid,state=states[r['state_uid']],eta=r['eta'],eta_index=r['eta_index'],seed=r['seed'],phase='tt_ff_matched')
            raw=rollout(runtime,c,r['chain']);record=raw_record(raw,c['state'],r['chain'],'new',dict(batch=batch,worker=worker,case_id=rid))
            assert record['seed_key']==r['seed_key'] and record['controller_uid']==r['controller_uid']
            tmp=path.with_suffix('.tmp');write(tmp,record);tmp.replace(path);attempts+=1
        buf.append(record)
        if len(buf)>=32:flush()
    flush()
    write(folder/f'worker{worker}.json',dict(worker=worker,workers=workers,cases=len(selected),new_attempts=attempts,
        recovered_without_rerun=recovered,journals=journals,protocol_sha256=sha(ROOT/'protocol.json'),completed=True))
    print(json.dumps(dict(batch=batch,worker=worker,new_attempts=attempts,recovered=recovered,completed=True)),flush=True)


def smoke():
    p=protocol();from validation_stage2.run_rollouts import make_runtime
    from validation_stage2.factorial_controller import action as reference_action
    import jax
    rows=[]
    for scene in ('toy_give_way','ring_exchange'):
        rt=make_runtime(scene);state=next(s for s in p['states'] if s['scenario']==scene and not s['uid'].startswith('tt_ff_v1_'))
        eta=np.array(p['eta'][0]);key=key_for(state,0,0)
        if scene=='toy_give_way':
            from field_pipeline_v1.external_toy_proposals import key_for as original_key
            from field_pipeline_v1.external_toy_runner import controller_uid
        else:
            from field_pipeline_v1.external_ring_proposals import key_for as original_key
            from field_pipeline_v1.ring_t0_source import controller_uid
        np.testing.assert_array_equal(key,original_key(state,0,0));assert controller_uid()==p['controllers'][scene]['FF']['controller_uid']
        for chain,mode in (('TT','old'),('FF','field')):
            env=rt.make_env(state);got,cert,n=rt.action(env,key,eta,mode)
            reference,rcert,_,_=reference_action(rt,rt.make_env(state),key,eta,chain)
            np.testing.assert_allclose(got,reference,atol=1e-12,rtol=0)
            archived_error=None
            if chain=='FF':
                import sqlite3
                with sqlite3.connect('file:'+str(FIELD/'field_pipeline_v1/field_rollout.sqlite')+'?mode=ro',uri=True) as db:
                    rr=db.execute("SELECT result_json FROM continuation WHERE state_uid=? AND seed=0 AND phase IN ('toy_source_screen','ring_t0_screen')",(state['uid'],)).fetchall()
                match=next(json.loads(r[0]) for r in rr if json.loads(r[0])['eta']==p['eta'][0])
                archived_error=float(np.max(abs(got-np.asarray(match['first_action']))));assert archived_error<1e-10
            rows.append(dict(scene=scene,chain=chain,reference_action_error=float(np.max(abs(got-reference))),
                historical_FF_error=archived_error,native_x64=bool(jax.config.jax_enable_x64),safe=cert['max_violation']<=1e-9))
    write(ROOT/'execution_smoke.json',dict(success=True,cases=rows,new_task_rollouts=0,protocol_sha256=sha(ROOT/'protocol.json'),
        execution_sha256=sha(ROOT/'execution.py'),rng_key_parity=True,TT_FF_UID_disjoint=True))
    print(json.dumps(dict(smoke=True,new_task_rollouts=0,rows=rows)),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('run','smoke'));a.add_argument('--batch');a.add_argument('--worker',type=int,default=0);a.add_argument('--workers',type=int,default=5);q=a.parse_args()
    smoke() if q.action=='smoke' else run(q.batch,q.worker,q.workers)

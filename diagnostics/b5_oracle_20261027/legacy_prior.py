"""Strong legacy B3 comparator selected only from archived seed0..15 evidence."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from legacy import OUT,OLD,db,sha,dump,register,work,init_worker


def main():
    tasks=json.load(open(OUT/'LEGACY_FROZEN.json'))['tasks']
    old=[json.load(p.open())for p in (OLD/'validation_v2/results').glob('*_old.json')]
    cases=[];frozen={}
    for c in tasks:
        if c['method']!='B3':continue
        sid=c['state']['uid'];rr=[r for r in old if r['state_uid']==sid and r['seed']<16 and r['eta_name']!='zero']
        groups={name:[r for r in rr if r['eta_name']==name]for name in sorted({r['eta_name']for r in rr})}
        label=min(groups,key=lambda name:(-sum(r['success']for r in groups[name])/len(groups[name]),sum(r['steps']for r in groups[name])/len(groups[name]),name))
        e=groups[label][0]['eta'];assert all(lo<=v<=hi for lo,v,hi in zip([.5,-.5,0.],e,[1.25,.5,.75]))
        frozen[sid]=dict(eta=e,source_label=label,source_trials=len(groups[label]),source_success=sum(r['success']for r in groups[label]))
        c=deepcopy(c);c['method']='B3_archived';c['spec']=dict(mode='b3',eta=e+[0.,0.])
        c['payload'].update(method=c['method'],spec=c['spec'],selection=frozen[sid],selection_code=sha(__file__))
        c['controller_uid']=db.uid('ctl',c['payload']);c['eta_uid']=db.eta_identity(e)[0]
        c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
        cases.append(c)
    dump(OUT/'LEGACY_ARCHIVED_B3_FROZEN.json',frozen)
    # Preserve the original preflight artifacts as well as this supplement.
    missing=register(cases)
    (OUT/'legacy_prior_cache_preflight.json').write_bytes((OUT/'legacy_cache_preflight.json').read_bytes())
    (OUT/'legacy_prior_planned_rollouts.json').write_bytes((OUT/'legacy_planned_rollouts.json').read_bytes())
    path=OUT/'legacy_prior_candidates.jsonl';done={r['rollout_uid']for r in map(json.loads,path.open())}if path.exists()else set()
    todo=[c for c in cases if c['rollout_uid']not in done]
    assert all(tuple(c[k]for k in ('state_uid','eta_uid','controller_uid','seed_key'))in missing for c in todo)
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=init_worker)as pool,path.open('a')as f:
        for fu in as_completed([pool.submit(work,c)for c in todo]):
            r=fu.result();f.write(json.dumps(r)+'\n');f.flush();print(r['scenario'],r['method'],r['outcome'],flush=True)

if __name__=='__main__':main()

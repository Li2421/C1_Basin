"""Exact fresh-state request preflight and deduplicated outcome tensors."""
from common import *
from actions import case
from registry import register
from early_reuse import infer,validated
from methods import METHODS

def design():
    prereg();assert (HERE/'NEW_EVAL_CHOICES.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'NEW_EVAL_STATES.json');choices=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json');rows={};profiles={};links=[]
    for r in choices['states']:
        for q in (0,1):
            out={}
            for method,a in r['methods'].items():
                c,cp=case(base,r['controller'],states[r['state_index']],a,q,r['state_index'],a.get('candidate_index',-1));rid=c['rollout_uid'];rows.setdefault(rid,c);profiles[cp['controller_uid']]=cp;out[method]=rid
            links.append(dict(controller=r['controller'],type=r['type'],state_index=r['state_index'],seed=q,methods=out))
    assert len(rows)<=28416;register('B_evaluation',dict(base,states=states,phase='new_evaluation'),list(rows.values()),profiles);write(HERE/'B_evaluation_ACTION_LINKS.json',links);missing=read(HERE/'B_evaluation_MISSING_CASES.json');aliases={}
    with rollout_db.connect(True) as db:
        for c in missing:
            a=infer(c,base,db,validated())
            if a:aliases[c['rollout_uid']]=a
    execution=[c for c in missing if c['rollout_uid'] not in aliases];write(HERE/'B_evaluation_LOCAL_EQUIVALENCE.json',aliases);write(HERE/'B_evaluation_EXECUTION_CASES.json',execution);write(HERE/'B_evaluation_BUDGET.json',dict(utc=utc(),maximum_requests=28416,unique_physical_requests=len(rows),exact_cached=len(rows)-len(missing),early_equivalent=len(aliases),real_execution_upper_limit=len(execution)))
    seal('B_evaluation_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('B_evaluation_protocol.json','B_evaluation_CASES.json','B_evaluation_PREFLIGHT.receipt.json','B_evaluation_LOCAL_EQUIVALENCE.json','B_evaluation_EXECUTION_CASES.json','B_evaluation_BUDGET.json','B_evaluation_ACTION_LINKS.json','b_evaluation.py','rollout_worker_v3.py','dynamic_reuse.py','early_reuse.py','NEW_EVAL_CHOICES.receipt.json','FROZEN_PIPELINE.receipt.json')],before_submission=True)

def assemble():
    assert (HERE/'B_evaluation_COMPLETE.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');names=read(HERE/'OPENING_CONFIG.json')['B_evaluation'];methods=list(METHODS);shape=(74,24,len(methods),2);labels=np.full(shape,-2,np.int8);uids=np.full(shape,'',dtype='U96');events={k:np.zeros(shape,np.int8) for k in ('collision','timeout','unknown')};aliases=read(HERE/'B_evaluation_ALL_LOCAL_EQUIVALENCE.json')
    with rollout_db.connect(True) as db:
        for link in read(HERE/'B_evaluation_ACTION_LINKS.json'):
            for m,rid in link['methods'].items():
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone();r=dict(row) if row else aliases[rid]['outcome'];ix=(names.index(link['controller']),link['state_index'],methods.index(m),link['seed']);assert labels[ix]==-2;labels[ix]=-1 if r['numerical_failure'] else r['success'];uids[ix]=rid
                for k in events:events[k][ix]=r['numerical_failure'] if k=='unknown' else r[k]
    assert not np.any(labels==-2);np.savez_compressed(HERE/'NEW_EVAL_BUNDLE.npz',names=names,types=[base['types'][n] for n in names],methods=methods,labels=labels,uids=uids,**events)
    seal('NEW_EVAL_BUNDLE.receipt.json',[HERE/n for n in ('NEW_EVAL_BUNDLE.npz','B_evaluation_COMPLETE.receipt.json','b_evaluation.py','NEW_EVAL_CHOICES.receipt.json')])
if __name__=='__main__':globals()[sys.argv[1]]()

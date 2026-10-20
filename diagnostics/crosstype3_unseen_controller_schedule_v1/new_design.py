"""Official exact preflight and fixed actual evaluation cap, after all choices."""
from common import *
from design import action_case,freeze
def main():
    for receipt in ('FROZEN_PIPELINE.receipt.json','NEW_EVAL_CHOICES.receipt.json'):
        for path,h in read(HERE/receipt)['files'].items():assert sha(path)==h,path
    base=read(HERE/'EVALUATION_PROTOCOL_BASE.json');states=read(HERE/'NEW_EVAL_STATES.json');chosen=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states'];rows={};profiles={};maps=[];base_uids=set();conservative_uids=set()
    with rollout_db.connect(True) as db:
        assert all(db.execute('SELECT 1 FROM rollout WHERE state_uid=? LIMIT 1',(s['state_uid'],)).fetchone() is None for s in states)
    for record in chosen:
        name=record['controller'];si=record['state_index'];mapping={}
        for method,a in record['methods'].items():
            rids=[]
            for seed in (0,1):
                c,p=action_case(base,name,states[si],a,seed,si,a.get('candidate_index',-1));c['phase_state_index']=si;rid=c['rollout_uid'];rids.append(rid);rows.setdefault(rid,c);profiles[p['controller_uid']]=p
                (conservative_uids if method=='conservative' else base_uids).add(rid)
            mapping[method]=rids
        maps.append(dict(controller=name,type=record['type'],cohort=record['cohort'],state_index=si,state_uid=record['state_uid'],method_uids=mapping))
    write(HERE/'NEW_EVAL_METHOD_UIDS.json',maps)
    freeze('new_evaluation',dict(base,states=states,phase='new_evaluation'),list(rows.values()),profiles)
    missing={r['rollout_uid'] for r in read(HERE/'new_evaluation_MISSING_CASES.json')};extra=conservative_uids-base_uids
    assert len(missing&extra)<=3744
    budget=dict(utc=utc(),base_union_requests=len(base_uids),base_missing=len(missing&base_uids),conservative_additional_requests=len(extra),conservative_additional_missing_cap=len(missing&extra),conservative_absolute_ceiling=3744,total_missing_cap=len(missing),unseen_priority=True,authorization=str(HERE/'CONSERVATIVE_EVALUATION_AUTHORIZATION.receipt.json'))
    write(HERE/'NEW_EVAL_BUDGET.json',budget)
    seal('NEW_EVAL_BUDGET.receipt.json',[HERE/'NEW_EVAL_BUDGET.json',HERE/'new_evaluation_EXACT_BUDGET.json',HERE/'NEW_EVAL_METHOD_UIDS.json',HERE/'NEW_EVAL_CHOICES.receipt.json',HERE/'new_design.py',HERE/'CONSERVATIVE_EVALUATION_AUTHORIZATION.receipt.json'],before_evaluation_submission=True)
if __name__=='__main__':main()

"""Deduplicated frozen-action union and separately charged necessary baselines."""
from common import *
from registry import register
from design import action_case,DEPENDENCIES
def main():
    for receipt in ('FROZEN_PIPELINE.receipt.json','NEW_EVAL_CHOICES.receipt.json'):
        for path,h in read(HERE/receipt)['files'].items():assert sha(path)==h,path
    base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'NEW_EVAL_STATES.json');chosen=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states'];rows={};profiles={};maps=[];base_uids=set();extra_uids=set()
    for record in chosen:
        name=record['controller'];s=record['state_index'];state=states[s];mapping={}
        for method,a in record['methods'].items():
            c,p=action_case(base,name,state,a,0,s,a.get('candidate_index',-1));c['phase_state_index']=s;profiles[p['controller_uid']]=p;method_uids=[]
            for seed in (0,1):
                c,p=action_case(base,name,state,a,seed,s,a.get('candidate_index',-1));c['phase_state_index']=s;rid=c['rollout_uid'];method_uids.append(rid)
                if rid not in rows:rows[rid]=c
                if method in record['base_request_methods']:base_uids.add(rid)
                elif method in record['additional_request_methods']:extra_uids.add(rid)
            mapping[method]=method_uids
        maps.append(dict(controller=name,type=record['type'],state_index=s,state_uid=state['state_uid'],method_uids=mapping))
    extra_uids-=base_uids;assert set(rows)==base_uids|extra_uids
    protocol={**base,'states':states,'phase':'new_evaluation'}
    register('new_evaluation',protocol,list(rows.values()),profiles)
    missing={r['rollout_uid'] for r in read(HERE/'new_evaluation_MISSING_CASES.json')};extra_missing=len(missing&extra_uids);base_missing=len(missing&base_uids)
    budget=dict(utc=utc(),original_per_controller_state_action_cap=10,base_raw_request_cap=62*24*2*10,base_union_requests=len(base_uids),base_genuinely_missing=base_missing,additional_union_requests=len(extra_uids),additional_genuinely_missing=extra_missing,additional_cap=10000,additional_authorized_by_user=True,pure_training_requested=62*48*26,pure_training_missing=sum(len(read(HERE/f'{p}_MISSING_CASES.json')) for p in ('pilot','remaining')),base_total_missing_cap=93000,method_union_semantics='Base ten requested methods; other required methods charged only when their exact UID is absent from this base union')
    write(HERE/'NEW_EVAL_BUDGET.json',budget);write(HERE/'NEW_EVAL_METHOD_UIDS.json',maps)
    seal('NEW_EVAL_BUDGET.receipt.json',[HERE/'NEW_EVAL_BUDGET.json',HERE/'NEW_EVAL_METHOD_UIDS.json',HERE/'new_evaluation_PREFLIGHT.receipt.json',HERE/'USER_DECISIONS.receipt.json'])
    assert budget['pure_training_missing']+base_missing<=93000,'Exact baseline budget would exceed cap; stop before submitting'
    assert extra_missing<=10000,'Additional required baseline gap exceeds user-authorized cap; stop before submitting'
    seal('new_evaluation_EXECUTION_FROZEN.receipt.json',[HERE/'new_evaluation_protocol.json',HERE/'new_evaluation_CASES.json',HERE/'new_evaluation_PREFLIGHT.receipt.json',HERE/'NEW_EVAL_BUDGET.receipt.json',HERE/'NEW_EVAL_CHOICES.receipt.json',*[HERE/n for n in DEPENDENCIES],HERE/'new_design.py'],split='new_evaluation',before_rollouts=True)
    print('Fresh exact missing budget:',json.dumps(budget),flush=True)
if __name__=='__main__':main()

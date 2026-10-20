"""A diagnosis uses only previously sealed pairs; no model or pair reselection."""
from common import *
from actions import action,case,TIMES
from registry import register
from early_reuse import infer,validated

def main():
    prereg();assert read(HERE/'EXECUTION_EQUIVALENCE.json')['passed'];assert (HERE/'EARLY_TERMINATION_EQUIVALENCE.receipt.json').exists()
    base=read(HERE/'BASE_PROTOCOL.json');A=read(HERE/'A_OBJECTS.json');rows={};profiles={};links=[]
    def add(obj,act,q,role):
        c,cp=case(base,obj['controller'],obj['state'],act,q,obj['source_state_index']);rid=c['rollout_uid'];rows.setdefault(rid,c);profiles[cp['controller_uid']]=cp
        return rid
    for oi,obj in enumerate(A['records']):
        original=obj['selected_action'];a,b=original['eta_a'],original['eta_b'];pairdiag=obj['diagonal']
        for q in (0,1):
            methods={}
            methods['old_selected']=add(obj,action(a,b,'fixed',original['switch_step']),q,'old_selected')
            for t in TIMES:methods['selected_'+str(t)]=add(obj,action(a,b,'fixed',t),q,'selected_'+str(t))
            methods['selected_progress']=add(obj,action(a,b,'progress'),q,'selected_progress')
            for t in TIMES:methods['human_'+str(t)]=add(obj,action(base['eta'][13],base['eta'][10],'fixed',t),q,'human_'+str(t))
            methods['human_progress']=add(obj,action(base['eta'][13],base['eta'][10],'progress'),q,'human_progress')
            coverage={}
            if obj['coverage'] and q==0:
                for i in range(8):
                    for j in range(8):
                        if i==j:continue
                        coverage[f'{i},{j}']=add(obj,action(obj['proposals'][i],obj['proposals'][j],'progress'),q,'proposal_coverage')
            links.append(dict(object_index=oi,controller=obj['controller'],type=obj['type'],state_uid=obj['state_uid'],seed=q,diagonal=pairdiag,coverage=bool(coverage),methods=methods,proposal_pairs=coverage))
    p=dict(base,states=A['states'],phase='A');register('A',p,list(rows.values()),profiles);write(HERE/'A_ACTION_LINKS.json',links)
    vt=validated();aliases={}
    with rollout_db.connect(True) as db:
        missing=read(HERE/'A_MISSING_CASES.json')
        for k,c in enumerate(missing):
            alias=infer(c,base,db,vt)
            if alias:aliases[c['rollout_uid']]=alias
            if k%1000==0:print('A local preflight',k,'/',len(missing),'reusable',len(aliases),flush=True)
    execution=[c for c in missing if c['rollout_uid'] not in aliases]
    write(HERE/'A_LOCAL_EQUIVALENCE.json',aliases);write(HERE/'A_EXECUTION_CASES.json',execution)
    write(HERE/'A_BUDGET.json',dict(utc=utc(),requested_unique=len(rows),DB_present=len(rows)-len(missing),DB_missing=len(missing),early_equivalence=len(aliases),actual_execution_upper_limit=len(execution),validated_types=sorted(vt),engineering_validation_separate=True))
    seal('A_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('A_protocol.json','A_CASES.json','A_PREFLIGHT.receipt.json','A_ACTION_LINKS.json','A_LOCAL_EQUIVALENCE.json','A_EXECUTION_CASES.json','A_BUDGET.json','a_design.py','actions.py','early_reuse.py','EARLY_TERMINATION_EQUIVALENCE.receipt.json','EXECUTION_EQUIVALENCE.receipt.json','rollout_worker.py')],before_submission=True,no_retraining=True,no_pair_reselection=True)
    print('A actual execution upper limit',len(execution),flush=True)
if __name__=='__main__':main()

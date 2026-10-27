"""Result-only analysis; no controller inference or rollout generation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,csv
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
NEW=('T1','T2','T3','T4','T5','P2','R2','R3','Y1','Y2','Y3','Y4','S1','C1','C2')

def load(n):return json.load(open(OUT/n))
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def key(r):return(r['scene'],r['controller'],r['state_index'])
def avg(v):return float(np.mean(v))if len(v)else None

def ci(blocks):
    x=np.asarray(blocks,float)
    if len(x)<2:return None
    rng=np.random.default_rng(2026102815);sample=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return np.quantile(sample,[.025,.975]).tolist()

def metric(rr,baseline=None):
    blocks=defaultdict(list)
    for r in rr:blocks[(r['scene'],r['state_index'])].append(r['success'])
    x=dict(n=len(rr),success=sum(r['success']for r in rr),outcomes=dict(Counter(r['outcome']for r in rr)),
        success_ci=ci([np.mean(v)for v in blocks.values()]),physical_blocks=len(blocks),
        conditional_time=avg([r['completion_time']for r in rr if r['success']]),truncated_time=avg([r['truncated_time']for r in rr]),
        worker_seconds=avg([r['elapsed_seconds']for r in rr]),ms_per_step=avg([1000*r['elapsed_seconds']/r['episode_length']for r in rr if r['episode_length']]),
        restoration_steps=sum(r.get('restoration_steps',0)for r in rr),constraint_violation_steps=sum(r.get('constraint_violation_steps',0)for r in rr))
    if baseline:
        bb={key(r):r for r in baseline};pairs=[(r,bb[key(r)])for r in rr if key(r)in bb];ss=[(r,b)for r,b in pairs if r['success']and b['success']];dif=defaultdict(list)
        for r,b in pairs:dif[(r['scene'],r['state_index'])].append(r['success']-b['success'])
        x.update(paired_success=len(ss),paired_method_time=avg([r['completion_time']for r,b in ss]),paired_B3_time=avg([b['completion_time']for r,b in ss]),
            rescued=sum(r['success']and not b['success']for r,b in pairs),broken=sum(b['success']and not r['success']for r,b in pairs),
            paired_rate_ci=ci([np.mean(v)for v in dif.values()]),paired_truncated_delta=avg([r['truncated_time']-b['truncated_time']for r,b in pairs]))
    return x

def main():
    allrows=list(map(json.loads,open(OUT/'candidates.jsonl')));byid={r['rollout_uid']:r for r in allrows};assert len(byid)==len(allrows)
    p=load('PROTOCOL.json');hard={(c['scene'],c['controller'],c['state_index'])for c in p['cases']if c['hard']};guards={(c['scene'],c['controller'],c['state_index'])for c in p['cases']if not c['hard']}
    def stage(n):return [byid[c['rollout_uid']]for c in load(n+'_planned.json')['exact_configs']]
    def wins(rr):return set(key(r)for r in rr if r['success'])
    def group_summary(rr):return dict(candidates=len(rr),success_candidates=sum(r['success']for r in rr),rescued=len(wins(rr)&hard),guards=len(wins(rr)&guards),
        collision=sum(r['collision']for r in rr),numerical=sum(r['numerical_failure']for r in rr),cases=sorted(wins(rr)),
        new_seconds=sum(r['elapsed_seconds']for r in rr if r['source']=='new_rollout'))
    single=stage('single');primary={};variants={};budget=[]
    for name in NEW:
        rr=[r for r in single if list(r['spec']['coefficients'])==[name]and r['spec'].get('t5_seed','tangent')=='tangent'and r['spec'].get('clearance_side',1)==1]
        primary[name]=group_summary(rr);percase=defaultdict(list)
        for r in rr:percase[key(r)].append(r)
        for K in range(1,5):
            ww=set(k for k,v in percase.items()if any(r['success']for r in v[:K]))
            budget.append(dict(column=name,K=K,rescued=len(ww&hard),guards=len(ww&guards),hard_n=len(hard),guard_n=len(guards)))
    for recipe in ('tangent','goal','nominal'):
        variants['T5_'+recipe]=group_summary([r for r in single if list(r['spec']['coefficients'])==['T5']and r['spec']['t5_seed']==recipe])
    variants['Y3_right']=group_summary([r for r in single if list(r['spec']['coefficients'])==['Y3']and r['spec'].get('clearance_side')==-1])
    matched={label:group_summary([byid[i]for i in ids])for label,ids in load('MATCHED_CONTROLS.json')['labels'].items()}
    restart_labels=load('RESTART_LABELS.json');restart={}
    for name,ids in restart_labels['labels'].items():
        rr=[byid[i]for i in ids];entry=group_summary(rr);entry['rescued']=len(wins(rr));entry['hard_n']=len(restart_labels['failed_after_B3']);entry.pop('guards')
        # Initial unfinished group has to make .05m net goal progress, or arrive.
        delays=[]
        for r in rr:
            tr=np.load(r['trace']);d=np.linalg.norm(tr['goals'][None]-tr['positions'],axis=-1);active=d[0]>.08
            hits=np.flatnonzero(np.all((d[:,active]<=d[0,active]-.05)|(d[:,active]<=.08),axis=1))
            if len(hits):delays.append(hits[0]*.05)
        entry.update(diagnostic_progress_episodes=len(delays),conditional_progress_time=avg(delays));restart[name]=entry
    fixed={};fixed_by_scene={};fixed_by_controller={}
    for prefix,filename in [('test','TEST_FIXED_LABELS.json'),('confirm','CONFIRMATORY_FIXED_LABELS.json')]:
        labels=load(filename);baseline=[byid[i]for i in labels['B3_frozen']]
        fixed[prefix]={label:metric([byid[i]for i in ids],baseline)for label,ids in labels.items()}
        scenes=sorted({r['scene']for r in baseline});controllers=sorted({r['controller']for r in baseline})
        fixed_by_scene[prefix]={scene:{label:metric([byid[i]for i in ids if byid[i]['scene']==scene])for label,ids in labels.items()}for scene in scenes}
        fixed_by_controller[prefix]={ctl:{label:metric([byid[i]for i in ids if byid[i]['controller']==ctl])for label,ids in labels.items()}for ctl in controllers}
    oracle={}
    for prefix in ('test','confirm'):
        b16=wins(stage(prefix+'_b3_16'));b48=b16|wins(stage(prefix+'_b3_48'));b64=b48|wins(stage(prefix+'_b3_64'))
        informed=wins(stage(prefix+'_informed_b3'))
        x=dict(B3_16=len(b16),B3_48=len(b48),B3_64=len(b64),B3_informed16=len(informed),B3_union_max80=len(b64|informed),cohort=16)
        if prefix=='test':x['augmented64']={label:len(b48|wins([byid[i]for i in ids]))for label,ids in load('TEST_ORACLE_LABELS.json').items()}
        else:x.update(augmented64=len(b48|wins(stage('confirm_family_oracle'))),pure_family16=len(wins(stage('confirm_pure_family'))))
        oracle[prefix]=x
    restart_oracle=dict(B3_16=len(wins(stage('restart_b3'))),B3_48=len(wins(stage('restart_b3'))|wins(stage('restart_b3_48'))),cohort=14,C2_single_rescued=restart['C2']['rescued'],
        compressed_pool_rescued=len(wins(stage('restart_compressed'))),pure_compressed16=len(wins(stage('restart_pure_family'))))
    costs={}
    for name in NEW:
        rr=[r for r in single if list(r['spec']['coefficients'])==[name]and r['source']=='new_rollout'];stats=[r['dictionary']for r in rr if r.get('dictionary')]
        calls=sum(x['calls']for x in stats)
        costs[name]=dict(new_rollouts=len(rr),kernel_us_per_call=1e6*sum(x['seconds']for x in stats)/max(1,calls),
            svd_us_per_call=1e6*sum(x['svd_seconds']for x in stats)/max(1,calls),
            unavailable_fraction=sum(x['unavailable_steps'][name]for x in stats)/max(1,calls),
            mean_total_ms_step=avg([1000*r['elapsed_seconds']/r['episode_length']for r in rr if r['episode_length']]))
    sensitivity=load('mechanism_validation.json');changed=0;svdchanged=0
    for state in sensitivity['states']:
        ss=state['T5_sensitivity'];changed+=len({tuple(x['ranks'])for x in ss if x['svd_relative']==1e-10})>1
        svdchanged+=any(len({tuple(x['ranks'])for x in ss if x['near_relative']==tau})>1 for tau in (.01,.05,.1))
    x=dict(primary_single=primary,recipe_variants=variants,matched_controls=matched,restart=restart,restart_oracle=restart_oracle,
        minimal=load('MINIMAL_FAMILIES.json'),fixed=fixed,fixed_by_scene=fixed_by_scene,fixed_by_controller=fixed_by_controller,oracle=oracle,costs=costs,budget_curve=budget,
        static=load('static_summary.json'),T5_near_threshold_rank_changed_states=changed,T5_svd_tolerance_rank_changed_states=svdchanged,
        C2_fd_max_error=sensitivity['max_C2_fd_error'],records=len(allrows),sources=dict(Counter(r['source']for r in allrows)),outcomes=dict(Counter(r['outcome']for r in allrows)),
        numerical_failures=sum(r['numerical_failure']for r in allrows),new_simulator_seconds=sum(r['elapsed_seconds']for r in allrows if r['source']=='new_rollout'),
        maximum_worker_cpu=max(cpu for r in allrows if r['source']=='new_rollout'for cpu in r['worker_cpu']),
        restoration_steps=sum(r.get('restoration_steps',0)for r in allrows),original_constraint_violation_steps=sum(r.get('constraint_violation_steps',0)for r in allrows))
    x['minimal'].pop('witnesses');dump('summary.json',x)
    with (OUT/'budget_curves.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(budget[0]),lineterminator='\n');w.writeheader();w.writerows(budget)
    with (OUT/'single_screen.csv').open('w')as f:
        keys=['column','candidates','success_candidates','rescued','guards','collision','numerical','new_seconds'];w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader()
        for n,v in primary.items():w.writerow(dict(column=n,**{k:v[k]for k in keys[1:]}))
    print('records',len(allrows),'new',x['sources'].get('new_rollout'),'oracle',oracle,'restart',restart_oracle)

if __name__=='__main__':main()

"""Frozen discovery cohort; selection uses only prior experiment evidence."""
import json
from pathlib import Path
from runner import OUT,OLD,ROOT,dump,digest,file_sha,historical,inputs,make_case
from shared_control.basis_dictionary import NEW,SIGNED
import numpy as np


def key(r):return(r['scene'],r['controller'],r['state_index'])
def merit(r):return(-r['success'],r['truncated_time']if r['success']else sum(r.get('final_goal_distances',[1e20])),r['rollout_uid'])
def original_b3(r):
    sp=r['spec'];e=sp['eta']
    return sp['mode']=='b3'and not sp.get('theta')and not sp.get('schedule')and .5<=e[0]<=1.25 and -.5<=e[1]<=.5 and 0<=e[2]<=.75 and not sp.get('test_label')

def freeze():
    path=OUT/'PROTOCOL.json'
    if path.exists():return json.load(path.open())
    rows=[r for r in historical()if original_b3(r)and r['seed']==0]
    scenes,ctls,_,_=inputs();types=[c['type_code']for c in ctls]
    hard=[(s,types[ci],si)for s,ci,si in json.load(open(OLD/'expanded_A168_failures.json'))]
    dev=['xsc_hs_plus_w025','xsc_hs_box_n09_r014_f25','xsc_hs_swap_n07','hs_obst_n08_R140_r045']
    guard=[]
    for s in dev:
        for ctl in types:
            valid=sorted({key(r)for r in rows if r['scene']==s and r['controller']==ctl and r['success']and 8<=r['state_index']<=15})
            guard.append(valid[0])
    cohort=hard+guard;cases=[]
    for k in cohort:
        rr=[r for r in rows if key(r)==k];best=min(rr,key=merit)
        traces=[r for r in rr if r.get('trace')and not r.get('numerical_failure')]
        failed=[r for r in traces if not r['success']];trace=min(failed or traces,key=merit)
        cases.append(dict(scene=k[0],controller=k[1],state_index=k[2],hard=k in hard,baseline_eta=best['spec']['eta'][:3],
            baseline_rollout_uid=best['rollout_uid'],baseline_success=best['success'],diagnostic_trace=trace['trace']))
    pair_pilot=[]
    for scene in sorted({k[0]for k in hard}):pair_pilot.extend([c for c in cases if c['hard']and c['scene']==scene][:2])
    x=dict(base_commit='5da96e7',dictionary=list(NEW),cases=cases,pair_pilot=pair_pilot,
        nominal_frozen=ctls,test_scenes=['xsc_hs_plus_w030_te','xsc_hs_box_n09_r013_f25_te','xsc_hs_swap_n06_te','hs_obst_n08_R142_r050_te'],
        test_states=[18,19],test_seeds=[3],selection='prior B3 failures14 + prior successes8; test untouched; one seed avoids pseudo-replication',
        evidence=dict(prior_report=str(OLD/'report.md'),prior_results_sha256=file_sha(OLD/'candidates_verified.jsonl'),
            reused='All exact compatible B3/T1/T2 outcomes; original B3 Oracle remains baseline subset.',
            missing='15-class dictionary, activation rules, pair interactions and new heldout state/seed evidence do not exist.'),
        no_gpu=True,no_training=True,resource_policy='Use currently idle Slurm CPU allocation, 24 CPUs found free; BLAS1',
        rounds=['all single columns, four configurations each','all105 pairs, two Sobol configurations on6 stratified hard states',
                'automatic promising pair/full sparse expansion on22 cases','automatic deletion and minimum witness-cover analysis','frozen holdout'],
        limitation='Finite support/parameter screening, not globally optimal minimal continuous family.')
    dump(path,x);return x


def case(c,sp,seed=0):
    scenes,ctls,_,_=inputs();ctl=next(x for x in ctls if x['type_code']==c['controller'])
    return make_case(scenes[c['scene']],ctl,c['state_index'],sp,seed)


def singles(c):
    out=[]
    for name in NEW:
        for base in (c['baseline_eta'],[.875,0.,.375]):
            for gain in ((-.75,.75)if name in SIGNED else(.5,1.)):
                out.append(dict(eta=base,coefficients={name:gain}))
        if name=='T5':
            for seed in ('goal','nominal'):
                for gain in(-.75,.75):out.append(dict(eta=c['baseline_eta'],coefficients={name:gain},t5_seed=seed))
        if name=='Y3':
            for gain in(.5,1.):out.append(dict(eta=c['baseline_eta'],coefficients={name:gain},clearance_side=-1))
    return out

if __name__=='__main__':print('Frozen',len(freeze()['cases']),'discovery cases')

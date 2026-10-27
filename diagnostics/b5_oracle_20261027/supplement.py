"""Matched interventions after expanded fixed-family search; no new model fitting."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json
from copy import deepcopy
from oracle import OUT,inputs,make_case,execute,b3_pool,dump


def main():
    scenes,ctls,_,_=inputs();failures=json.load(open(OUT/'expanded_A168_failures.json'))
    rows=list(map(json.loads,(OUT/'candidates.jsonl').open()));cases=[];winners=[]
    for s,ci,si in failures:
        for mode in ('add','blend'):
            for recipe in ('goal','cluster','wall'):
                rr=[r for r in rows if r['scene']==s and r['state_index']==si and r['controller']==ctls[ci]['type_code']and r['spec']['mode']==mode and r['spec'].get('recipe')==recipe and not r['spec'].get('keep')and not r['spec'].get('schedule')and not r['spec']['name'].endswith('_g0')and not r['spec'].get('test_label')]
                best=min(rr,key=lambda r:(-r['success'],r['truncated_time'],sum(r.get('final_goal_distances',[1e9])),r['spec']['name']))
                sp=dict(best['spec'],name='release_'+best['spec']['name'],schedule='release_third',release_eta=[.875,0.,.375])
                cases.append(make_case(scenes[s],ctls[ci],si,sp,0))
                if not best['success']:continue
                winners.append(dict(scene=s,controller_index=ci,state_index=si,source=best['rollout_uid'],spec=best['spec']))
                for label,keep in [('GRTY',[0,2,3,4]),('GPTY',[0,1,3,4]),('GTR',[0,2,3]),('GTY',[0,3,4]),('noT',[0,1,2,4]),('noY',[0,1,2,3])]:
                    sp=dict(best['spec'],name=label+'_'+best['spec']['name'],keep=keep,ablation=label)
                    cases.append(make_case(scenes[s],ctls[ci],si,sp,0))
        for k,base in enumerate(b3_pool()[:16]):
            sp=dict(base,name='sched_'+base['name'],theta=.5,eta_b=b3_pool()[(k+8)%16]['eta'][:3])
            cases.append(make_case(scenes[s],ctls[ci],si,sp,0))
    dump(OUT/'supplement_frozen.json',dict(winners=winners,scope='Paired coefficient deletion on successful witnesses and scheduled best fixed candidates; not a reoptimized reduced-family oracle except pilot E.'))
    execute(cases,'supplement',16)


if __name__=='__main__':main()

"""Freeze pilot choices, then test all methods on the same untouched cohort."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib
from copy import deepcopy
import numpy as np
from oracle import OUT,inputs,DEV_SCENES,TEST_SCENES,make_case,execute,dump


def freeze():
    rows=list(map(json.loads,(OUT/'candidates.jsonl').open()))
    # Only the initial pilot, with the original box, fixed schedule, full columns.
    pilot=[r for r in rows if r['state_index']==8 and r['scene']in DEV_SCENES and not r['spec'].get('keep')
        and not r['spec'].get('theta')and not r['spec'].get('schedule')and not r['spec']['name'].endswith('_g0')and not r['spec']['name'].startswith('b3g0')]
    def rank(r):return (-r['success'],r['truncated_time'],sum(r.get('final_goal_distances',[1e9])),r['spec']['name'])
    designs={}
    for mode in ('add','blend'):
        for recipe in ('goal','cluster','wall'):
            rr=[r for r in pilot if r['spec']['mode']==mode and r['spec'].get('recipe')==recipe]
            best=min(rr,key=rank);sp=deepcopy(best['spec'])
            if sp.get('role'):
                sp['role_order']='ascending'if sp['role']['yielder']<sp['role']['passer']else'descending'
            designs[mode+'_'+recipe]=sp
    designs['T_only']=dict(name='frozen_T_only',mode='add',recipe='goal',eta=[.875,0.,.375,-.75,0.])
    full=designs['add_goal']
    for label,keep in [('GRTY',[0,2,3,4]),('GPTY',[0,1,3,4]),('GTR',[0,2,3]),('GTY',[0,3,4])]:
        designs[label]=dict(deepcopy(full),keep=keep,name='frozen_'+label)
    designs['add_goal_release']=dict(deepcopy(full),schedule='release_third',release_eta=[.875,0.,.375],name='frozen_release')
    designs['safety']=dict(name='safety',mode='b3',eta=[0.]*5)
    for name,mode in [('RHR_A','exact_rhr_goal'),('RHR_B','exact_rhr_centre')]:
        designs[name]=dict(name=name,mode=mode,eta=[0.]*5,adapter_sha256=hashlib.sha256((OUT/'rhr_exact.py').read_bytes()).hexdigest())
    b3={}
    for s in DEV_SCENES:
        for ctl in inputs()[1]:
            rr=[r for r in pilot if r['scene']==s and r['controller']==ctl['type_code']and r['spec']['mode']=='b3']
            b3[s+'__'+ctl['type_code']]=min(rr,key=rank)['spec']
    frozen=dict(designs=designs,b3=b3,source='Initial pilot state8 only, before any independent test; global B5 transfer to both controllers.',
        role_policy='Nearest initial pair; preserve frozen ordered orientation and waypoint. No test outcome selects role or T.',
        test_scenes=TEST_SCENES,test_states=[16,17],test_future_indices=[1,2])
    path=OUT/'INDEPENDENT_FROZEN.json'
    if path.exists():assert json.load(path.open())==frozen
    else:dump(path,frozen)
    return frozen


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--workers',type=int,default=16);a=ap.parse_args()
    frozen=freeze()
    if a.freeze_only:return
    scenes,ctls,_,_=inputs();cases=[]
    for s,dev in zip(TEST_SCENES,DEV_SCENES):
        for ctl in ctls:
            designs=dict(frozen['designs'],B3=frozen['b3'][dev+'__'+ctl['type_code']])
            for si in (16,17):
                p=np.asarray(scenes[s]['states'][si]['positions']);d=np.linalg.norm(p[:,None]-p[None,:],axis=-1);np.fill_diagonal(d,np.inf)
                i,j=map(int,np.unravel_index(np.argmin(d),d.shape))
                for label,original in designs.items():
                    sp=deepcopy(original);sp['test_label']=label
                    if sp.get('role'):
                        ii,jj=(i,j)if sp['role_order']=='ascending'else(j,i)
                        sp['role'].update(yielder=ii,passer=jj)
                    cases.extend(make_case(scenes[s],ctl,si,sp,seed)for seed in (1,2))
    execute(cases,'independent',a.workers,reuse=False)


if __name__=='__main__':main()

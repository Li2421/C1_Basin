#!/usr/bin/env python3
"""Distinguish form failure from >32-label oracle dependence, offline only."""
import csv,json,itertools,time
from pathlib import Path
from collections import defaultdict
from revise_oracle import select
from shared_families import fit
from cross_validate import read,write,dump,make_cloud,metrics,state_score,gate_summary,hpkey

HERE=Path(__file__).resolve().parent
FAMS={
 'affine_superbody':{'p':(2,4,6),'cuts':(0,)},
 'superbody_with_cuts':{'p':(2,4,6),'cuts':(1,2)},
 'native_conditional_band':{'p':(2,4,6),'cuts':(0,1)},
 'rotated_asymmetric_slab':{'p':(2,4,6),'cuts':(0,1)},
 'domain_minus_boundary_caps':{'p':(2,),'cuts':(2,3,4)},
}
def main():
 if (HERE/'oracle_heavy_diagnostic.json').exists():raise FileExistsError('frozen')
 panel={r['state_id'] for r in read(HERE/'scenario_state_inventory.csv')};inv=[r for r in read(HERE/'exact_q64_inventory.csv') if r['state_id'] in panel and r['in_E_bridge']=='True'];by=defaultdict(list)
 for r in inv:by[r['state_id']].append(r)
 subs={s:[r['eta_key'] for r in select(rr,48)] for s,rr in by.items()};dump('oracle_budget_48_diagnostic.json',dict(status='DIAGNOSTIC_ABOVE_PRIMARY_32_LABEL_TARGET',same_shared_V1_policy=True,states=subs))
 folds=json.load(open(HERE/'cv_folds.json'))['folds'];cloud=make_cloud();out=[];t=time.time()
 for family,cfg in FAMS.items():
  hps=[dict(p=p,gamma=g,max_cuts=k,safety=.9) for p,g,k in itertools.product(cfg['p'],(.5,.6,.7,.8,.9),cfg['cuts'])];allm={}
  for hp in hps:
   hk=hpkey(hp)
   for sid,rr in by.items():
    lookup={r['eta_key']:r for r in rr};keys=subs[sid];m=fit(family,[lookup[k] for k in keys],**hp);allm[sid,hk]=metrics(sid,rr[0]['scenario'],m,rr,set(keys),cloud,48,hp)
  outer=[];fh=[]
  for f in folds:
   best=max(hps,key=lambda hp:state_score([allm[s,hpkey(hp)] for s in f['development']]));fh.append(dict(fold=f['fold'],hyperparameters=best,development_score=state_score([allm[s,hpkey(best)] for s in f['development']])))
   outer += [dict(allm[s,hpkey(best)],fold=f['fold']) for s in f['heldout']]
  gate=gate_summary(outer,family,fh);out.append(gate);write(f'oracle_heavy/{family}_48label_cv.csv',outer);dump(f'oracle_heavy/{family}_gate.json',gate)
 dump('oracle_heavy_diagnostic.json',dict(runtime_seconds=time.time()-t,budget=48,gates=out,any_cached_screen_pass=any(g['cached_screen_pass'] for g in out)))
 print(json.dumps([{'family':g['family'],'usable':g['usable_states'],'recall':g['median_independent_B63_recall'],'transfer':g['median_transfer_coverage'],'pass':g['cached_screen_pass']} for g in out]))
if __name__=='__main__':main()

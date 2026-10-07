#!/usr/bin/env python3
"""Scenario-stratified state-level CV for shared conservative families."""
import csv,json,hashlib,itertools,time,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist
from shared_families import *

HERE=Path(__file__).resolve().parent
FAMILIES={
 'affine_superbody':{'directory':'affine_superbody','cuts':[0]},
 'superbody_with_cuts':{'directory':'superbody_with_cuts','cuts':[1,2]},
 'native_conditional_band':{'directory':'native_conditional_band','cuts':[0,1]},
 'rotated_asymmetric_slab':{'directory':'rotated_asymmetric_slab','cuts':[0,1]},
}
PGRID=(2,4,6);GGRID=(.5,.6,.7,.8,.9)

def read(p):return list(csv.DictReader(open(p)))
def write(p,rows,fields=None):
 p=HERE/p;p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'w',newline='') as f:
  if not rows:
   f.write('');return
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(p,x):
 p=HERE/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def eta_norm(r):return (np.array([float(r[f'eta{i}']) for i in (1,2,3)])-AFF)/SCALE
def hpkey(h):return json.dumps(h,sort_keys=True,separators=(',',':'))
def make_cloud(n=32768):
 lo=np.array([-7/6,-.5,-.5]);hi=np.array([.5,.5,.5]);sob=qmc.Sobol(3,scramble=False).random_base2(18);z=lo+(hi-lo)*sob;z=z[domain_contains(z)]
 assert len(z)>=n;return z[:n]

def classify_roles(r):
 s=(r.get('phases','')+';'+r.get('sources','')+';'+r.get('retained_validation_batches','')).lower()
 independent=(r.get('geometry_role')=='heldout' or 'coverage64' in s or 'coverage_screen' in s or 'independent_interpolation' in s or 'ep0082_intrusion' in s)
 transfer=('cross_transfer' in s or 'common_core' in s or 'val_neighbor' in s or 'cross_scenario_common_core' in s)
 return independent,transfer
def max_sep(X):return float(np.max(pdist(X))) if len(X)>=2 else 0.

def metrics(sid,scenario,m,rr,fitkeys,cloud,budget,hp):
 Z=np.array([eta_norm(r) for r in rr]);pos=np.array([r['B63']=='True' for r in rr]);fit=np.array([r['eta_key'] in fitkeys for r in rr]);ind=np.array([classify_roles(r)[0] for r in rr]);transfer=np.array([classify_roles(r)[1] for r in rr])
 if m is None:
  return dict(state_id=sid,scenario=scenario,budget=budget,hp=hpkey(hp),fit_success=False,fit_B63=int((fit&pos).sum()),fit_nonB63=int((fit&~pos).sum()),cached_false_inclusion='',heldout_false_inclusion='',full_recall=0,retained_recall=0,independent_recall=0,independent_B63=int((~fit&ind&pos).sum()),transfer_recall='',transfer_B63=int((~fit&transfer&pos).sum()),retained_diameter=0,retained_volume_fraction=0,retained_B63=0,retained_B63_separation=0,independent_retained_B63=0,training_region_usable=False,model='')
 full=contains(m,Z,False);ret=contains(m,Z,True);evalmask=~fit;indpos=evalmask&ind&pos
 if not indpos.any():indpos=evalmask&pos
 tr=evalmask&transfer&pos;extent=approx_extent(m,cloud,Z[ret&pos]);retP=Z[ret&pos]
 knownfalse=int((ret&~pos).sum());heldfalse=int((ret&~pos&evalmask).sum());indcapt=int((ret&indpos).sum())
 usable=(knownfalse==0 and extent['diameter']>=.15 and len(retP)>=2 and max_sep(retP)>=.10 and indcapt>=1)
 row=dict(state_id=sid,scenario=scenario,budget=budget,hp=hpkey(hp),fit_success=True,fit_B63=int((fit&pos).sum()),fit_nonB63=int((fit&~pos).sum()),cached_false_inclusion=knownfalse,heldout_false_inclusion=heldfalse,
  full_recall=float(full[evalmask&pos].mean()) if np.any(evalmask&pos) else 0,retained_recall=float(ret[evalmask&pos].mean()) if np.any(evalmask&pos) else 0,
  independent_recall=float(ret[indpos].mean()) if indpos.any() else 0,independent_B63=int(indpos.sum()),transfer_recall=float(ret[tr].mean()) if tr.any() else '',transfer_B63=int(tr.sum()),
  retained_diameter=extent['diameter'],retained_volume_fraction=extent['volume_fraction'],retained_cloud_points=extent['points'],retained_B63=len(retP),retained_B63_separation=max_sep(retP),independent_retained_B63=indcapt,
  training_region_usable=usable,model=json.dumps(m,separators=(',',':')))
 return row

def state_score(rows):
 if not rows:return (-1,)
 usable=sum(str(r['training_region_usable']) in ('True','true') or r['training_region_usable'] is True for r in rows)
 rates=[]
 for sc in ('ToyGiveWay_2A','DoubleBottleneck_4A'):
  q=[r for r in rows if r['scenario']==sc];rates.append(sum(bool(r['training_region_usable']) for r in q)/len(q) if q else 0)
 rec=np.median([float(r['independent_recall']) for r in rows]);trs=[float(r['transfer_recall']) for r in rows if r['transfer_recall']!=''];tr=np.median(trs) if trs else 0;dia=np.median([float(r['retained_diameter']) for r in rows]);false=sum(int(r['cached_false_inclusion']) for r in rows if r['cached_false_inclusion']!='')
 hp=json.loads(rows[0]['hp']);complexity=hp.get('max_cuts',0)
 return (min(rates),usable,-false,rec,tr,dia,-complexity,-float(hp['p']),-float(hp['gamma']))
def gate_summary(rows,family,fold_hps):
 usable=[r for r in rows if bool(r['training_region_usable'])];n=len(rows);by={}
 for sc in sorted(set(r['scenario'] for r in rows)):
  q=[r for r in rows if r['scenario']==sc];by[sc]=dict(states=len(q),usable=sum(bool(r['training_region_usable']) for r in q),usable_rate=float(np.mean([bool(r['training_region_usable']) for r in q])),median_independent_recall=float(np.median([float(r['independent_recall']) for r in q])))
 tr=[float(r['transfer_recall']) for r in rows if r['transfer_recall']!=''];ind=[float(r['independent_recall']) for r in rows];dia=[float(r['retained_diameter']) for r in rows]
 criteria={
  'state_coverage':len(usable)>=10,
  'both_scenarios':all(v['usable']>0 for v in by.values()),
  'scenario_balance':all(v['usable_rate']>=.75 for v in by.values()),
  'median_diameter':float(np.median([float(r['retained_diameter']) for r in usable]))>=.20 if usable else False,
  'diameter_fraction':float(np.mean(np.array(dia)>=.15))>=.80,
  'independent_recall':float(np.median(ind))>=.25 and all(v['median_independent_recall']>=.20 for v in by.values()),
  'neighbor_transfer':bool(tr) and float(np.median(tr))>=.30,
  'cached_reliability_on_usable':all(int(r['cached_false_inclusion'])==0 for r in usable),
 }
 return dict(family=family,outer_cv_states=n,usable_states=len(usable),scenario_metrics=by,median_retained_diameter=float(np.median([float(r['retained_diameter']) for r in usable])) if usable else 0,
  diameter_ge_015_fraction=float(np.mean(np.array(dia)>=.15)),median_independent_B63_recall=float(np.median(ind)),median_transfer_coverage=float(np.median(tr)) if tr else None,
  total_cached_false_inclusions=sum(int(r['cached_false_inclusion']) for r in rows if r['cached_false_inclusion']!=''),fold_selected_hyperparameters=fold_hps,criteria=criteria,cached_screen_pass=all(criteria.values()),fresh_validation='NOT_RUN')

def run_family(family,cloud,inv,subs,folds):
 cfg=FAMILIES[family];hps=[]
 for p,g,k in itertools.product(PGRID,GGRID,cfg['cuts']):hps.append(dict(p=p,gamma=g,max_cuts=k,safety=.9))
 states=sorted({r['state_id'] for r in inv});by=defaultdict(list)
 for r in inv:by[r['state_id']].append(r)
 all_metrics={};all_models={}
 for hp in hps:
  hk=hpkey(hp)
  for sid in states:
   keys=subs[sid]['32'];lookup={r['eta_key']:r for r in by[sid]};fitrows=[lookup[k] for k in keys];m=fit(family,fitrows,**hp)
   row=metrics(sid,by[sid][0]['scenario'],m,by[sid],set(keys),cloud,32,hp);all_metrics[sid,hk]=row;all_models[sid,hk]=m
 fold_hps=[];outer=[]
 for f in folds:
  dev=f['development'];best=max(hps,key=lambda hp:state_score([all_metrics[s,hpkey(hp)] for s in dev]));fold_hps.append(dict(fold=f['fold'],hyperparameters=best,development_score=state_score([all_metrics[s,hpkey(best)] for s in dev])))
  for sid in f['heldout']:
   r=dict(all_metrics[sid,hpkey(best)],fold=f['fold'],outer_heldout=True);outer.append(r)
 # Final global hp is chosen by fold vote, then all-state score; this is frozen before fresh rollout.
 votes=Counter(hpkey(x['hyperparameters']) for x in fold_hps);top=max(votes.values());choices=[json.loads(k) for k,v in votes.items() if v==top];final_hp=max(choices,key=lambda hp:state_score([all_metrics[s,hpkey(hp)] for s in states]))
 final_rows=[all_metrics[s,hpkey(final_hp)] for s in states];gate=gate_summary(outer,family,fold_hps);gate['final_global_hyperparameters']=final_hp;gate['final_global_cached_metrics']=gate_summary(final_rows,family,fold_hps)
 d=cfg['directory'];write(f'{d}/cv_results.csv',outer);write(f'{d}/retained_metrics.csv',final_rows);write(f'{d}/fitted_parameters.csv',[dict(state_id=s,scenario=by[s][0]['scenario'],hyperparameters=hpkey(final_hp),model=json.dumps(all_models[s,hpkey(final_hp)],separators=(',',':'))) for s in states])
 dump(f'{d}/gate.json',gate)
 return gate,final_hp

def budget_performance(family,hp,cloud,inv,subs):
 by=defaultdict(list)
 for r in inv:by[r['state_id']].append(r)
 out=[]
 for b in (16,24,32):
  for sid,rr in sorted(by.items()):
   lookup={r['eta_key']:r for r in rr};keys=subs[sid][str(b)];m=fit(family,[lookup[k] for k in keys],**hp);out.append(metrics(sid,rr[0]['scenario'],m,rr,set(keys),cloud,b,hp))
 return out

def main():
 if (HERE/'family_comparison.json').exists():
  assert '--audited-oracle-v1' in sys.argv and (HERE/'oracle_revision.json').exists(), 'Frozen CV already completed; preserve it before an explicitly audited protocol revision'
 inv=[r for r in read(HERE/'exact_q64_inventory.csv') if r['state_id'] in json.load(open(HERE/'oracle_budget_subsamples.json'))['states'] and r['in_E_bridge']=='True']
 subs=json.load(open(HERE/'oracle_budget_subsamples.json'))['states'];folds=json.load(open(HERE/'cv_folds.json'))['folds'];cloud=make_cloud()
 gates=[];selected={};t=time.time()
 for family in FAMILIES:
  gate,hp=run_family(family,cloud,inv,subs,folds);gates.append(gate);selected[family]=hp
  perf=budget_performance(family,hp,cloud,inv,subs);write(f"{FAMILIES[family]['directory']}/oracle_budget_performance.csv",perf)
 dump('family_comparison.json',dict(runtime_seconds=time.time()-t,gates=gates,selected_global_hyperparameters=selected))
 print(json.dumps([{ 'family':g['family'],'usable':g['usable_states'],'pass':g['cached_screen_pass'],'recall':g['median_independent_B63_recall'],'diameter':g['median_retained_diameter']} for g in gates]))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Evaluate evidence-motivated synthesized shared families under unchanged gates."""
import csv,json,itertools,time,sys
from pathlib import Path
from collections import defaultdict,Counter
from shared_families import fit
from cross_validate import read,write,dump,make_cloud,metrics,state_score,gate_summary,hpkey

HERE=Path(__file__).resolve().parent
SYNTH={
 'affine_capsule_with_cuts':{'cuts':[0,1],'reason':'One connected analytic body aligned to two robust witnesses; tests the 88/96 success-success interpolation evidence while allowing one sharp exclusion.'},
 'two_superbody_union':{'cuts':[0],'reason':'Two conservative analytic components test whether low transfer coverage reflects two separated robust modes rather than a need for dense geometry.'},
 'domain_minus_boundary_caps':{'cuts':[2,3,4],'p':[2],'reason':'A shared E_bridge base minus a small number of boundary-open spherical exclusions directly tests the broad-core plus state-dependent intrusion hypothesis.'},
}
def run(family,cloud,inv,subs,folds):
 hps=[dict(p=p,gamma=g,max_cuts=k,safety=.9) for p,g,k in itertools.product(SYNTH[family].get('p',(2,4,6)),(.5,.6,.7,.8,.9),SYNTH[family]['cuts'])]
 by=defaultdict(list)
 for r in inv:by[r['state_id']].append(r)
 states=sorted(by);allm={};models={}
 for hp in hps:
  hk=hpkey(hp)
  for sid in states:
   lookup={r['eta_key']:r for r in by[sid]};keys=subs[sid]['32'];m=fit(family,[lookup[k] for k in keys],**hp);allm[sid,hk]=metrics(sid,by[sid][0]['scenario'],m,by[sid],set(keys),cloud,32,hp);models[sid,hk]=m
 outer=[];foldhp=[]
 for f in folds:
  best=max(hps,key=lambda hp:state_score([allm[s,hpkey(hp)] for s in f['development']]));foldhp.append(dict(fold=f['fold'],hyperparameters=best,development_score=state_score([allm[s,hpkey(best)] for s in f['development']])))
  for sid in f['heldout']:outer.append(dict(allm[sid,hpkey(best)],fold=f['fold'],outer_heldout=True))
 votes=Counter(hpkey(x['hyperparameters']) for x in foldhp);top=max(votes.values());choices=[json.loads(k) for k,v in votes.items() if v==top];final=max(choices,key=lambda hp:state_score([allm[s,hpkey(hp)] for s in states]));finalrows=[allm[s,hpkey(final)] for s in states]
 gate=gate_summary(outer,family,foldhp);gate.update(final_global_hyperparameters=final,final_global_cached_metrics=gate_summary(finalrows,family,foldhp),synthesis_reason=SYNTH[family]['reason'])
 d=f'synthesized_families/{family}';write(f'{d}/cv_results.csv',outer);write(f'{d}/retained_metrics.csv',finalrows);write(f'{d}/fitted_parameters.csv',[dict(state_id=s,scenario=by[s][0]['scenario'],hyperparameters=hpkey(final),model=json.dumps(models[s,hpkey(final)],separators=(',',':'))) for s in states]);dump(f'{d}/gate.json',gate)
 perf=[]
 for b in (16,24,32):
  for sid in states:
   lookup={r['eta_key']:r for r in by[sid]};keys=subs[sid][str(b)];m=fit(family,[lookup[k] for k in keys],**final);perf.append(metrics(sid,by[sid][0]['scenario'],m,by[sid],set(keys),cloud,b,final))
 write(f'{d}/oracle_budget_performance.csv',perf)
 return gate
def main():
 out=HERE/'synthesis_comparison.json'
 if out.exists():raise FileExistsError('Synthesized-family CV already frozen')
 subs=json.load(open(HERE/'oracle_budget_subsamples.json'))['states'];inv=[r for r in read(HERE/'exact_q64_inventory.csv') if r['state_id'] in subs and r['in_E_bridge']=='True'];folds=json.load(open(HERE/'cv_folds.json'))['folds'];cloud=make_cloud();t=time.time();g=[]
 for family in SYNTH:g.append(run(family,cloud,inv,subs,folds))
 dump('synthesis_comparison.json',dict(runtime_seconds=time.time()-t,gates=g))
 print(json.dumps([{'family':x['family'],'usable':x['usable_states'],'recall':x['median_independent_B63_recall'],'transfer':x['median_transfer_coverage'],'pass':x['cached_screen_pass']} for x in g]))
if __name__=='__main__':main()

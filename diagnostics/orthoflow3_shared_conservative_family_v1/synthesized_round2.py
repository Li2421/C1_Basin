#!/usr/bin/env python3
import json,time,sys,shutil
from pathlib import Path
from synthesized_cv import run
from cross_validate import read,dump,make_cloud
HERE=Path(__file__).resolve().parent
def main():
 out=HERE/'synthesis_round2_comparison.json'
 if out.exists():
  assert '--erosion-subset-fix' in sys.argv
  shutil.copy2(out,HERE/'synthesis_round2_comparison_pre_erosion_fix.json')
  d=HERE/'synthesized_families/domain_minus_boundary_caps'
  for n in ('cv_results.csv','retained_metrics.csv','fitted_parameters.csv','gate.json','oracle_budget_performance.csv'):
   p=d/n
   if p.exists():shutil.copy2(p,p.with_name(p.stem+'_pre_erosion_fix'+p.suffix))
 subs=json.load(open(HERE/'oracle_budget_subsamples.json'))['states'];inv=[r for r in read(HERE/'exact_q64_inventory.csv') if r['state_id'] in subs and r['in_E_bridge']=='True'];folds=json.load(open(HERE/'cv_folds.json'))['folds'];t=time.time();g=run('domain_minus_boundary_caps',make_cloud(),inv,subs,folds);dump('synthesis_round2_comparison.json',dict(runtime_seconds=time.time()-t,gate=g));print(json.dumps({'family':g['family'],'usable':g['usable_states'],'recall':g['median_independent_B63_recall'],'transfer':g['median_transfer_coverage'],'pass':g['cached_screen_pass']}))
if __name__=='__main__':main()

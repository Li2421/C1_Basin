#!/usr/bin/env python3
"""Aggregate frozen TEST and perturbation outcomes before fresh-WIDE."""
from __future__ import annotations
import csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_basin_margin_learning_v1'
def read_csv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def read_jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write_csv(p,rr,fields=None):
 if fields is None:fields=list(rr[0])
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def collect(name):
 out=[]
 for p in sorted((HERE/'raw'/name).glob('shard*.jsonl')):out+=read_jsonl(p)
 return out
def main():
 geo=read_csv(HERE/'test_geometric_predictions.csv');roll=collect('test64');by=defaultdict(list)
 for x in roll:by[(x['controller'],x['state_id'])].append(x)
 states=[]
 for (ctl,sid),rr in sorted(by.items()):
  suc=sum(x['success'] for x in rr);out=Counter(x['outcome'] for x in rr);j=[x['J_def'] for x in rr if x['success']];steps=[x['continuation_steps'] for x in rr]
  g=next(x for x in geo if x['model']==ctl and x['state_id']==sid)
  states.append({'state_id':sid,'controller':ctl,'successes':suc,'trials':len(rr),'Q64':suc/64,'B63':suc>=63,'deadlock':out['safe_deadlock'],'timeout':out['timeout'],'collision':out['collision'],'other_numerical':out['other_numerical'],'successful_J_def_mean':float(np.mean(j)) if j else None,'successful_J_def_median':float(np.median(j)) if j else None,'episode_length_mean':float(np.mean(steps)),'rho':g['rho'],'normalized_center_distance':g['normalized_center_distance'],'inside_ball':g['inside_ball'],'inside_retained':g['inside_retained'],'inside_Ebridge':g['inside_Ebridge'],'eta1':g['eta1'],'eta2':g['eta2'],'eta3':g['eta3']})
 write_csv(HERE/'test_64seed_closedloop.csv',states)
 pair=[]
 for sid in sorted({x['state_id'] for x in states}):
  c=next(x for x in states if x['state_id']==sid and x['controller']=='g_center');m=next(x for x in states if x['state_id']==sid and x['controller']=='g_margin')
  trans='BOTH_B63' if c['B63'] and m['B63'] else 'CENTER_FAIL_TO_MARGIN_B63' if not c['B63'] and m['B63'] else 'CENTER_B63_TO_MARGIN_FAIL' if c['B63'] and not m['B63'] else 'BOTH_FAIL'
  pair.append({'state_id':sid,'center_Q64':c['Q64'],'margin_Q64':m['Q64'],'center_B63':c['B63'],'margin_B63':m['B63'],'transition':trans,'center_rho':c['rho'],'margin_rho':m['rho']})
 write_csv(HERE/'test_pairwise_center_vs_margin.csv',pair)
 # Frozen rho bins.
 bins=[('rho<=0.5',lambda z:z<=.5),('0.5<rho<=0.8',lambda z:.5<z<=.8),('0.8<rho<=1.0',lambda z:.8<z<=1.),('rho>1.0',lambda z:z>1.)];er=[]
 for name,fn in bins:
  selected=[x for x in states if x['controller'] in ('g_center','g_margin') and fn(float(x['rho']))]
  er.append({'rho_bin':name,'predictions':len(selected),'mean_Q64':float(np.mean([x['Q64'] for x in selected])) if selected else None,'B63_count':sum(x['B63'] for x in selected),'B63_rate':float(np.mean([x['B63'] for x in selected])) if selected else None,'mean_raw_center_L2':float(np.mean([float(x['normalized_center_distance']) for x in selected])) if selected else None})
 write_csv(HERE/'error_to_margin.csv',er)
 # Perturbation aggregation; canceled/resumed shard records are already deduplicated by exact plan partition.
 pert=collect('perturb');points=read_csv(HERE/'perturbation_points.csv');pby=defaultdict(list)
 for x in pert:pby[x['point_id']].append(x)
 pr=[]
 for p in points:
  rr=pby.get(p['point_id'],[]);pr.append({**p,'trials':len(rr),'successes':sum(x['success'] for x in rr),'success_fraction':sum(x['success'] for x in rr)/len(rr) if rr else None,'all_8_success':len(rr)==8 and all(x['success'] for x in rr),'outcomes':json.dumps(dict(Counter(x['outcome'] for x in rr)),sort_keys=True),'not_evaluated_reason':'' if rr else 'outside_Ebridge'})
 write_csv(HERE/'perturbation_robustness.csv',pr)
 # Robustness/deformation, paired only where all three are B63 (all states in this audit).
 comp=[]
 for sid in sorted({x['state_id'] for x in states}):
  z={x['controller']:x for x in states if x['state_id']==sid};l=z['g_lowj'];c=z['g_center'];m=z['g_margin'];ok=all(z[k]['B63'] for k in z)
  comp.append({'state_id':sid,'all_B63':ok,'J_lowj':l['successful_J_def_mean'],'J_center':c['successful_J_def_mean'],'J_margin':m['successful_J_def_mean'],'J_center_over_lowj':c['successful_J_def_mean']/l['successful_J_def_mean'] if ok and l['successful_J_def_mean']>0 else None,'J_margin_over_lowj':m['successful_J_def_mean']/l['successful_J_def_mean'] if ok and l['successful_J_def_mean']>0 else None,'J_margin_over_center':m['successful_J_def_mean']/c['successful_J_def_mean'] if ok and c['successful_J_def_mean']>0 else None,'length_lowj':l['episode_length_mean'],'length_center':c['episode_length_mean'],'length_margin':m['episode_length_mean']})
 write_csv(HERE/'robustness_vs_deformation.csv',comp)
 summary={}
 for ctl in ('g_center','g_margin','g_lowj'):
  q=[x for x in states if x['controller']==ctl];summary[ctl]={'B63_states':sum(x['B63'] for x in q),'states':len(q),'mean_Q64':float(np.mean([x['Q64'] for x in q])),'total_successes':sum(x['successes'] for x in q),'trials':sum(x['trials'] for x in q),'deadlock':sum(x['deadlock'] for x in q),'timeout':sum(x['timeout'] for x in q),'collision':sum(x['collision'] for x in q),'successful_J_def_mean':float(np.mean([x['successful_J_def_mean'] for x in q])),'successful_J_def_median_of_state_medians':float(np.median([x['successful_J_def_median'] for x in q])),'rho_mean':float(np.mean([float(x['rho']) for x in q])) if ctl!='g_lowj' else None,'rho_median':float(np.median([float(x['rho']) for x in q])) if ctl!='g_lowj' else None,'inside_ball':sum(x['inside_ball']=='True' for x in q) if ctl!='g_lowj' else None,'inside_retained':sum(x['inside_retained']=='True' for x in q) if ctl!='g_lowj' else None,'inside_Ebridge':sum(x['inside_Ebridge']=='True' for x in q)}
 trans=Counter(x['transition'] for x in pair);valid=[x for x in pr if x['trials']];pret={}
 for ctl in ('g_center','g_margin'):
  pret[ctl]={}
  for mag in (.25,.5):
   z=[x for x in valid if x['model']==ctl and float(x['magnitude_r'])==mag];pret[ctl][str(mag)]={'valid_points':len(z),'successes':sum(int(x['successes']) for x in z),'trials':sum(int(x['trials']) for x in z),'success_fraction':sum(int(x['successes']) for x in z)/sum(int(x['trials']) for x in z) if z else None,'all8_points':sum(bool(x['all_8_success']) for x in z)}
 dump(HERE/'intermediate_test_summary.json',{'controllers':summary,'paired_transitions':dict(trans),'error_to_margin':er,'perturbation':pret,'all_Q64_constant_one':all(x['Q64']==1 for x in states),'rho_predictiveness_interpretation':'not identifiable: every nominal TEST prediction was Q64=1/B63, including predictions outside the verified conservative balls'})
 print(json.dumps(load(HERE/'intermediate_test_summary.json') if False else {'controllers':summary,'paired':dict(trans),'perturbation':pret},indent=2))
if __name__=='__main__':main()

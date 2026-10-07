#!/usr/bin/env python3
"""Select scenario-fixed eta values from frozen train/development evidence only."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json, sqlite3
import numpy as np
import pyarrow.parquet as pq
from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import HardProjectionConfig

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
LOW=np.asarray([.5,-.5,0.]);HIGH=np.asarray([1.25,.5,.75]);CENTER=(LOW+HIGH)/2;WIDTH=HIGH-LOW

def main():
 states=pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v1/states.parquet').to_pylist()
 labels=pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v1/eta_labels.parquet').to_pylist()
 con=sqlite3.connect(ROOT/'shared_rollout_db/rollout.sqlite')
 result={}
 for sc in ('double_bottleneck','four_way_intersection','ring_exchange'):
  ss=[x for x in states if x['scenario']==sc and x['split'] in ('train','validation')]
  ids={x['state_uid'] for x in ss};zero={x['state_uid'] for x in ss if x['zero_sufficient']}
  by=defaultdict(list)
  for x in labels:
   if x['scenario']!=sc or x['state_uid'] not in ids:continue
   eta=tuple(json.loads(x['eta_raw']))
   if np.all(np.asarray(eta)>=LOW-1e-10) and np.all(np.asarray(eta)<=HIGH+1e-10):by[eta].append(x)
  candidates=[]
  for eta,rr in by.items():
   if {x['state_uid'] for x in rr}!=ids:continue
   robust={x['state_uid'] for x in rr if x['robust_15of16'] is True}
   preservation=len(robust & zero)
   j=[]
   for x in rr:
    vals=con.execute('SELECT j_def FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND numerical_failure=0 AND j_def IS NOT NULL',(x['state_uid'],x['eta_uid'],x['controller_uid'])).fetchall()
    j += [float(v[0]) for v in vals]
   norm=float(np.linalg.norm((np.asarray(eta)-CENTER)/WIDTH))
   candidates.append({'eta':list(eta),'eta_index':sorted(by).index(eta),'state_coverage':len(robust),
                      'states':len(ids),'success_control_preservation':preservation,'success_controls':len(zero),
                      'mean_executed_deformation':float(np.mean(j)) if j else None,
                      'normalized_eta_norm':norm,'evaluated_rollouts_with_deformation':len(j)})
  # New-benchmark source records predate J_def persistence.  Compute the
  # required executed instantaneous deformation mechanically on train/dev
  # states, but only for candidates tied on the first two selection criteria.
  best_cov=max(x['state_coverage'] for x in candidates);best_pres=max(x['success_control_preservation'] for x in candidates if x['state_coverage']==best_cov)
  tied=[x for x in candidates if x['state_coverage']==best_cov and x['success_control_preservation']==best_pres]
  if sc in ('four_way_intersection','ring_exchange') and any(x['mean_executed_deformation'] is None for x in tied):
   if sc=='four_way_intersection':
    from four_way_intersection.environment import Config,FourWayIntersectionEnv
    cfg=Config(**json.loads((ROOT/'diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_dataset/manifest.json').read_text())['scenario_config'])
    make=lambda:FourWayIntersectionEnv(cfg);snapshot=lambda env:env.snapshot()
   else:
    from ring_exchange.environment import LocalFrameConfig,RingExchangeEnv
    from ring_exchange.safety import polygonal_obstacle_snapshot
    cfg=LocalFrameConfig(**json.loads((ROOT/'diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json').read_text())['scenario_config'])
    make=lambda:RingExchangeEnv(cfg);snapshot=lambda env:polygonal_obstacle_snapshot(env,sides=48)
   projector=CertifiedHardSafetyFilter(HardProjectionConfig());basis=get_basis_family('orthoflow3')
   accum={tuple(x['eta']):[] for x in tied}
   for state in ss:
    physical=json.loads(state['structured_state']);cond=json.loads(state['conditioning']);env=make()
    if sc=='four_way_intersection':env.reset(np.asarray(physical['positions']),np.asarray(physical['velocities']))
    else:env.reset(np.asarray(physical['positions']),velocities=np.asarray(physical['velocities']),goals=np.asarray(physical['goals']))
    flow=np.asarray(cond['reference_current_raw_flow'],float);safe=np.asarray(projector(snapshot(env),flow).velocity,float)
    fields=basis.compute(env.positions,env.goals,safe,cfg.max_speed)
    for x in tied:
     executed=np.asarray(projector(snapshot(env),safe+fields.correction(np.asarray(x['eta']))).velocity,float)
     accum[tuple(x['eta'])].append(float(np.sum((executed-safe)**2)))
   for x in tied:
    vals=accum[tuple(x['eta'])];x['mean_executed_deformation']=float(np.mean(vals));x['evaluated_rollouts_with_deformation']=len(vals)
  # Missing j_def sorts behind measured deformation; no outcome/test evidence enters.
  ranked=sorted(candidates,key=lambda x:(-x['state_coverage'],-x['success_control_preservation'],
       float('inf') if x['mean_executed_deformation'] is None else x['mean_executed_deformation'],
       x['normalized_eta_norm'],x['eta_index']))
  result[sc]={'selected':ranked[0],'candidate_count':len(ranked),'top10':ranked[:10],
              'selection_population':'orthoflow3_basin_dataset_v1 train+validation parents only'}
 con.close()
 (OUT/'fair_fixed_eta.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({k:v['selected'] for k,v in result.items()},indent=2))
if __name__=='__main__':main()

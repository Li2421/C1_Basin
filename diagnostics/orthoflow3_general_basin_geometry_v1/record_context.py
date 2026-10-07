#!/usr/bin/env python3
from audit import *
def main():
 names=['orthoflow3_t0_basin_completion_v1','orthoflow3_t0_multiball_basin_learning_v1','orthoflow3_existing_b63_geometry_v1','orthoflow3_b63_manifold_representation_v1',
 'orthoflow3_t0_basin_shape_v1','orthoflow3_t0_pact_training_readiness_v1','orthoflow3_true_t0_point_learning_v1','orthoflow3_t0_eta_continuity_cross_transfer_v1','orthoflow3_analytic_basin_margin_learning_v1']
 rows=[]
 for name in names:
  root=D/name
  for p in sorted(root.glob('*.json')):
   if 'manifest' not in p.name and 'decision' not in p.name and p.name!='runtime_statistics.json':continue
   data=json.load(open(p));rows.append(dict(experiment=name,path=str(p),sha256=sha(p),top_level_fields=';'.join(data.keys()) if isinstance(data,dict) else 'list'))
  for p in sorted(root.glob('*report.md')):rows.append(dict(experiment=name,path=str(p),sha256=sha(p),top_level_fields='scientific_report_read'))
 write('prior_manifest_index.csv',rows)
 write('excluded_evidence_sources.csv',[
  dict(source='historical intermediate44 verified balls',reason='phase is intermediate, not true_t0; not pooled in current Basin claims',role='prior geometry context only'),
  dict(source='Q-v2 / Direct-eta / Q-guided historical root2026092702',reason='future RNG root and/or queried h differs from fixed40 namespace2026092811; do not merge by physical-state resemblance',role='incompatible unless full tuple matched in accepted historical audit'),
  dict(source='Double-Bottleneck historical seed16/local8',reason='initial Flow xi0 changes across seed; not Q64 conditioned on one h0',role='candidate discovery only'),
  dict(source='8/8,16/16 screening records',reason='not64 distinct matched future seeds',role='separate lower_seed_inventory; never hard B63'),
  dict(source='historical invalid feature replay directories',reason='xi0/h conditioning mismatch',role='excluded, never revived by geometry fit')])
 dump('search_history.json',{'status':'RUNNING','network_training':False,'stop_condition_satisfied':False,'rounds':[
  {'id':'initial_cached','exact_state_eta':1131,'adequate_states':8,'families':['common_core_minus_exclusions','conditional_band_with_cuts','asymmetric_slab_with_notches','star_convex_with_exclusions','semialgebraic'],'result':'none passes cached screen; not family impossibility'},
  {'id':'common_core','job_id':877,'frozen_new_Q64_max':222,'purpose':'test9 robust target modes and RAP witness across40 states; no degeneracy threshold'},
  {'id':'geometry_round1','status':'manifest frozen, not launched','purpose':'sampled star segments, exact conditional lines, failure-to-boundary paths'},
  {'id':'db_preflight','job_id':883,'dependency':877,'purpose':'verify fixed-current Flow and cost in genuinely compatible joint4A scenario'}]})
 print('Indexed',len(rows),'prior reports/manifests; excluded incompatible evidence separately')
if __name__=='__main__':main()

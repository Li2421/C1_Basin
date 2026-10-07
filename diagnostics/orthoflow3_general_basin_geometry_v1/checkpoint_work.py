#!/usr/bin/env python3
"""Compact authoritative working memory; preserve scientific decisions separately."""
from audit import *
import subprocess
def main():
 previous=json.load(open(HERE/'working_state.json')) if (HERE/'working_state.json').exists() else {}
 jobs=subprocess.run(['squeue','-h','-o','%i|%u|%T|%j'],capture_output=True,text=True,check=True).stdout.strip().splitlines()
 prior=read(HERE/'prior_manifest_index.csv');cache=json.load(open(HERE/'cache_provenance.json'))
 evidence=json.load(open(HERE/'evidence_index.json')) if (HERE/'evidence_index.json').exists() else {}
 evidence.update({f'P{i:03d}':r for i,r in enumerate(prior)})
 evidence.update({f'C{i:03d}':r for i,r in enumerate(cache)})
 artifacts=['conditioning_semantics.json','inventory_summary.json','family_gates_initial.json','candidate_protocol.md','protocol.md','validation_protocol.md','offline_unit_tests.json',
 'scenario_compatibility.csv','double_bottleneck_state_panel.json','double_bottleneck_cost_estimate.json','candidate_falsifications_initial.csv','search_history.json',
 'execution_anomaly_resolution.json','cached_link_protocol.json']
 evidence.update({f'A{i:03d}':{'path':str(HERE/name),'sha256':sha(HERE/name)} for i,name in enumerate(artifacts) if (HERE/name).exists()})
 dump('evidence_index.json',evidence)
 gates=json.load(open(HERE/'family_gates_initial.json'));hs=[]
 forms={'common_core_minus_exclusions':'E_bridge AND ||z-c_j||>=r_j, j<=4 boundary-open caps',
 'conditional_band_with_cuts':'elliptic x-support AND affine lower(x)<=z3<=upper(x), <=2 caps',
 'asymmetric_slab_with_notches':'PCA asymmetric box AND <=2 cap exclusions',
 'star_convex_with_exclusions':'medoid-centered radial ellipsoid AND <=2 caps',
 'semialgebraic':'one quadratic polynomial g(z)>=0 AND E_bridge'}
 for i,g in enumerate(gates):
  failed=[]
  if g['cached_retained_false_inclusions']:failed.append('retained_precision')
  if g['median_full_recall']<.7:failed.append('full_recall')
  if g['median_retained_recall']<.4:failed.append('retained_recall')
  if g['fraction_recall_ge_half']<.8:failed.append('80pct_state_recall')
  if g['mean_state_transfer_recall']<.7:failed.append('transfer_coverage')
  hs.append(dict(hypothesis_id='H'+str(i+1),mathematical_form=forms[g['family']],family=g['family'],current_status='INITIAL_FIT_REJECTED_CLASS_NOT_EXHAUSTED',
   key_support={'full_recall':g['median_full_recall']},key_failure={'failed_gates':failed,'retained_false':g['cached_retained_false_inclusions'],'retained_recall':g['median_retained_recall'],'transfer_coverage':g['mean_state_transfer_recall']},
   unresolved_property='unmeasured boundary intrusions, conditional interval changes, prospective retained safety',next_discriminating_test='geometry_round1 and common_core; refit only with newly informative evidence'))
 hs.append(dict(hypothesis_id='H6',mathematical_form='finite common eta panel across40 fixed h0',current_status='EXACT_Q64_ACQUISITION_RUNNING',
  key_support='historical target cross-transfer',key_failure=None,unresolved_property='coverage of9 modes plusRAP witness across all40',next_discriminating_test='common_core job877 completion'))
 if not (HERE/'hypothesis_status.json').exists():dump('hypothesis_status.json',hs)
 elif list(HERE.glob('family_gates_*.json')):
  latest=max(HERE.glob('family_gates_*.json'),key=lambda p:p.stat().st_mtime)
  existing=json.load(open(HERE/'hypothesis_status.json'));newg={g['family']:g for g in json.load(open(latest))}
  for h in existing:
   if latest.name=='family_gates_after_round1.json':continue # explicitly invalidated provisional incomplete-inventory fit
   if h.get('family') in newg and h.get('latest_gate_file')!=latest.name:
    g=newg[h['family']];h['current_status']='CACHED_REFIT_REQUIRES_GATE_INTERPRETATION';h['latest_gate_file']=latest.name
    h['key_support']={'full_recall':g['median_full_recall']};h['key_failure']={'retained_false':g['cached_retained_false_inclusions'],'retained_recall':g['median_retained_recall'],'transfer_coverage':g.get('pooled_transfer_recall',g.get('mean_state_transfer_recall'))}
  dump('hypothesis_status.json',existing)
 ledger=[]
 for root in sorted((HERE/'targeted_probe_rounds').iterdir()):
  p=root/'cost_estimate.json'
  if not p.exists():continue
  e=json.load(open(p));outputs=list((HERE/'raw'/root.name).glob('shard*_runtime.json'))
  if root.name.startswith('db_'):
   dbout=list((HERE/'db_raw').glob(root.name+'_*.jsonl'));dbcount=sum(sum(1 for _ in open(p)) for p in dbout)
   observed=dbcount;completed=len(dbout) if dbcount==e.get('new_continuations') else 0
  else:observed=sum(json.load(open(p))['new_continuations'] for p in outputs);completed=len(outputs)
  if not completed and not root.name.startswith('db_'):
   live_raw=list((HERE/'runs'/root.name).glob('shard*/raw/pilot_rollouts.jsonl'))
   observed=sum(sum(1 for _ in open(p)) for p in live_raw)
  quarantined=1 if root.name=='db_geometry1' and (HERE/'execution_anomaly_resolution.json').exists() else 0
  ledger.append(dict(experiment_id=root.name,decision_to_distinguish={'common_core':'Is high common overlap a real robust core?',
    'geometry_round1':'Star segments, native conditional intervals, failure-to-boundary corridors',
    'db_geometry1':'Does the same geometry class survive fixed-h0 joint4A control?'}.get(root.name,'see frozen manifest'),
   manifest=str(root/'manifest.csv'),expected_new_continuations=e.get('new_continuations'),completed_shards=completed,
   observed_new_continuations=observed,quarantined_execution_errors=quarantined,status=('COMPLETE_WITH_QUARANTINE' if quarantined else 'COMPLETE') if completed==e.get('shards') else 'PENDING_OR_RUNNING',
   expected_wall_seconds=e.get('estimated_seconds',e.get('estimated_wall_seconds')),evidence_outputs=str(HERE/'raw'/root.name)))
 db=list((HERE/'db_raw').glob('db_preflight_*.jsonl'));pre=[json.loads(line) for p in db for line in open(p)]
 ledger.append(dict(experiment_id='db_preflight',decision_to_distinguish='Exact fixed-current Flow compatibility and empirical cost',manifest=str(HERE/'double_bottleneck_state_panel.json'),expected_new_continuations=8,
  completed_shards=len(db),observed_new_continuations=len(pre),status='COMPLETE_VALID' if len(pre)==8 and all(r['scientific_outcome_valid'] for r in pre) else 'INCOMPLETE',expected_wall_seconds=20,evidence_outputs=str(HERE/'db_raw')))
 write('experiment_ledger.csv',ledger)
 state={'updated_local':time.strftime('%Y-%m-%d %H:%M:%S %Z'),'goal_status':'ACTIVE','training_prohibited':True,'scientific_gates_unchanged':True,
 'completed_stages':['prior manifest/report inspection','exact cache inventory and conflict audit','scenario implementation compatibility','initial5 geometric fits','offline unit tests','observed-only SVGs'],
 'evidence_summary':json.load(open(HERE/'inventory_summary.json')),'live_job_observation':jobs,'hypothesis_ledger':'hypothesis_status.json','experiment_ledger':'experiment_ledger.csv',
 'unresolved_questions':['common core all40 matrix','validated success paths and failure corridors','native conditional intervals','DB conditioned Q64 geometry','fresh retained precision','shared-family gates','saturation not reached'],
 'next_action':'Complete common-core and DB cost preflight; launch registered geometry round and DB targeted geometry within total CPU/shard policy.',
 'do_not_reread':'Use evidence_index references; original files only for exact numerical/semantic conflicts.',
 'anomalies':[{'issue':'old summaries compared outcome==deadlock instead of safe_deadlock','resolution':'count canonical raw failure strings'},
  {'issue':'CPU-only DB process CUDA plugin discovery warning','resolution':'explicit CPU backend; all8 preflight scientific_outcome_valid; no GPU job submitted'}]}
 if len(pre)==8:state['completed_stages'].append('DB fixed-current preflight8 valid continuations')
 state['completed_stages']=list(dict.fromkeys(previous.get('completed_stages',[])+state['completed_stages']))
 if 'common_core40x10 exact matrix' in state['completed_stages']:state['unresolved_questions']=[q for q in state['unresolved_questions'] if q!='common core all40 matrix']
 state['next_action']=previous.get('next_action',state['next_action'])
 if any(j.startswith(('888_','889_')) for j in jobs):state['next_action']='Monitor live jobs888/889; once complete aggregate exactQ64, evaluate registered geometry properties and update family hypotheses. Do not relaunch.'
 if any(j.startswith('900_') for j in jobs):state['next_action']='Wait for job900 shard5 remainder; exclude approved numerical failure, aggregate147/148 DB Q64, then refit after_round1_complete. No replacement seed or projection change.'
 if any(j.startswith(('901_','902_')) for j in jobs):state['next_action']='Monitor901 prospective retained validation and902 DBcross-scenario geometry. Preserve frozen validation parameters; do not rerun completed tuples.'
 if any(j.startswith('913_') for j in jobs):state['next_action']='Monitor913 property probes and902 DBgeometry. Semialgebraic prospective validation rejected21/42; diagnose extent vs intrusion before new family synthesis.'
 if any(j.startswith('919_') for j in jobs):state['next_action']='Monitor919 pocket scans and902 DBgeometry. Then aggregate and fit bounded_family.py under frozen synthesized protocol. No accepted family.'
 if any(j.startswith(('925_','926_')) for j in jobs):state['next_action']='Monitor925 bounded retained validation and926 heldout DBinterpolation. Audit then evaluate frozen validation; no refit/reuse of tested parameters after failure.'
 if any(j.startswith(('937_','938_')) for j in jobs):state['next_action']='Monitor937/938 fresh72point12state validation; audit then evaluate both frozen manifests. Same family/erosion, prior rejected validation now disclosed development. No accepted family.'
 if any(j.startswith(('949_','950_')) for j in jobs):state['next_action']='Monitor949/950 fresh72point F2 validation. F1 rejected22/72. Audit and evaluate frozen F2 manifests; then property diagnosis or justified saturation.'
 state['anomalies']=previous.get('anomalies',state['anomalies'])
 state['actual_stage_jobs']={'common_core':877,'db_preflight':883,'conditioning_no_rollout_replay':887,'geometry_round1':888,'db_geometry1':889,'db_geometry1_unexecuted_remainder':900}
 state['actual_stage_jobs'].update(previous.get('actual_stage_jobs',{}))
 for k in ['round1_geometry','round2_properties','round3_properties','candidate_status','fresh_failure_property','saturation_status']:
  if k in previous:state[k]=previous[k]
 for k,v in previous.items():state.setdefault(k,v)
 dump('working_state.json',state)
 print(json.dumps({'memory_checkpointed':True,'ledger_experiments':len(ledger),'evidence_references':len(evidence),'live_job_rows':len(jobs)}))
if __name__=='__main__':main()

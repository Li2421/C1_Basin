#!/usr/bin/env python3
from audit import *
def main():
 state=json.load(open(HERE/'working_state.json'))
 state['updated_local']=time.strftime('%Y-%m-%d %H:%M:%S %Z')
 state['evidence_summary']=json.load(open(HERE/'inventory_summary.json'))
 state['completed_stages']=list(dict.fromkeys(state['completed_stages']+['geometry_round1:222 exactQ64 complete','DBgeometry1:147 exactQ64,1 numerical-affected eta unresolved','negative-constrained five-family refit after_round1_complete']))
 state['next_action']='Monitor901 retained42eta and902 DBgeometry2 88eta. Preserve frozen semialgebraic validation parameters; no training. Any fresh false inclusion rejects this frozen validation attempt.'
 state['actual_stage_jobs'].update(retained_validation_r1=901,db_geometry2=902)
 state['anomalies'].append(dict(issue='DB_T0_3 one second-projection numerical failure after3 identical retries',resolution='exact record quarantined; no tolerance/control change;200 unexecuted continuations resumed via900; affected eta has63 valid trials and is NOT Q64'))
 state['round1_geometry']=dict(star_all_three_B63_paths=25,star_paths=32,native_single_sampled_interval_lines=24,native_tested_lines=24,sampled_failure_corridors_to_boundary=4,tested_failure_corridors=10,confirmed_enclosed_holes=0,hole_absence_proven=False)
 state['candidate_status']=dict(family='semialgebraic',cached_gate='PASS_ON10_ADEQUATE_STATES_NOT_FINAL',full_recall=.870935960591133,retained_recall=.4928571428571429,known_negative_inclusions=0,
  empty_retained_state='T0_WIDE_perm00_ep0082',fresh_validation='42points/7 nonempty Toy states; pending',cross_scenario_gate='12adequate states not yet reached')
 dump('working_state.json',state)
 hs=json.load(open(HERE/'hypothesis_status.json'));gates={g['family']:g for g in json.load(open(HERE/'family_gates_after_round1_complete.json'))}
 for h in hs:
  if h.get('family') not in gates:continue
  g=gates[h['family']];h['latest_gate_file']='family_gates_after_round1_complete.json';h['key_support']={k:g[k] for k in ['median_full_recall','median_retained_recall','pooled_transfer_recall']}
  h['key_failure']={'full_false':g['cached_full_false_inclusions'],'retained_false':g['cached_retained_false_inclusions'],'fraction_recall_ge_half':g['fraction_recall_ge_half']}
  h['current_status']='CACHED_PASS_FRESH_VALIDATION_PENDING' if g['cached_screen_pass'] else 'ALL_KNOWN_NEGATIVE_CONSTRAINED_INSTANCE_RECALL_FAILED'
  h['next_discriminating_test']='prospective retained precision901 and cross-scenario conditional/common-mode902'
  if h['family']=='semialgebraic':h['key_failure']['empty_instance']='ep0082; do not hide under aggregate recall'
 dump('hypothesis_status.json',hs)
 ei=json.load(open(HERE/'evidence_index.json'))
 for name in ['audit.py','families.py','analyze.py','run_db.py','checkpoint_work.py','prepare_db_geometry.py','prepare_round2.py','validation_protocol.md','family_gates_after_round1_complete.json','execution_anomaly_resolution.json','cached_link_protocol.json']:
  ei['INSPECTED:'+name]={'path':str(HERE/name),'sha256':sha(HERE/name),'used_in_current_stage':True}
 dump('evidence_index.json',ei)
 dump('round1_runtime_summary.json',dict(geometry_new_continuations=sum(json.load(open(p))['new_continuations'] for p in (HERE/'raw/geometry_round1').glob('*runtime.json')),
  geometry_physical_steps=sum(json.load(open(p))['physical_steps'] for p in (HERE/'raw/geometry_round1').glob('*runtime.json')),
  geometry_critical_worker_seconds=max(json.load(open(p))['wall_seconds'] for p in (HERE/'raw/geometry_round1').glob('*runtime.json')),
  db_new_attempts=sum(1 for p in (HERE/'db_raw').glob('db_geometry1_*.jsonl') for _ in open(p)),db_quarantined=1,db_q64_complete=147,
  max_gpu_shards=6,max_cpu_threads=12,slurm_accounting='disabled; runtime ledgers and resource snapshots used'))
 print('Round1 checkpointed; next jobs901/902; no final gate claimed.')
if __name__=='__main__':main()

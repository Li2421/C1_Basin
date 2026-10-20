"""Independent delivery checks; run only after numerical completion."""
from common import *
import csv
from datetime import datetime

def main():
 assert (HERE/'NUMERICAL_COMPLETE.receipt.json').exists(),'Batch still running'
 receipt=read(HERE/'NUMERICAL_COMPLETE.receipt.json')
 for path,digest in receipt['files'].items():assert sha(path)==digest,path
 for name in ['PREREGISTRATION.receipt.json','CHOICES_FROZEN.receipt.json','CONSISTENCY_GATE.json','main_PREFLIGHT.receipt.json','SPEEDUP_PREFLIGHT.receipt.json','DB_COMPLETENESS.json','FINAL_RESERVATION_AUDIT.json','INTEGRITY_BEGIN.json','INTEGRITY_END.json','INTEGRITY_COMPARISON.json','CONTROLLER_MEMBERSHIP_AUDIT.json','EXACT_CHANGED_PATHS.json','GAPS.json','HANDOFF.md','AXIS_TABLE.csv','AXIS_TABLE.md','REPORT.md']:
  assert (HERE/name).exists(),name
 times={n:datetime.fromisoformat(read(HERE/n)['utc']) for n in ['PREREGISTRATION.receipt.json','CHOICES_FROZEN.receipt.json','main_SUBMISSION.json','ENDPOINT_OPENING.receipt.json']}
 ordered=list(times.values());assert ordered==sorted(ordered)
 gate=read(HERE/'CONSISTENCY_GATE.json');assert gate['passed'] and len(gate['checks'])==16 and all(c['passed'] for c in gate['checks'])
 complete=read(HERE/'DB_COMPLETENESS.json');assert complete['registered_records']==16128 and complete['missing_unexecuted']==0 and complete['conflicts']==0
 assert not read(HERE/'FINAL_RESERVATION_AUDIT.json')['remaining']
 z=np.load(HERE/'LABELS.npz');assert z['lower'].shape==(14,96,6,2);assert np.all(z['upper']>=z['lower']);assert int((z['upper']-z['lower']).sum())==complete['unknown_preserved']
 rows=read(HERE/'main_CASES.json');assert len(rows)==len({r['rollout_uid'] for r in rows})==16128
 trace_count=0
 for c in rows:
  r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json");assert r.get('case')==c and r['rollout_uid']==c['rollout_uid'];assert not r['jax_enable_x64'];assert not (r['success'] and (r['collision'] or r['numerical_failure']))
  assert sha(r['trace_path'])==r['trace_sha256'];trace_count+=1
 result=read(HERE/'results.json')
 for cohort,ns in [('1a',64),('1b',32)]:
  q=result[cohort];assert q['controllers']==14 and q['states']==ns and q['bootstrap_replicates']==5000
  for name,m in q['methods'].items():
   low,high=m['success_pct'];ci=m['success_95CI_pct'];assert -1e-8<=low<=high<=100+1e-8 and ci[0]<=ci[1]
   assert m['mean_regret_pp'][0]<=m['max_regret_pp'][0]+1e-8
  assert q['methods']['F10']['catastrophes_certain']==0 and q['methods']['safety baseline']['harm_certain']==0
  assert sum(x['denominator'] for x in q['nominal_to_safety'].values())==14*ns*2
  for row in q['nominal_to_safety'].values():assert sum(row['counts'].values())==row['denominator']
  e=q['endpoint'];rg=e['A3_vs_crossfit_oracle_regret_pp'];ci=e['A3_minus_safety_CI_pp'];expected='SUPPORTED' if rg[1]<=3 and ci[0]>0 else 'REJECTED' if rg[0]>3 or ci[1]<=0 else 'INCONCLUSIVE';assert e['status']==expected
  ci=q['ST3']['CI_pp'];expected='gain' if ci[0]>0 else 'harm' if ci[1]<0 else 'undetected';assert q['ST3']['status']==expected
 scope=read(HERE/'CONTROLLER_MEMBERSHIP_AUDIT.json');assert scope['exact_checkpoint_overlap']==0 and scope['parameter_hash_overlap']==0 and result['scope']['seen_exact_checkpoint_count']==0
 assert 'GAP' in result['scope']['strict_state_axis']
 table=list(csv.DictReader((HERE/'AXIS_TABLE.csv').open()));assert len(table)>=6
 audit=read(HERE/'INTEGRITY_COMPARISON.json');report=(HERE/'REPORT.md').read_text();assert ('完整性比对通过' if audit['passed'] else '完整性比对未通过') in report.splitlines()[0]
 assert '纯State轴' in report.splitlines()[0]
 write(HERE/'INDEPENDENT_DELIVERY_REVIEW.json',dict(utc=utc(),passed=True,raw_records=len(rows),traces_SHA256_verified=trace_count,chronology={k:v.isoformat() for k,v in times.items()},unknown_preserved=complete['unknown_preserved'],all_required_artifacts_present=True,scope_gap_explicit=True,integrity_passed=audit['passed'],integrity_qualification_preserved=True))
 print('PASS:16128 records/traces, paired stats invariants, temporal freeze, DB completeness, explicit audit/scope qualifications')
if __name__=='__main__':main()

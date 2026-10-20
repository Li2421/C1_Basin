from common import *
from shared_rollout_db.src.rollout_db import uid,canonical
import collections,ast,csv

def main():
 prereg();errors=[];counts={};unknowns={};uids=[];canonical_ids=[];physicalsteps=collections.Counter();phases=[s for s in ['gate','expert_gate','baseline','horizon','horizon_expert','pilot','geometry','geometry_expert'] if (HERE/f'{s}_CASES.json').exists()]
 def check(ok,msg):
  if not ok:errors.append(msg)
 for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 for receipt in ['REFERENCE_CHOICES_FROZEN.receipt.json','PSTAR_CONDITIONAL_CANDIDATE_FROZEN.receipt.json','ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json','CAPABILITY_GATE_FROZEN.receipt.json']:
  for f,h in read(HERE/receipt)['files'].items():check(sha(f)==h,'Frozen digest '+f)
 with rollout_db.connect(True) as db:
  for stage in phases:
   cases=read(HERE/f'{stage}_CASES.json');counts[stage]=len(cases);unknowns[stage]=0;check(len({c['rollout_uid'] for c in cases})==len(cases),'Duplicate request '+stage);pre=read(HERE/f'{stage}_PREFLIGHT.receipt.json');sub=read(HERE/f'{stage}_SUBMISSION.json') if (HERE/f'{stage}_SUBMISSION.json').exists() else read(HERE/'gate_pre_execution_design/gate_SUBMISSION.json');check(pre['utc']<sub['utc'] or stage=='gate','Preflight chronology '+stage);aliases=[]
   for v in read(HERE/f'{stage}_REGISTRATION.receipt.json')['preexisting_row_hashes'].values():check(v['before']==v['after'],'Existing registry mutation '+stage)
   for c in cases:
    actualseed=canonical(dict(future_index=c['seed']));actual=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=actualseed));uids.append(c['rollout_uid']);canonical_ids.append(actual);aliases.append(dict(request_rollout_uid=c['rollout_uid'],db_rollout_uid=actual));f=HERE/stage/'raw'/f"{c['rollout_uid']}.json";q=read(f);check(q['case']==c,'Raw/case mismatch '+str(f));check(not q['jax_enable_x64'],'Precision '+str(f));check(q['timeout']==int(q['outcome']=='timeout') and q['deadlock']==0,'Termination taxonomy '+str(f));check(q['numerical_failure']==int(q['error'] is not None),'Unknown flags '+str(f));unknowns[stage]+=q['numerical_failure'];check(sha(q['trace_path'])==q['trace_sha256'],'Trace digest '+str(f));z=np.load(q['trace_path']);pos=z['positions'];vel=z['velocities'];act=z['actions'];n=len(pos)-1;check(q['physical_steps']==n and len(act)==n,'Trace lengths '+str(f));check(q['new_physical_steps']+q['prefix_reused_steps']==n,'Physical step accounting '+str(f));check(q['actual_config']['max_steps']==c['H'] and q['actual_config']['obstacle_radius']==c['r'],'Factor config '+str(f));check(q['actual_config']['agent_radius']==.16 and q['actual_config']['max_speed']==.52 and q['actual_config']['dt']==.05,'Physical invariant '+str(f))
    if n:check(np.array_equal(pos[1:],pos[:-1]+.05*act),'Exact physical integration '+str(f))
    src=q.get('prefix_provenance')
    if src:
     check(sha(src['raw_path'])==src['raw_sha256'] and sha(src['trace_path'])==src['trace_sha256'],'Prefix source digests '+str(f));old=np.load(src['trace_path']);prefix=q['prefix_reused_steps'];check(np.array_equal(pos[:prefix+1],old['positions'][:prefix+1]) and np.array_equal(vel[:prefix+1],old['velocities'][:prefix+1]),'Prefix bytes '+str(f))
    row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(actual,)).fetchone();check(row is not None,'DB exact record '+actual)
    if row:
     check(not row['conflict_quarantined'],'Conflict '+actual);check(row['raw_record_hash']==hashlib.sha256(canonical(q).encode()).hexdigest(),'DB raw hash '+actual);check(all(row[k]==q[k] for k in ['success','collision','timeout','deadlock','numerical_failure']),'DB outcomes '+actual)
    for k in ['new_physical_steps','prefix_reused_steps']:physicalsteps[k]+=q[k]
   write(HERE/f'{stage}_DB_IDENTITY_MAP.json',aliases);manifest=HERE/f'{stage}_CANONICAL_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[dict(state_uid=c['state_uid'],eta_uid=c['eta_uid'],controller_uid=c['controller_uid'],seed_key=canonical(dict(future_index=c['seed']))) for c in cases]));pf=planner.preflight(manifest);write(HERE/f'{stage}_CANONICAL_POSTFLIGHT.json',pf);check(pf['summary']['ambiguous']==0,'Canonical preflight ambiguous '+stage)
  dbcount=db.execute('SELECT COUNT(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0];check(dbcount==sum(counts.values()),'Authoritative DB count')
 check(len(set(uids))==len(uids) and len(set(canonical_ids))==len(canonical_ids),'Crossstage duplicate');counter=collections.Counter()
 for f in (cache_writer.ROOT/'journals'/EXP).iterdir():
  if f.suffix not in ['.merged','.jsonl']:continue
  for line in f.read_text().splitlines():counter[json.loads(line)['case']['rollout_uid']]+=1
 good=set(counter)==set(uids) and all(n==1 for n in counter.values());check(good,'Journal missing/repeated');write(HERE/'JOURNAL_UNIQUENESS_AUDIT.json',dict(passed=good,records=sum(counter.values()),unique=len(counter),duplicates={k:v for k,v in counter.items() if v!=1}));check(read(HERE/'CONSISTENCY_GATE.json')['passed_count']==16 and read(HERE/'EXPERT_CONSISTENCY_GATE.json')['passed_count']==16,'Historical reproduction gates');check(read(HERE/'HORIZON_NUMERICAL_COMPLETE.receipt.json')['utc']<read(HERE/'pilot_SUBMISSION.json')['utc'],'Horizon-before-geometry');check(read(HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json')['utc']<read(HERE/'baseline_SUBMISSION.json')['utc'],'Analysis freeze chronology');check(read(HERE/'REFERENCE_CHOICES_FROZEN.receipt.json')['utc']<read(HERE/'baseline_SUBMISSION.json')['utc'],'Choice freeze chronology')
 result=read(HERE/'results.json');choices=read(HERE/'FROZEN_REFERENCE_CHOICES.json')['controllers']['A3'];names=read(HERE/'HISTORICAL_INVENTORY.json')['controller_names']
 for label,r in result.items():
  if not isinstance(r,dict) or 'methods' not in r:continue
  z=np.load(HERE/(label+'_LABELS.npz'));arms=z['arms'].tolist();rel='S500' if r['H']==700 else 'Srel';ix=np.array([arms.index(rel if choices[n]=='S500' else choices[n]) for n in names]);value=z['lower'][np.arange(14)[:,None],np.arange(32)[None,:],ix[:,None]].mean()*100;check(abs(value-r['methods']['A3']['success_pct'][0])<1e-10,'Independent A3 aggregate '+label)
 check(result['endpoints']['HZ3']['status']=='GAP' and all(v['status']=='INCONCLUSIVE' for v in result['endpoints']['HZ4'].values()),'Undefined pstar endpoint preserved');check(len(list(csv.DictReader((HERE/'AXIS_TABLE.csv').open())))==8,'Eight table rows')
 required=['REPORT.md','HANDOFF.md','AXIS_TABLE.md','AXIS_TABLE.csv','METHOD_METRICS.json','METHOD_METRICS.csv','results.json','GAPS.json','INTEGRITY_DETAILS.json','JOURNAL_UNIQUENESS_AUDIT.json','CAPABILITY_GATE.json','PSTAR_SOURCE_COVERAGE.json','PSTAR_CALIBRATION_CANDIDATES.json','SEED_TRANSPORT_REPAIR.json']
 for f in required:check((HERE/f).exists(),'Missing delivery '+f)
 comp=read(HERE/'INTEGRITY_COMPARISON.json');check(('未完全通过' in (HERE/'REPORT.md').read_text()) if not comp['passed'] else ('审计通过' in (HERE/'REPORT.md').read_text()),'Integrity disclosure');review=dict(utc=utc(),passed=not errors,errors=errors,counts=counts,unknowns=unknowns,authoritative_DB_rollouts=dbcount,source_integrity_passed=comp['passed'],physical_step_accounting=dict(physicalsteps),no_unknown_retries=True,all_source_prefix_trace_and_raw_hashes_checked=True);write(HERE/'INDEPENDENT_DELIVERY_REVIEW.json',review);assert not errors,errors[:20];seal('NUMERICAL_COMPLETE.receipt.json',[HERE/f for f in required]+[HERE/'INDEPENDENT_DELIVERY_REVIEW.json',HERE/'INTEGRITY_COMPARISON.json'],numerical_complete=True,source_integrity_passed=comp['passed'],counts=counts,unknowns=unknowns,Sprog='GAP');print('Round35 numerical review PASS',counts,unknowns,flush=True)
if __name__=='__main__':main()

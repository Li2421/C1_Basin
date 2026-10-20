from common import *
from pipeline import lookup
from prefix import choose_prefix
import collections,csv,ast

def main():
 prereg();errors=[];counts={};unknowns={};unique={};steps=collections.Counter();phaseuids={}
 def check(ok,msg):
  if not ok:errors.append(msg)
 for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 receipts=['GATE_IMPLEMENTATION_FROZEN.receipt.json','ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json','CONFIG_TRAJECTORY_UNIT_FROZEN.receipt.json','REFERENCE_CHOICES_FROZEN.receipt.json','GEOMETRY_ELIGIBILITY_FROZEN.receipt.json','B_ANALYSIS_FROZEN.receipt.json','N2_GATE_AUDIT.receipt.json','N2_HISTORICAL_TABLE.receipt.json','DELIVERY_IMPLEMENTATION_FROZEN.receipt.json']
 if (HERE/'PSTAR_DECISION.receipt.json').exists():receipts+=['PSTAR_DECISION.receipt.json']
 if (HERE/'B_COMPLETE.receipt.json').exists():receipts+=['B_COMPLETE.receipt.json','SELECTED_PROGRESS_TEST.receipt.json']
 for name in receipts:
  for f,h in read(HERE/name)['files'].items():check(sha(f)==h,'Frozenhash '+f)
 check(read(HERE/'CONFIG_UNIT_TEST.json')['passed'],'r.90configs');g=read(HERE/'CONSISTENCY_GATE.json');check(g['passed'] and all(x['total']==x['passed']==16 for x in g['groups'].values()),'Two16-byte-gates');check(read(HERE/'A1_STATE_DEDUP.json')['passed'],'A1dedup')
 phases=[x for x in ['gate','A','GA3','B'] if (HERE/f'{x}_CASES.json').exists()]
 with rollout_db.connect(True) as db:
  for stage in phases:
   cases=read(HERE/f'{stage}_CASES.json');phaseuids[stage]={c['rollout_uid'] for c in cases};counts[stage]=len(cases);unknowns[stage]=0;check(len(phaseuids[stage])==len(cases),'Withinphaseunique '+stage);check(read(HERE/f'{stage}_PREFLIGHT.receipt.json')['utc']<read(HERE/f'{stage}_SUBMISSION.json')['utc'],'Preflight chronology '+stage)
   for x in read(HERE/f'{stage}_REGISTRATION.receipt.json')['preexisting_row_hashes'].values():check(x['before']==x['after'],'Existingregistrymutation '+stage)
   for c in cases:
    path=lookup(c,stage);q=read(path);unknowns[stage]+=q['numerical_failure'];check(q['case']==c,'Rawcase '+c['rollout_uid'])
    if c['rollout_uid'] in unique:check(unique[c['rollout_uid']]['raw_sha256']==sha(path),'ReusedUIDdifferentraw '+c['rollout_uid']);continue
    unique[c['rollout_uid']]=dict(raw_path=str(path),raw_sha256=sha(path),experiment_uid=q['experiment_uid']);check(not q['jax_enable_x64'],'Precision '+c['rollout_uid']);check(q['deadlock']==0 and q['timeout']==int(q['outcome']=='timeout') and q['numerical_failure']==int(q['error'] is not None),'Taxonomy '+c['rollout_uid']);check(sha(q['trace_path'])==q['trace_sha256'],'Tracehash '+c['rollout_uid']);check(q['actual_config']['obstacle_radius']==c['r'] and q['actual_config']['max_steps']==c['H'] and q['actual_config']['agent_radius']==.16 and q['actual_config']['max_speed']==.52 and q['actual_config']['dt']==.05,'Physicalconfig '+c['rollout_uid'])
    with np.load(q['trace_path']) as z:
     pos=z['positions'];vel=z['velocities'];act=z['actions'];n=len(pos)-1;check(n==q['physical_steps']==q['episode_length']==len(act),'Tracelength '+c['rollout_uid']);check(q['new_physical_steps']+q['prefix_reused_steps']==n,'Stepcost '+c['rollout_uid'])
     if n:check(np.array_equal(pos[1:],pos[:-1]+.05*act) and np.array_equal(vel[1:],act),'Nativeintegration '+c['rollout_uid'])
     ref=q.get('prefix_provenance')
     if ref:
      old=read(ref['raw_path']);check(sha(ref['raw_path'])==ref['raw_sha256'] and sha(ref['trace_path'])==ref['trace_sha256'],'Prefixhash '+c['rollout_uid']);check(old['case']['base_controller_name']==c['base_controller_name'] and old['case']['seed']==c['seed'] and old.get('flow_checkpoint_sha256',old.get('checkpoint_sha256'))==q['flow_checkpoint_sha256'] and q['flow_checkpoint_sha256'] is not None,'Prefixcontroller/RNG '+c['rollout_uid'])
      with np.load(ref['trace_path']) as oz:
       end=q['prefix_reused_steps'];check(np.array_equal(pos[:end+1],oz['positions'][:end+1]) and np.array_equal(vel[:end+1],oz['velocities'][:end+1]),'Prefixbytes '+c['rollout_uid']);choice=choose_prefix(c,old['case'],oz['positions'],oz['goals'],old['outcome']);check(choice==ref['compatibility'] and choice['steps']==end,'Prefixproof '+c['rollout_uid'])
      if ref['compatibility']['kind']=='UNKNOWN_REUSE':check(q['error']==old['error'] and q['new_physical_steps']==0,'Unknownrerun '+c['rollout_uid'])
    r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();check(r is not None,'MissingDB '+c['rollout_uid'])
    if r:check(not r['conflict_quarantined'] and r['raw_record_hash']==hashlib.sha256(rollout_db.canonical(q).encode()).hexdigest() and all(r[k]==q[k] for k in ['success','collision','timeout','deadlock','numerical_failure']),'DBraw '+c['rollout_uid'])
    for k in ['new_physical_steps','prefix_reused_steps']:steps[k]+=q[k]
   check(read(HERE/f'{stage}_POSTFLIGHT.json')['planner']['summary']['ambiguous']==0,'Ambiguous '+stage)
  actual={r[0] for r in db.execute('SELECT rollout_uid FROM rollout WHERE experiment_uid=?',(EXP,))};expected={uid for uid,v in unique.items() if v['experiment_uid']==EXP};check(actual==expected,'AuthoritativeDBUIDset');dbcount=len(actual)
 counter=collections.Counter()
 for f in (cache_writer.ROOT/'journals'/EXP).iterdir():
  if f.suffix in ['.jsonl','.merged']:
   for line in f.read_text().splitlines():counter[json.loads(line)['case']['rollout_uid']]+=1
 check(set(counter)==expected and all(n==1 for n in counter.values()),'Journalmultiplicity');write(HERE/'JOURNAL_UNIQUENESS_AUDIT.json',dict(passed=set(counter)==expected and all(n==1 for n in counter.values()),records=sum(counter.values()),unique=len(counter),reused_stage_queries_not_rejournaled=True));check(counts['gate']==32 and counts['A']==2240,'Prespecifiedcontrolcount')
 if 'GA3' in phases:
  overlap=phaseuids['GA3']&phaseuids['A'];wanted={c['rollout_uid'] for c in read(HERE/'A2_CASES.json') if c['r'] in read(HERE/'CONTROL_RESULTS.json')['GA3']['eligible_radii']};check(overlap==wanted,'ExactA2GA3reuse');check(read(HERE/'REFERENCE_CHOICES_FROZEN.receipt.json')['utc']<read(HERE/'A_SUBMISSION.json')['utc'],'ChoicebeforeA2');check(read(HERE/'GEOMETRY_ELIGIBILITY_FROZEN.receipt.json')['utc']<read(HERE/'GA3_SUBMISSION.json')['utc'],'Conditionalmainchronology')
 if 'B' in phases:
  d=read(HERE/'PSTAR_DECISION.receipt.json');check(d['authorized_rollouts'] and read(HERE/'SELECTED_PROGRESS_TEST.receipt.json')['utc']<read(HERE/'B_SUBMISSION.json')['utc'] and d['utc']<read(HERE/'B_SUBMISSION.json')['utc'],'Bauthorization/freeze');check(counts['B']==2688,'Bcount')
 else:check(not (HERE/'B_SUBMISSION.json').exists() and not (HERE/'B/raw').exists(),'No unauthorizedB')
 control=read(HERE/'CONTROL_RESULTS.json')
 with np.load(HERE/'A1_LABELS.npz') as z:
  lo=np.median(z['lower'].mean((1,3)).max(1))*100;hi=np.median(z['upper'].mean((1,3)).max(1))*100;check([lo,hi]==control['GA1']['median_max_success_pct'],'IndependentGA1')
 check(read(HERE/'OLD_SOURCE_IMMUTABILITY.json')['passed'],'Old35/36immutable');check(len(list(csv.DictReader((HERE/'AXIS_TABLE.csv').open())))==8,'Eightrows');check('阳性对照' in (HERE/'AXIS_TABLE.csv').read_text() and '同 state 跨半径' in (HERE/'AXIS_TABLE.csv').read_text(),'Addedcolumns');check(read(HERE/'N2_GATE_AUDIT.json')['conclusion']=='门的执行正确，N=2 Flow 在相向任务上能力不足','Cconclusion');integrity=read(HERE/'INTEGRITY_COMPARISON.json');check(('审计通过' in (HERE/'REPORT.md').read_text()) if integrity['passed'] else ('未完全通过' in (HERE/'REPORT.md').read_text()),'Integritydisclosed')
 required=['REPORT.md','HANDOFF.md','results.json','AXIS_TABLE.md','AXIS_TABLE.csv','METHOD_METRICS.json','METHOD_METRICS.csv','GAPS.json','A_CAPABILITY_TABLE.json','A_CAPABILITY_TABLE.md','CONTROLLER_RADIUS_CURVES.csv','CONTROLLER_RADIUS_CURVES.json','CONTROLLER_RADIUS_CURVES.png','CONTROLLER_RADIUS_CURVES.pdf','N2_GATE_AUDIT.json','N2_GATE_AUDIT.md','N2_GATE_EPISODES.csv','N2_HISTORICAL_EVIDENCE_TABLE.md','INTEGRITY_DETAILS.json','OLD_SOURCE_IMMUTABILITY.json','JOURNAL_UNIQUENESS_AUDIT.json','CONFIG_UNIT_TEST.json','CONSISTENCY_GATE.json']
 for f in required:check((HERE/f).exists(),'Missingdelivery '+f)
 review=dict(utc=utc(),passed=not errors,errors=errors,query_counts=counts,unknown_query_counts=unknowns,authoritative_DB_records=dbcount,unique_evidence_records=len(unique),physical_step_accounting=dict(steps),unknown_retries=0,source_integrity_passed=integrity['passed'],B_executed='B' in phases);write(HERE/'INDEPENDENT_DELIVERY_REVIEW.json',review);assert not errors,errors[:20];seal('NUMERICAL_COMPLETE_B.receipt.json',[HERE/f for f in required]+[HERE/'INDEPENDENT_DELIVERY_REVIEW.json',HERE/'INTEGRITY_COMPARISON.json',HERE/'A_COMPLETE.receipt.json'],A_C_complete=True,B_executed='B' in phases,B_status=read(HERE/'results.json')['B']['status']);print('Round37review PASS',review,flush=True)
if __name__=='__main__':main()

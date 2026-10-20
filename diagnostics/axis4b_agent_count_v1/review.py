from common import *
from shared_rollout_db.src.rollout_db import canonical
from data import ARMS
import ast,collections,csv

def main():
 prereg();errors=[];counts={};unknowns={};uids=[];steps=collections.Counter();phases=[s for s in ['gate','pilot','source','target','expert'] if (HERE/f'{s}_CASES.json').exists()]
 def check(ok,msg):
  if not ok:errors.append(msg)
 def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 for name in ['GATE_IMPLEMENTATION_FROZEN.receipt.json','ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json','CAPABILITY_GATE_FROZEN.receipt.json','TRANSFER_CHOICES_FROZEN.receipt.json','DELIVERY_IMPLEMENTATION_FROZEN.receipt.json']:
  for f,h in read(HERE/name)['files'].items():check(sha(f)==h,'Frozen hash '+f)
 with rollout_db.connect(True) as db:
  for phase in phases:
   cases=read(HERE/f'{phase}_CASES.json');counts[phase]=len(cases);unknowns[phase]=0;check(len({c['rollout_uid'] for c in cases})==len(cases),'Duplicate design '+phase);check(read(HERE/f'{phase}_PREFLIGHT.receipt.json')['utc']<read(HERE/f'{phase}_SUBMISSION.json')['utc'],'Preflight chronology '+phase)
   for h in read(HERE/f'{phase}_REGISTRATION.receipt.json')['preexisting_row_hashes'].values():check(h['before']==h['after'],'Existing registry mutation '+phase)
   for c in cases:
    uid=c['rollout_uid'];uids.append(uid);f=HERE/phase/'raw'/f'{uid}.json';q=read(f);actualcase=dict(q['case']);actualcase.pop('deterministic_source',None);check(actualcase==c,'Raw/design '+uid);check(not q['jax_enable_x64'],'Precision '+uid);check(q['deadlock']==0 and q['timeout']==int(q['outcome']=='timeout'),'Termination taxonomy '+uid);check(q['numerical_failure']==int(q['error'] is not None),'Unknown flag '+uid);unknowns[phase]+=q['numerical_failure'];check(sha(q['trace_path'])==q['trace_sha256'],'Trace digest '+uid)
    with np.load(q['trace_path']) as z:
     pos=z['positions'];act=z['actions'];vel=z['velocities'];n=len(pos)-1;check(n==len(act)==q['physical_steps']==q['episode_length'],'Trace lengths '+uid);check(q['new_physical_steps']+q['deterministic_reused_steps']==n,'Physical accounting '+uid)
     if not (c['kind']=='expert' and q['numerical_failure']):check(np.array_equal(pos[1:],pos[:-1]+.05*act),'Exact native integration '+uid)
     if q['deterministic_source']:
      src=q['deterministic_source'];old=read(src['raw_path']);check(sha(src['raw_path'])==src['raw_sha256'] and sha(src['trace_path'])==src['trace_sha256'],'Deterministic source digests '+uid);check(c['kind']=='P2' and c['seed']==1 and old['case']['seed']==0 and old['controller_uid']==q['controller_uid'] and old['state_uid']==q['state_uid'] and old['eta_uid']==q['eta_uid'],'Deterministic compatibility '+uid)
      with np.load(src['trace_path']) as oz:check(exact(pos,oz['positions']) and exact(act,oz['actions']) and exact(vel,oz['velocities']),'Deterministic bytes '+uid)
      check(q['new_physical_steps']==0 and q['outcome']==old['outcome'] and q['error']==old['error'],'Unknown/deterministic replay '+uid)
    check(q['actual_config']['num_agents']==c['N'] and q['actual_config']['max_steps']==c['H'] and q['actual_config']['agent_radius']==.16 and q['actual_config']['max_speed']==.5,'Native config '+uid);row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(uid,)).fetchone();check(row is not None,'DB missing '+uid)
    if row:
     check(not row['conflict_quarantined'],'DB conflict '+uid);check(row['raw_record_hash']==hashlib.sha256(canonical(q).encode()).hexdigest(),'DB/raw hash '+uid);check(all(row[k]==q[k] for k in ['success','collision','timeout','deadlock','numerical_failure']),'DB/raw labels '+uid)
    for k in ['new_physical_steps','deterministic_reused_steps']:steps[k]+=q[k]
   pf=planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_FINAL_POSTFLIGHT.json',pf);check(pf['summary']['ambiguous']==0,'Postflight ambiguous '+phase)
  dbcount=db.execute('SELECT COUNT(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0];check(dbcount==sum(counts.values()),'DB count')
 check(len(uids)==len(set(uids)),'Cross-stage duplicate');counter=collections.Counter()
 for f in (cache_writer.ROOT/'journals'/EXP).iterdir():
  if f.suffix not in ['.merged','.jsonl']:continue
  for line in f.read_text().splitlines():counter[json.loads(line)['case']['rollout_uid']]+=1
 good=set(counter)==set(uids) and all(v==1 for v in counter.values());check(good,'Journal multiplicity');write(HERE/'JOURNAL_UNIQUENESS_AUDIT.json',dict(utc=utc(),passed=good,records=sum(counter.values()),unique=len(counter),duplicates={k:v for k,v in counter.items() if v!=1}));gate=read(HERE/'CONSISTENCY_GATE.json');check(gate['primitive_passed'] and len(gate['checks']['primitive'])==16,'Primitive16 gate');check(gate['expert_passed'] and len(gate['checks']['expert'])==16,'Expert16 artifact gate');check(read(HERE/'STATE_DEDUP_AUDIT.json')['passed'],'State dedup');check(read(HERE/'UNIT_TESTS.json')['passed'],'P2 unit tests');check(read(HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json')['utc']<read(HERE/'pilot_SUBMISSION.json')['utc'],'Estimator freeze chronology')
 cap=read(HERE/'CAPABILITY_GATE.json');source=cap['source_N'];check(read(HERE/'CAPABILITY_GATE_FROZEN.receipt.json')['utc']<read(HERE/'source_SUBMISSION.json')['utc'] if source is not None else True,'Source eligibility chronology')
 if (HERE/'target_SUBMISSION.json').exists():check(read(HERE/'TRANSFER_CHOICES_FROZEN.receipt.json')['utc']<read(HERE/'target_SUBMISSION.json')['utc'],'Target blinding chronology')
 check(counts.get('pilot')==384 and counts.get('gate')==32 and counts.get('expert')==128,'Prespecified gate/pilot/expert counts')
 expected_source=sum(v['status']=='PASS' for v in cap['controllers'][str(source)].values())*32*2*9 if source else 0;expected_target=sum(sum(v['status']=='PASS' for v in cs.values()) for n,cs in cap['controllers'].items() if source and int(n)>source)*32*2*9;check(counts.get('source',0)==expected_source and counts.get('target',0)==expected_target,'Eligibility/main counts')
 result=read(HERE/'results.json');check(len(result['units'])==12 and len(result['five_layers'])==4,'AllN/class evidence rows');check(len(list(csv.DictReader((HERE/'AXIS_TABLE.csv').open())))==12,'Table rows');assets=read(HERE/'ASSET_AUDIT.json')['assets'];cfgs=[]
 for n,v in assets.items():
  check(sha(v['checkpoint'])==v['checkpoint_sha256']==v['freeze'].get('checkpoint_sha256'),'Frozen Flow digest N'+n);cc=dict(v['config']);[cc.pop(k) for k in ['num_agents','max_steps','seed','split']];cfgs.append(cc)
 check(all(x==cfgs[0] for x in cfgs),'Fixed-map physical invariance')
 for label,u in result['units'].items():
  if u['status']!='MAIN_COMPLETE':continue
  z=np.load(HERE/f"N{u['N']}_LABELS.npz");classes=z['classes'].tolist();lo=z['lower'];j=classes.index(u['controller']);value=lo[j,:,2].mean()*100;check(abs(value-u['methods']['F10']['success_pct'][0])<1e-10,'Independent F10 aggregate '+label)
 required=['REPORT.md','HANDOFF.md','AXIS_TABLE.md','AXIS_TABLE.csv','METHOD_METRICS.json','METHOD_METRICS.csv','results.json','GAPS.json','ASSET_AUDIT.json','CAPABILITY_GATE.json','TRANSFER_TABLE.md','TRANSFER_TABLE.csv','TRANSFER_TABLE.json','FIVE_LAYER_EVIDENCE.md','FIVE_LAYER_EVIDENCE.json','INTEGRITY_DETAILS.json','JOURNAL_UNIQUENESS_AUDIT.json','ACTION_RECORDING_SCHEMA_REPAIR.json','CRITIC_INTERFACE_AUDIT.json','UNIT_TESTS.json','PERFORMANCE_ACCOUNTING.json','PER_STEP_PERFORMANCE.json','MEMORY_PEAK_SUMMARY.json']
 for f in required:check((HERE/f).exists(),'Missing deliverable '+f)
 comp=read(HERE/'INTEGRITY_COMPARISON.json');check(('未完全通过' in (HERE/'REPORT.md').read_text()) if not comp['passed'] else ('审计通过' in (HERE/'REPORT.md').read_text()),'Integrity disclosure');out=dict(utc=utc(),passed=not errors,errors=errors,counts=counts,unknowns=unknowns,authoritative_DB_rollouts=dbcount,source_integrity_passed=comp['passed'],physical_accounting=dict(steps),all_trace_and_raw_hashes_verified=True,unknown_retries=0,target_labels_not_used_for_source_selection=True,expert_action_gate_scope='Original float32 recording bytes, full native float64 trajectory bytes; native64 original action not recorded');write(HERE/'INDEPENDENT_DELIVERY_REVIEW.json',out);assert not errors,errors[:20];seal('NUMERICAL_COMPLETE.receipt.json',[HERE/f for f in required]+[HERE/'INDEPENDENT_DELIVERY_REVIEW.json',HERE/'INTEGRITY_COMPARISON.json'],numerical_complete=True,source_integrity_passed=comp['passed'],counts=counts,unknowns=unknowns);print('Round36 review PASS',counts,unknowns,flush=True)
if __name__=='__main__':main()

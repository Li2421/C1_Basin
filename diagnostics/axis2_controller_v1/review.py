"""Independent artifact/data consistency checks, without running simulation."""
from common import *
from datetime import datetime
import ast,csv,collections

def main():
 prereg();errors=[];counts={};unknowns={};allids=[];first_main=None
 def check(ok,msg):
  if not ok:errors.append(msg)
 for p in HERE.glob('*.py'):
  ast.parse(p.read_text(),filename=str(p))
 for name in ['MIXED_FROZEN.receipt.json','NEW_CONTROLLERS_FROZEN.receipt.json','CHOICES_FROZEN.receipt.json','R5_FROZEN.receipt.json','ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json','HISTORICAL_SUPPLEMENT_CHOICES.receipt.json']:
  for p,s in read(HERE/name)['files'].items():check(sha(p)==s,'Frozen hash mismatch '+str(p))
 with rollout_db.connect(True) as db:
  for phase in ['gate','baseline','pilot','main']:
   rows=read(HERE/f'{phase}_CASES.json');uids=[c['rollout_uid'] for c in rows];allids+=uids;check(len(set(uids))==len(rows),phase+' duplicate design identities');counts[phase]=len(rows);unknowns[phase]=0
   sub=read(HERE/f'{phase}_SUBMISSION.json');pre=read(HERE/f'{phase}_PREFLIGHT.receipt.json');check(pre['utc']<sub['utc'],phase+' preflight not before submission')
   for r in read(HERE/f'{phase}_REGISTRATION.receipt.json')['preexisting_row_hashes'].values():check(r['before']==r['after'],phase+' old registry row changed')
   for c in rows:
    p=HERE/phase/'raw'/f"{c['rollout_uid']}.json";check(p.exists(),'Missing raw '+str(p))
    if not p.exists():continue
    r=read(p);check(r['case']==c,'Case mismatch '+str(p));check(not r['jax_enable_x64'],'x64 enabled '+str(p));check(sha(r['trace_path'])==r['trace_sha256'],'Trace digest mismatch '+str(p));unknowns[phase]+=r['numerical_failure'];found=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();check(found is not None,'Missing DB UID '+c['rollout_uid'])
    if found:
     check(not found['conflict_quarantined'],'DB quarantined '+c['rollout_uid'])
     for field in ['success','collision','timeout','numerical_failure']:check(found[field]==r[field],'DB/raw outcome mismatch '+field+' '+c['rollout_uid'])
   write(HERE/f'{phase}_POSTFLIGHT.json',planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json'))
  prows=read(HERE/'probe_CASES.json');counts['probe_unlabeled']=len(prows)
  for c in prows:
   r=read(HERE/'probe/raw'/f"{c['rollout_uid']}.json");check(not any(k in r for k in ['success','outcome','termination']),'Probe has terminal label');check(db.execute('SELECT COUNT(*) FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()[0]==0,'Fake probe rollout row');check(sha(r['trace_path'])==r['trace_sha256'],'Probe digest mismatch')
  probe_sources=[dict(r) for r in db.execute("SELECT path,classification,rows_imported FROM source_file WHERE experiment_uid=?",(EXP,)) if Path(r['path']).name.startswith('probe_')];check(len(probe_sources)==len(prows),'Probe source count mismatch');check(all(r['classification']=='SUMMARY_ONLY' and r['rows_imported']==0 for r in probe_sources),'Probe source has task import');write(HERE/'PROBE_DB_SUMMARY_ONLY_AUDIT.json',dict(utc=utc(),sources=len(probe_sources),passed=len(probe_sources)==len(prows) and all(r['classification']=='SUMMARY_ONLY' and r['rows_imported']==0 for r in probe_sources)))
  dbcount=db.execute('SELECT COUNT(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0];check(dbcount==len(allids),'Unexpected experiment rollout count')
 journals=cache_writer.ROOT/'journals'/EXP;counter=collections.Counter()
 for path in journals.iterdir():
  if path.suffix not in ['.merged','.jsonl']:continue
  for line in path.read_text().splitlines():
   record=json.loads(line);counter[record['case']['rollout_uid']]+=1
 expected=set(allids)|{c['rollout_uid'] for c in prows};check(set(counter)==expected,'Journal/design identity mismatch');check(all(n==1 for n in counter.values()),'Repeated episode journal identity')
 write(HERE/'JOURNAL_UNIQUENESS_AUDIT.json',dict(utc=utc(),records=sum(counter.values()),unique=len(counter),expected=len(expected),duplicates={k:v for k,v in counter.items() if v!=1},passed=set(counter)==expected and all(n==1 for n in counter.values())))
 check(len(allids)==len(set(allids)),'Cross-stage identity collision')
 check(counts['gate']==80 and counts['baseline']==6784 and counts['pilot']==768 and counts['main']==12288,'Unexpected phase counts')
 for before,after in [('MIXED_FROZEN.receipt.json','pilot_SUBMISSION.json'),('NEW_CONTROLLERS_FROZEN.receipt.json','pilot_SUBMISSION.json'),('CHOICES_FROZEN.receipt.json','main_SUBMISSION.json'),('ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json','main_SUBMISSION.json')]:check(read(HERE/before)['utc']<read(HERE/after)['utc'],'Freeze chronology '+before)
 check(read(HERE/'CONSISTENCY_GATE.json')['passed'],'Source reproduction gate failed')
 labels=np.load(HERE/'LABELS.npz');check(labels['lower'].shape==(24,32,8,2),'Labels shape');check(np.all(labels['upper']>=labels['lower']),'Unknown envelope inverted');check(int(labels['unknown'].sum())==unknowns['main'],'Unknown array/raw mismatch')
 choices=read(HERE/'FROZEN_CHOICES.json')['controllers'];arms=labels['arms'].tolist();names=labels['controller_names'].tolist();ix=np.asarray([arms.index(choices[n]['A3_full']) for n in names]);a3=labels['lower'][np.arange(24)[:,None],np.arange(32)[None,:],ix[:,None]];res=read(HERE/'results.json');check(abs(a3.mean()*100-res['2d_all']['methods']['A3 full']['success_pct'][0])<1e-10,'Independent frozen A3 rate mismatch')
 for key in ['2a','2b','2cN2','2cTEST','2d_all','G','DM','VO','PT']:check(key in res,'Missing result unit '+key)
 table=list(csv.DictReader((HERE/'AXIS_TABLE.csv').open()));check(len(table)==14,'Required table rows missing')
 comparison=read(HERE/'INTEGRITY_COMPARISON.json');check(('未完全通过' in (HERE/'REPORT.md').read_text()) if not comparison['passed'] else ('审计通过' in (HERE/'REPORT.md').read_text()),'Report integrity disclosure mismatch')
 required=['REPORT.md','HANDOFF.md','AXIS_TABLE.md','AXIS_TABLE.csv','GAPS.json','METHOD_METRICS.csv','HISTORICAL_RESULTS.json','MIXED_LOTO_RESULTS.json','INTEGRITY_DETAILS.json','UNIT_TESTS.json','STATISTICS_TESTS.json','COPIED_SOURCE_DIFFS.json','JOURNAL_UNIQUENESS_AUDIT.json','DEPLOYMENT_PAIRING_AUDIT.json']
 for f in required:check((HERE/f).exists(),'Missing delivery '+f)
 result=dict(utc=utc(),passed=not errors,errors=errors,counts=counts,unknowns=unknowns,authoritative_DB_rollouts=dbcount,source_integrity_passed=comparison['passed'],all_trace_digests_checked=True,unknown_retries=0,scope='Independent mechanical delivery verification; source integrity qualifications retained separately')
 write(HERE/'INDEPENDENT_DELIVERY_REVIEW.json',result);assert not errors,errors
 seal('NUMERICAL_COMPLETE.receipt.json',[HERE/f for f in required]+[HERE/'results.json',HERE/'INDEPENDENT_DELIVERY_REVIEW.json',HERE/'INTEGRITY_COMPARISON.json'],numerical_complete=True,source_integrity_passed=comparison['passed'],counts=counts,unknowns=unknowns,no_next_round_started=True)
 print('Round33 numerical delivery verified',counts,unknowns,flush=True)
if __name__=='__main__':main()

from common import *
import subprocess
from shared_rollout_db.src.rollout_db import connect
from shared_rollout_db.src.planner import preflight
PY=sys.executable

def call(args,log):
 with (HERE/log).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))

def main():
 for path,digest in read(HERE/'FINALIZATION_FROZEN_V3.receipt.json')['files'].items():assert sha(path)==digest,path
 rows=read(HERE/'main_CASES.json');assert len(rows)==16128
 missing=[c['rollout_uid'] for c in rows if not (HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()]
 write(HERE/'FINAL_RAW_INVENTORY.json',dict(utc=utc(),expected=len(rows),present=len(rows)-len(missing),missing=missing));assert not missing

 reservations=[]
 for c in rows:
  r=read(HERE/'main/raw'/f"{c['rollout_uid']}.json")
  if r.get('schema')=='round32_work_reservation_v1':reservations.append(c['rollout_uid']);continue
  assert r.get('rollout_uid')==c['rollout_uid'] and r.get('case')==c and 'outcome' in r,('Incomplete or mismatched raw',c['rollout_uid'])
 write(HERE/'FINAL_RESERVATION_AUDIT.json',dict(remaining=reservations,complete_records=len(rows)-len(reservations)))
 assert not reservations,'Reservation markers are not evidence'

 for path,digest in read(HERE/'CHOICES_FROZEN.receipt.json')['files'].items():assert sha(path)==digest,path
 call([PY,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],'main_MERGE.log')
 write(HERE/'main_POSTFLIGHT.json',preflight(HERE/'main_PLANNED_ROLLOUTS.json'))
 with connect(True) as db:
  records=[dict(db.execute('SELECT rollout_uid,numerical_failure,conflict_quarantined,compatibility_quality FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() or {}) for c in rows]
 assert all(r and not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE' for r in records)
 write(HERE/'DB_COMPLETENESS.json',dict(utc=utc(),expected=len(rows),registered_records=len(records),unknown_preserved=sum(r['numerical_failure'] for r in records),conflicts=0,missing_unexecuted=0,unknown_planner_missing_must_not_be_rerun=True))
 call([PY,'-B',str(HERE/'test_statistics.py')],'FINAL_STATISTICS_TESTS.log')
 call([PY,'-B',str(HERE/'analyze.py')],'ANALYSIS.log')
 call([PY,'-B',str(HERE/'audit.py'),'END'],'INTEGRITY_END.log')
 supplement=read(HERE/'POC_SOURCE_SUPPLEMENT.json');changes=[p for p,h in supplement['files'].items() if sha(p)!=h];write(HERE/'POC_SOURCE_SUPPLEMENT_END.json',dict(utc=utc(),changed=changes,passed=not changes,late_initial_receipt=True))
 if changes:
  a=read(HERE/'INTEGRITY_COMPARISON.json');a['passed']=False;a['changed_sections'].append('PoC supplemental imported sources');write(HERE/'INTEGRITY_COMPARISON.json',a)
 call([PY,'-B',str(HERE/'delivery_v3.py')],'DELIVERY.log')
 write(HERE/'LIVE_STATUS.json',dict(utc=utc(),phase='round32 finalized',status='NUMERICAL_COMPLETE_WITH_AUDIT_QUALIFICATION',round33_started=False))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(HERE/'FINALIZE_ERROR.json',dict(utc=utc(),error=repr(exc),no_automatic_rollout_retry=True));raise

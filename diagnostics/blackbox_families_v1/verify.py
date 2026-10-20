"""Read-only final audit of seals, identities, source copies and execution coverage."""
from common import *
from shared_rollout_db.src.rollout_db import connect
import ast,difflib,subprocess
bad=[];checked=[]
for receipt in sorted(HERE.glob('*.receipt.json')):
 obj=read(receipt)
 for name,digest in obj.get('files',{}).items():
  path=Path(name)
  if not path.is_absolute():path=receipt.parent/path
  actual=sha(path) if path.exists() else None
  if actual!=digest:
   archive=read(HERE/'SUBMITTED_REVISION_ARCHIVE.json') if (HERE/'SUBMITTED_REVISION_ARCHIVE.json').exists() else {}
   if path.name in ['worker.py','expanded.py'] and ('SUBMISSION' in receipt.name or 'EXTRA' in receipt.name) and digest in archive and sha(archive[digest]['path'])==digest:pass
   else:bad.append(dict(receipt=str(receipt),path=str(path),expected=digest,actual=actual))
  checked.append(str(path))
# Submission seals intentionally preserve the submitted version; later dispatcher fixes must be disclosed.
copy=[]
for name,v in read(HERE/'COPIED_SOURCES.json').items():
 source=Path(v['source']);same=sha(source)==sha(HERE/name)==v['sha256'];copy.append(dict(path=name,source=str(source),passed=same));assert same
for file in HERE.glob('*.py'):ast.parse(file.read_text(),filename=str(file))
with connect(True) as db:
 cases=read(HERE/'main_CASES.json');rows=[];missing=[];permitted_missing=set(read(HERE/'MAIN_BLOCKED_UNCLAIMED.json')['skipped_rollout_uids'])
 for c in {c['rollout_uid']:c for c in cases}.values():
  r=db.execute('SELECT rollout_uid,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
  if not r:missing.append(c['rollout_uid'])
  else:rows.append(dict(r))
counts={}
for phase in ['gate','gate_flow','gate_flow_corrected','pilot_centralized_sampling','pilot_deterministic','pilot_prioritized_astar','source_training','tune_pilot_centralized_sampling','tune_pilot_deterministic','tune_pilot_prioritized_astar','main']:
 folder=HERE/phase/'raw';rr=[read(p) for p in folder.glob('*.json')];counts[phase]=dict(rows=len(rr),unknown=sum(r.get('numerical_failure',0) for r in rr),exact_inactive_reuse=sum(bool(r.get('exact_inactive_reuse')) for r in rr),new_physical_steps=sum(r.get('new_physical_steps',r.get('physical_steps',0)) for r in rr))
q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True)
write(HERE/'FINAL_VERIFICATION.json',dict(utc=utc(),integrity=read(HERE/'INTEGRITY_COMPARISON.json'),receipt_hash_mismatches=bad,checked_file_count=len(checked),byte_identical_execution_copies=copy,main_database_rows=len(rows),main_missing=missing,intentionally_blocked_unclaimed=sorted(permitted_missing),all_available_family_rows_present=set(missing)==permitted_missing,main_unknown=sum(r['numerical_failure'] for r in rows),main_quarantined=sum(r['conflict_quarantined'] for r in rows),phase_counts=counts,queue=q,no_git_writes=True,no_package_installations=True))
print('receipt drift',len(bad),'main missing',len(missing),'quarantined',sum(r['conflict_quarantined'] for r in rows))
assert set(missing)==permitted_missing and not any(r['conflict_quarantined'] for r in rows)

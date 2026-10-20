"""Read-only protected-tree audit; outputs only in round30c."""
import sys, os, json, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import shared_rollout_db
from shared_rollout_db.src import cache_writer
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
assert (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
OFF=Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')
POC=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def git(p):
 def run(*args):return subprocess.check_output(['git','-C',str(p),*args])
 return dict(head=run('rev-parse','HEAD').decode().strip(),status=run('status','--porcelain','--untracked-files=all').decode(),diff_sha256=hashlib.sha256(run('diff')).hexdigest())
def tree(p):return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def write(name,x):(HERE/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main(phase):
 data=dict(utc=datetime.now(timezone.utc).isoformat(),basin=git(ROOT),official=git(OFF),poc=tree(POC),round30=tree(ROOT/'diagnostics/teacher_student_macflow_v1'),round30b=tree(ROOT/'diagnostics/teacher_student_macflow_v1b'))
 write('INTEGRITY_'+phase.upper()+'.json',data)
 if phase=='end':
  start=json.loads((HERE/'INTEGRITY_START.json').read_text());delta={}
  for k in ('official','poc','round30','round30b'):
   if data[k]!=start[k]:delta[k]=dict(changed=True)
  for k in ('head','diff_sha256'):
   if data['basin'][k]!=start['basin'][k]:delta['basin_'+k]=dict(before=start['basin'][k],after=data['basin'][k])
  before=set(start['basin']['status'].splitlines());after=set(data['basin']['status'].splitlines())
  external=[s for s in after-before if 'diagnostics/teacher_student_macflow_v1c/' not in s]
  removed=sorted(before-after)
  write('INTEGRITY_COMPARISON.json',dict(protected_content_pass=not delta,changes=delta,concurrent_new_status_entries=external,removed_status_entries=removed,round30c_entries=[s for s in after-before if 'diagnostics/teacher_student_macflow_v1c/' in s]))
if __name__=='__main__':main(sys.argv[1])

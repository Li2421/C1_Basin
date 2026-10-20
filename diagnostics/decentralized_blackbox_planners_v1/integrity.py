from core import *
from guard import guard
import subprocess
G=guard()
HIST=['decentralized_pipeline_v1','teacher_student_macflow_v1','teacher_student_macflow_v1b','teacher_student_macflow_v1c','teacher_student_macflow_v1d','teacher_student_macflow_v1f','teacher_student_line_closeout_v1','axis2_controller_v1','ring_controller_critic_v1','controller_flow_blend_v1']
OFF=Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')
def git(p):
 def run(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(p),*a])
 return dict(head=run('rev-parse','HEAD').decode().strip(),status=run('status','--porcelain').decode(),diff_sha256=hashlib.sha256(run('diff')).hexdigest(),cached_diff_sha256=hashlib.sha256(run('diff','--cached')).hexdigest())
def snapshot():
 roots=[FIELD]+[D/h for h in HIST];files={}
 for root in roots:
  for p in sorted(root.rglob('*')):
   if p.is_file():files[str(p)]=sha(p)
 # All tracked and untracked existing sources outside this round, without mutable DB artifacts.
 for p in ROOT.rglob('*.py'):
  if not p.is_relative_to(HERE) and '/.git/' not in str(p):files[str(p)]=sha(p)
 return dict(utc=utc(),guard=G,files=files,Basin=git(ROOT),Official=git(OFF))
if __name__=='__main__':
 tag=sys.argv[1];v=snapshot();write(HERE/f'INTEGRITY_{tag}.json',v)
 if tag=='END':
  b=read(HERE/'INTEGRITY_BEGIN.json');changed=[p for p,h in b['files'].items() if v['files'].get(p)!=h];extra=[p for p in v['files'] if p not in b['files']];checks=dict(protected_files_unchanged=not changed,official_git_unchanged=b['Official']==v['Official'],basin_head_diff_unchanged=all(b['Basin'][k]==v['Basin'][k] for k in ['head','diff_sha256','cached_diff_sha256']))
  write(HERE/'INTEGRITY_COMPARISON.json',dict(passed=all(checks.values()),checks=checks,changed=changed,new_external_files=extra,git_status_before=b['Basin']['status'],git_status_after=v['Basin']['status'],guard=G));assert all(checks.values()),checks
 print(tag,len(v['files']),flush=True)

from core import *
from guard import guard
import subprocess,time
G=guard()
HIST=['decentralized_fix_selection_v1','decentralized_blackbox_planners_v1','axis2_controller_v1','teacher_student_macflow_v1f','teacher_student_macflow_v1','teacher_student_macflow_v1b','teacher_student_macflow_v1c','teacher_student_macflow_v1d','teacher_student_line_closeout_v1','ring_independent_flow_confirmation_v2','decentralized_pipeline_v1','ring_exchange_stage1','controller_flow_blend_v1','ring_controller_critic_v1','controller_universal_eta_schedule_v1','controller_q_transfer_round3_v1','orthoflow3_tt_ff_matched_learnability_v1','controller_q_diversity_v1']
OFF=Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')
def git(p):
 def run(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(p),*a])
 return dict(head=run('rev-parse','HEAD').decode().strip(),status=run('status','--porcelain').decode(),diff_sha256=hashlib.sha256(run('diff')).hexdigest(),cached_diff_sha256=hashlib.sha256(run('diff','--cached')).hexdigest())
def concurrent():
 q=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%i|%j|%T|%C|%b'],text=True);tracked=subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),'ls-files','-z'],text=True).split('\0');tracked_dirs={p.split('/')[1] for p in tracked if p.startswith('diagnostics/') and len(p.split('/'))>2};out=[];cut=time.time()-86400
 for p in sorted(D.iterdir()):
  if not p.is_dir() or p==HERE or p.name in HIST:continue
  latest=0.;recent=[]
  for f in p.rglob('*'):
   if f.is_file():
    st=f.stat();latest=max(latest,st.st_mtime)
    if st.st_mtime>=cut:recent.append(str(f))
  if p.name not in tracked_dirs and latest>=cut:out.append(dict(path=str(p),reason='Untracked diagnostics directory with writes within previous 24 hours',latest_mtime=latest,recent_files=len(recent)))
 job_outputs=[]
 for line in q.splitlines():
  job=line.split('|')[0];detail=subprocess.check_output(['scontrol','show','job',job],text=True)
  for token in detail.split():
   if token.startswith(('StdOut=','StdErr=')):
    value=token.split('=',1)[1]
    if value.startswith('/'):
     parent=Path(value).parent
     if parent!=HERE and not any(parent.is_relative_to(D/h) for h in HIST):job_outputs.append(str(parent))
 for p in sorted(set(job_outputs)):
  if not any(Path(p).is_relative_to(Path(r['path'])) for r in out):out.append(dict(path=p,reason='Output directory of an already running external Slurm job'))
 write(HERE/'PARALLEL_TASK_DIRECTORIES.json',dict(utc=utc(),squeue=q,directories=out,protected_history_overrides=HIST,condition='No import or writes by this task; changes attributed separately.'))
 return [Path(r['path']) for r in out]
def snapshot(parallel):
 files={};links={};parallel_sources={}
 for root in [FIELD]+[D/h for h in HIST]:
  for p in sorted(root.rglob('*')):
   if p.is_symlink():links[str(p)]=os.readlink(p)
   if p.is_file():files[str(p)]=sha(p)
 for p in ROOT.rglob('*.py'):
  if p.is_relative_to(HERE) or '/.git/' in str(p):continue
  target=parallel_sources if any(p.is_relative_to(q) for q in parallel) else files;target[str(p)]=sha(p)
 return dict(utc=utc(),guard=G,files=files,symlinks=links,parallel_sources=parallel_sources,Basin=git(ROOT),Official=git(OFF))
def main():
 tag=sys.argv[1];parallel=concurrent() if tag=='BEGIN' else [Path(r['path']) for r in read(HERE/'PARALLEL_TASK_DIRECTORIES.json')['directories']];v=snapshot(parallel);write(HERE/f'INTEGRITY_{tag}.json',v)
 if tag=='END':
  b=read(HERE/'INTEGRITY_BEGIN.json');changed=[p for p,h in b['files'].items() if v['files'].get(p)!=h];extra=[p for p in v['files'] if p not in b['files']];pc=[p for p,h in b['parallel_sources'].items() if v['parallel_sources'].get(p)!=h];pa=[p for p in v['parallel_sources'] if p not in b['parallel_sources']];checks=dict(protected_files_unchanged=not changed,no_unclassified_new_external_sources=not extra,official_git_unchanged=b['Official']==v['Official'],basin_head_diff_unchanged=all(b['Basin'][k]==v['Basin'][k] for k in ['head','diff_sha256','cached_diff_sha256']),symlinks_unchanged=b['symlinks']==v['symlinks']);write(HERE/'INTEGRITY_COMPARISON.json',dict(passed=all(checks.values()),checks=checks,changed=changed,new_external_files=extra,parallel_changed_sources=pc,parallel_new_sources=pa,parallel_attribution='Listed parallel directory changes were not written/imported by this task; verified by imported-source audit.',git_status_before=b['Basin']['status'],git_status_after=v['Basin']['status'],guard=G));assert all(checks.values()),checks
 print(tag,len(v['files']),len(parallel),flush=True)
if __name__=='__main__':main()

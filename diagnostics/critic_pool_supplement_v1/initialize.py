from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,json,hashlib,subprocess
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'diagnostics/cross_scene_prospective_v1'
def sha(p):
 p=Path(p).resolve();assert not p.is_relative_to(P);return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def metadata(root):
 out={}
 for base,dirs,files in os.walk(root,followlinks=False):
  for name in dirs+files:
   p=Path(base)/name;s=p.stat(follow_symlinks=False);out[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,mode=s.st_mode)
 return out
def recent(root,cut):
 # Stat only, including nested files; stop at first recent write.
 stack=[root]
 while stack:
  d=stack.pop();s=d.stat(follow_symlinks=False)
  if s.st_mtime>=cut:return dict(path=str(d),mtime_ns=s.st_mtime_ns)
  try:
   with os.scandir(d)as it:
    for e in it:
     s=e.stat(follow_symlinks=False)
     if s.st_mtime>=cut:return dict(path=e.path,mtime_ns=s.st_mtime_ns)
     if e.is_dir(follow_symlinks=False):stack.append(Path(e.path))
  except (PermissionError,FileNotFoundError):continue
 return None

def main():
 v=R/'diagnostics/hetero_scene_pipeline_v1';assert json.loads((v/'COMPLETE.json').read_text())['status']=='COMPLETE'and json.loads((v/'INTEGRITY_COMPARE.json').read_text())['passed']
 assert not(H/'INTEGRITY_START.json').exists();now=datetime.now(timezone.utc);repos={}
 for repo in [R,R.parent/'02_C1_Toy_GiveWay']:
  status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).decode();diff=subprocess.check_output(['git','-C',str(repo),'diff']);cached=subprocess.check_output(['git','-C',str(repo),'diff','--cached'])
  repos[str(repo)]=dict(status=status,diff_sha256=hashlib.sha256(diff).hexdigest(),cached_diff_sha256=hashlib.sha256(cached).hexdigest())
 parallel={};dirs=[d for d in(R/'diagnostics').iterdir()if d.is_dir()and d!=H]
 with ThreadPoolExecutor(max_workers=16)as pool:
  futures={pool.submit(recent,d,now.timestamp()-86400):d for d in dirs}
  for f in as_completed(futures):
   d=futures[f];m=f.result()
   if m:parallel[str(d)]=m
 write('INTEGRITY_START.json',dict(utc=now.isoformat(),repositories=repos,parallel_directories=list(parallel),parallel_metadata=parallel,prospective_metadata=metadata(P),prospective_policy='stat only, never opened or hashed'))
 source=R/'docs/prompt_critic_pool_supplement_v1_20261026.md';(H/'PROTOCOL.md').write_bytes(source.read_bytes());write('PROTOCOL_SOURCE.receipt.json',dict(path=str(source),sha256=sha(source),fully_read=True))
 paths=[source,R/'AGENTS.md',R/'shared_rollout_db/README.md',v/'COMPLETE.json',v/'INTEGRITY_COMPARE.json']
 write('REUSED_EVIDENCE_SHA256_START.json',dict(files={str(p):sha(p)for p in paths}))
 write('RESOURCE_AMENDMENT.json',dict(maximum_total_cpus=18,maximum_rollout_workers=17,maximum_training_processes=14,gpu_shares=0,authority='Latest user instruction:18 CPU limit; supersedes prior24 allowance.'))
 write('CONVENTIONS.json',dict(bootstrap_replicates=5000,state_seed=2026102671,hierarchical_seed=2026102672,learning_curve_seed=2026102673,cv_group_seed=2026102674,full_and_cv_seeds=list(range(10)),class_addition_seeds=list(range(5)),learning_curve_seeds=list(range(3)),original_architecture_steps=4000,other_steps=3000,lr_cosine_period='The full prescribed training length of each variant',learning_curve_sample_count='round(fraction*number_of_available_scenes), at least3',state_identity='Canonical physical positions,velocities,goals within version/N; source UID aliases preserved in provenance',baseline_missing_curves='Per-candidate scene means only where candidate covers >=50% state/controller cells; average available qualifying source scenes, report coverage. Stable canonical ties.',validation_labels='Do not parse until FROZEN.receipt.json; mandated two raw replay records are the explicit exception.',training_size_estimate='All CV models use the general10-seed rule; this yields230 fits rather than the rough120 estimate. No recipe tuning.'))
 (H/'ITERATION_LOG.md').write_text('# 实现与运行修正\n\n尚无运行修正。\n');(H/'GAPS.md').write_text('# 证据缺口\n\n执行中。\n');(H/'STATUS.md').write_text('# 当前状态\n\n开工条件通过，正在复制与重放。18CPU总限额，训练不超过14并发。\n')
 print('Opening audit complete; recent directories',len(parallel),'; validation labels unopened.',flush=True)
if __name__=='__main__':main()

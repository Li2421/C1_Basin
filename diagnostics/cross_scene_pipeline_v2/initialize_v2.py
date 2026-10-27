"""Preconditions and read-only baseline; prospective evidence is metadata only."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, subprocess, difflib
from concurrent.futures import ThreadPoolExecutor,as_completed

H=Path(__file__).resolve().parent
R=H.parents[1]
V=R/'diagnostics/cross_scene_pipeline_v1'
P=R/'diagnostics/cross_scene_prospective_v1'

def sha(p):
    p=Path(p).resolve()
    assert not p.is_relative_to(P), 'Reserved prospective evidence must not be opened'
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name,obj):
    (H/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def metadata(root):
    out={}
    for base,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=Path(base)/name;s=p.stat(follow_symlinks=False)
            out[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,mode=s.st_mode)
    return out

def recent_metadata(root,cutoff):
    """Stop at first recent stat: only directory registration is required."""
    stat=root.stat()
    if stat.st_mtime>=cutoff:return dict(path=str(root),mtime_ns=stat.st_mtime_ns,kind='directory')
    for base,dirs,files in os.walk(root,followlinks=False):
        for name in files+dirs:
            p=Path(base)/name;s=p.stat(follow_symlinks=False)
            if s.st_mtime>=cutoff:return dict(path=str(p),mtime_ns=s.st_mtime_ns,kind='stat_only')
    return None

def main():
    assert json.loads((V/'COMPLETE.json').read_text())['status']=='COMPLETE'
    assert json.loads((V/'INTEGRITY_COMPARE.json').read_text())['passed']
    assert not(H/'INTEGRITY_START.json').exists()
    now=datetime.now(timezone.utc);repos={}
    for repo in (R,R.parent/'02_C1_Toy_GiveWay'):
        status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).decode()
        diff=subprocess.check_output(['git','-C',str(repo),'diff']);cached=subprocess.check_output(['git','-C',str(repo),'diff','--cached'])
        repos[str(repo)]=dict(status=status,diff_sha256=hashlib.sha256(diff).hexdigest(),cached_diff_sha256=hashlib.sha256(cached).hexdigest())
    parallel={};cutoff=now.timestamp()-86400
    dirs=[d for d in(R/'diagnostics').iterdir()if d.is_dir()and d!=H]
    with ThreadPoolExecutor(max_workers=16)as pool:
        pending={pool.submit(recent_metadata,d,cutoff):d for d in dirs}
        for future in as_completed(pending):
            d=pending[future];recent=future.result()
            if recent:parallel[str(d)]=dict(recent_stat_evidence=recent)
            print('Metadata registered',d.name,bool(recent),flush=True)
    jobs=subprocess.check_output(['squeue','-h','-r','-a','-o','%i|%u|%T|%C|%R|%j|%Z'],text=True)
    for line in jobs.splitlines():
        vals=line.split('|',6)
        if len(vals)==7 and vals[2]=='RUNNING':parallel.setdefault(vals[6],dict(running_job=vals[0],user=vals[1]))
    write('INTEGRITY_START.json',dict(utc=now.isoformat(),repositories=repos,parallel_directories=list(parallel),parallel_metadata=parallel,
        slurm_jobs=jobs,prospective_access_policy='os.stat only; no opens or hashes',prospective_metadata=metadata(P),
        v1_root_metadata=dict(size=V.stat().st_size,mtime_ns=V.stat().st_mtime_ns),v1_protection='Hash every reused file; cache trees are not imported or modified.'))
    source=R/'docs/prompt_cross_scene_pipeline_v2_20261025.md'
    write('PROTOCOL_SOURCE.receipt.json',dict(path=str(source),sha256=sha(source),full_read=True))
    write('REUSED_EVIDENCE_SHA256_START.json',dict(files={str(source):sha(source),str(V/'COMPLETE.json'):sha(V/'COMPLETE.json'),str(V/'INTEGRITY_COMPARE.json'):sha(V/'INTEGRITY_COMPARE.json')}))
    write('EVIDENCE_SHA256_START.json',dict(files={str(R/'AGENTS.md'):sha(R/'AGENTS.md'),str(R/'shared_rollout_db/README.md'):sha(R/'shared_rollout_db/README.md')}))
    write('RESOURCE_POLICY.json',dict(total_cpu_shares=18,max_workers=17,coordinator_cpu_shares=1,gpu_shares=0,no_inactivity_cancellation=True))
    (H/'ITERATION_LOG.md').write_text('# 实现与运行修正记录\n\n尚无修正；仅在本轮目录实现新行为。\n')
    (H/'GAPS.md').write_text('# 证据缺口\n\n执行中；结尾更新。\n')
    (H/'STATUS.md').write_text('# 当前状态\n\n已完整阅读说明，开工条件通过；正在隔离重建与历史证据核对。\n')
    print('V2 preconditions passed; read-only snapshots recorded; reserved prospective data not opened.')

if __name__=='__main__':main()

"""Recovery audit: prune protected prefixes before traversal; metadata only for peers."""
from pathlib import Path
import os,json,hashlib,subprocess,datetime,time
H=Path(__file__).resolve().parent;R=H.parents[1];D=R/'diagnostics';MAC=R.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official';POC=R.parent/'Basin_C1_flow_field_poc_20261004/poc'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbidden(name):return (name.startswith('decentralized_') and name!='decentralized_pipeline_v1') or name.startswith('teacher_student_')
def stat(p):
 s=p.stat();return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
def run(args,cwd=R):return subprocess.check_output(args,cwd=cwd,text=True)
def capture(label):
 peers={};blocked={}
 # Directory names and stat only. Never enter prohibited directories.
 for p in D.iterdir():
  if not p.is_dir() or p==H:continue
  if forbidden(p.name):blocked[p.name]=stat(p)
  elif p.stat().st_mtime>=time.time()-86400:peers[p.name]=stat(p)
 # Conservative parallel inventory includes all recently modified diagnostics roots.
 sources={}
 for root in [R,POC,MAC]:
  for base,dirs,files in os.walk(root,followlinks=False):
   dirs[:]=[n for n in dirs if n not in ['.git','__pycache__','node_modules'] and not n.startswith('.venv') and Path(base)/n!=H and not (Path(base)==D and (forbidden(n) or n in peers))]
   for n in files:
    p=Path(base)/n
    if n.endswith('.py') and not p.is_symlink():sources[str(p)]=sha(p)
 exclusions=[':(exclude)diagnostics/decentralized_*/**',':(exclude)diagnostics/teacher_student_*/**']+[f':(exclude)diagnostics/{n}/**' for n in peers]
 def repo(p):return dict(head=run(['git','rev-parse','HEAD'],p).strip(),status=run(['git','status','--porcelain'],p),diff_sha256=hashlib.sha256(run(['git','diff','--','.']+(exclusions if p==R else []),p).encode()).hexdigest())
 v=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,prohibited_metadata_only=blocked,parallel_metadata_only=peers,parallel_classification='Conservative: allowed diagnostics root modified within24h; metadata only; prohibited prefix directories pruned before traversal',Basin=repo(R),MAC=repo(MAC),queue=run(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b']))
 p=H/f'INTEGRITY_{label}.json';assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n');print(label,len(sources),len(blocked),len(peers),flush=True)
if __name__=='__main__':
 import sys;capture(sys.argv[1])

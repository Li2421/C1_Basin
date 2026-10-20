from pathlib import Path
import hashlib,json,os,time,subprocess
R=Path('/home/zhihan/research/Basin_C1');H=Path(__file__).resolve().parent;O=R/'diagnostics/multiscene_generalization_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blocked(n):return n.startswith(('teacher_student_macflow_v1','teacher_student_line_closeout_v1','decentralized_blackbox_planners_v1','decentralized_fix_selection_v1','decentralized_fix_selection_confirm_v1','round30'))
def cmd(a,p):return subprocess.check_output(a,cwd=p,text=True)
def capture(label):
 sources={};parallel=set()
 roots=[R,R.parent/'Basin_C1_flow_field_poc_20261004/poc',R.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official']
 for root in roots:
  for base,dirs,files in os.walk(root):
   dirs[:]=[d for d in dirs if not blocked(d) and d not in ['.git','__pycache__','node_modules'] and not d.startswith('.venv') and Path(base)/d!=H]
   for n in files:
    p=Path(base)/n
    if p.is_symlink():continue
    if n.endswith('.py'):sources[str(p)]=sha(p)
    if p.is_relative_to(R/'diagnostics') and p.stat().st_mtime>time.time()-86400:parallel.add(p.relative_to(R/'diagnostics').parts[0])
 ex=[':(exclude)diagnostics/decentralized_blackbox_planners_v1/**',':(exclude)diagnostics/teacher_student_macflow_v1*/**',':(exclude)diagnostics/teacher_student_line_closeout_v1/**',':(exclude)diagnostics/round30*/**',':(exclude)diagnostics/decentralized_fix_selection_v1/**',':(exclude)diagnostics/decentralized_fix_selection_confirm_v1/**']
 def repo(p):return dict(head=cmd(['git','rev-parse','HEAD'],p).strip(),status=cmd(['git','status','--porcelain'],p),diff_sha256=hashlib.sha256(cmd(['git','diff','--','.']+(ex if p==R else []),p).encode()).hexdigest())
 v=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),Basin=repo(R),MAC=repo(roots[2]),sources=sources,parallel_task_directories=sorted(parallel),queue=cmd(['squeue','-u',str(os.getuid())],R))
 (H/f'INTEGRITY_{label}.json').write_text(json.dumps(v,indent=2)+'\n');print(label,len(sources),flush=True)
 return v
if __name__=='__main__':
 import sys
 capture(sys.argv[1])
 if sys.argv[1]=='BEGIN':
  names=['geometry.py','single_agent.py','observation.py','train_policies.py','policies.py','demonstrations.py','navigation_all.py','catalog.py','sequential_expert.py','core.py','common.py','registry.py','design_generated.py','submit_stage.py','runtime.py','controllers.py','trigger.py'];v=[]
  for n in names:
   (H/n).write_bytes((O/n).read_bytes());v.append(dict(source=str(O/n),sha256=sha(O/n),destination=str(H/n),copy_sha256=sha(H/n)))
  (H/'COPIED_SOURCES.json').write_text(json.dumps(v,indent=2)+'\n')
  for n in ['GEOMETRY_EQUIVALENCE.json','SINGLE_AGENT_COPY_PROVENANCE.json','SINGLE_AGENT_STEP.diff','REPRODUCTION_GATE.json','CATALOG_FROZEN.json']:(H/('ROUND39_'+n)).write_bytes((O/n).read_bytes())

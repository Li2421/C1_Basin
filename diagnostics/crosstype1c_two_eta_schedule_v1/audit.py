"""Independent stdlib-only audit; forbidden research trees are stat-only."""
from pathlib import Path
import os,json,hashlib,subprocess,datetime

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
POC=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
OFFICIAL=Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')
ALLOWED_CROSS={'crosstype1_learned_zoo_selector_v1','crosstype1b_generator_critic_pipeline_v1',HERE.name}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def forbidden(p):
    name=p.name
    return name.startswith(('decentralized_','teacher_student_','multiscene_')) or (name.startswith('crosstype') and name not in ALLOWED_CROSS)
def metadata(folder):
    out={}
    for base,dirs,files in os.walk(folder,followlinks=False):
        dirs[:]=[d for d in dirs if d not in ('.git','__pycache__')]
        for n in files:
            p=Path(base)/n
            try:s=p.stat();out[str(p)]=[s.st_size,s.st_mtime_ns]
            except FileNotFoundError:pass
    return out
def snapshot(opening=None):
    forbidden_dirs=sorted(p for p in (ROOT/'diagnostics').iterdir() if p.is_dir() and forbidden(p))
    excluded=set(forbidden_dirs)|{HERE}
    protected={}
    for parent in (ROOT,POC,OFFICIAL):
        for base,dirs,files in os.walk(parent,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.venv','node_modules') and Path(base)/d not in excluded]
            for n in files:
                p=Path(base)/n
                if n.endswith('.py') and p.is_file() and not any(p.resolve().is_relative_to(q) for q in excluded):protected[str(p)]=sha(p)
    previous_docs=set(opening['existing_document_sha256']) if opening else None
    new_prompts={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in (ROOT/'docs').rglob('prompt_*.md') if previous_docs is not None and str(p) not in previous_docs}
    repositories={}
    for parent in (ROOT,OFFICIAL):
        def git(*args):return subprocess.check_output(['git',*args],cwd=parent)
        exclusions=[f':(exclude){p.relative_to(ROOT)}' for p in excluded]+[f':(exclude){Path(p).relative_to(ROOT)}' for p in new_prompts] if parent==ROOT else []
        diff=git('diff','HEAD','--','.',*exclusions)
        repositories[str(parent)]=dict(head=git('rev-parse','HEAD').decode().strip(),status=git('status','--porcelain').decode(),diff_sha256=hashlib.sha256(diff).hexdigest(),excluded_paths=exclusions)
    documents={str(p):sha(p) for p in (ROOT/'docs').rglob('*') if p.is_file() and str(p) not in new_prompts}
    return dict(utc=utc(),protected_python_sha256=protected,existing_document_sha256=documents,user_added_prompts_stat_only=new_prompts,repositories=repositories,forbidden_metadata={str(p):metadata(p) for p in forbidden_dirs},parallel_metadata={},parallel_detection='Opening global squeue empty; forbidden trees always stat-only. Recent directory metadata retained separately.',queue=subprocess.check_output(['squeue','-h','-o','%i %u %T %C %b %j']).decode(),recent_diagnostic_directories={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in (ROOT/'diagnostics').iterdir() if p.is_dir() and p!=HERE and datetime.datetime.now().timestamp()-p.stat().st_mtime<86400})

if __name__=='__main__':
    assert not (HERE/'INTEGRITY_START.json').exists()
    result=snapshot();dump(HERE/'INTEGRITY_START.json',result)
    print('Opening audit complete:',len(result['protected_python_sha256']),'Python files;',len(result['existing_document_sha256']),'existing documents')

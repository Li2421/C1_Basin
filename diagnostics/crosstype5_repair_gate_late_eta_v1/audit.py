"""Input-list audit; all other diagnostic trees are stat-only parallel work."""
from pathlib import Path
import os,json,hashlib,datetime,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];D=ROOT/'diagnostics';POC=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004');OFFICIAL=Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def forbidden(p):return any(x.startswith(('decentralized_','teacher_student_','multiscene_')) for x in Path(p).parts)
def dump(name,value):(HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def inputs():
    c4=D/'crosstype4_no_repair_schedule_diagnosis_v1';c3=D/'crosstype3_unseen_controller_schedule_v1';g=D/'orthoflow3_generator_critic_v1';paths=set()
    req={c4:'REPORT.md FINAL_REVIEW.md DIAGNOSIS.md HANDOFF.md GAPS.md AXIS_TABLE.md b_candidates.py actions.py CANDIDATES.receipt.json TRAINING_ACTIONS.npz TRAINING_BUNDLE.npz FROZEN_PIPELINE.json inference.py NEW_EVAL_INPUTS.npz NEW_EVAL_CHOSEN_ACTIONS.json B_CONTROLLER_RESULTS.csv A_OBJECTS.json A_ALL_ACTION_OUTCOMES.csv phase_complete.py coordinator.py submit_aux.py resource_guard.py BASE_PROTOCOL.json TRAINING_STATES.json STATE_INPUTS.npz CANDIDATES.json CONTROLLER_FEATURES.npz BOOTSTRAP_SOURCES.json SELECTED_MODELS.json NEW_EVAL_STATES.json',c3:'ZOO_MANIFEST.csv UNSEEN_CONTROLLER_CONFIGS.json UNSEEN_CAPABILITY.json unseen_models.py new_execution.py progress.py STATE_INPUTS.npz CANDIDATES.json TRAINING_BUNDLE.npz UNSEEN_FEATURE_GAPS.json',g:'REPORT.md ARCHITECTURE_AUDIT.md train_evaluate.py normalization.json',D/'orthoflow3_generator_critic_frozen_test_v1':'run_frozen_test.py',D/'crosstype1_learned_zoo_selector_v1':'features.py'}
    for folder,names in req.items():paths.update(folder/n for n in names.split())
    paths.update(c4.glob('*_ALL_LOCAL_EQUIVALENCE.json'));paths.update((g/'generator/seed41').rglob('*'));paths={p for p in paths if p.is_file()}
    paths.update(Path(r['source']) for r in json.loads((HERE/'BOOTSTRAP_SOURCES.json').read_text()))
    # Historical source/dependency/weight inventory is explicit input, not a
    # traversal through unrelated diagnostics. No prohibited path is opened.
    paths.add(c4/'HISTORY_READ_INVENTORY.json')
    paths.update(Path(r['path']) for r in json.loads((c4/'HISTORY_READ_INVENTORY.json').read_text()))
    for name in ('HISTORY.json','TASK.json','TRANSFORM.json','COMPLETE.receipt.json'):paths.update((c4/'critic_fits').glob('*/*/'+name))
    frozen=json.loads((c4/'FROZEN_PIPELINE.json').read_text())
    def models(m):
        if isinstance(m,dict):
            for k,v in m.items():
                if k.endswith('_path') and isinstance(v,str):paths.add(Path(v))
                elif isinstance(v,(list,dict)):models(v)
        elif isinstance(m,list):
            for v in m:models(v)
    models(frozen)
    for parent in (ROOT,POC,OFFICIAL):
        for base,dirs,files in os.walk(parent):
            dirs[:]=[x for x in dirs if x not in ('.git','__pycache__','.venv','node_modules') and Path(base)/x not in (D,HERE)]
            paths.update(Path(base)/n for n in files if n.endswith('.py'))
    paths.update([ROOT/'AGENTS.md',ROOT/'shared_rollout_db/README.md',ROOT/'docs/generalization_axes_protocol_20261015.md',ROOT/'docs/prompt_crosstype5_repair_gate_and_late_eta_proposals_20261019.md'])
    assert all(not forbidden(p) for p in paths);assert all(p.exists() for p in paths),[str(p) for p in paths if not p.exists()]
    return sorted(paths)
def stat_tree(folder):
    out={}
    for base,dirs,files in os.walk(folder):
        dirs[:]=[x for x in dirs if x not in ('.git','__pycache__')]
        for n in files:
            p=Path(base)/n
            try:s=p.stat();out[str(p)]=[s.st_size,s.st_mtime_ns]
            except FileNotFoundError:pass
    return out
def snapshot(start=None):
    paths=[Path(p) for p in start['input_sha256']] if start else inputs();hashes={str(p):sha(p) if p.exists() else None for p in paths};old_docs=set(start['existing_document_sha256']) if start else None
    new={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in (ROOT/'docs').rglob('prompt_*.md') if old_docs is not None and str(p) not in old_docs};docs={str(p):sha(p) for p in (ROOT/'docs').rglob('*') if p.is_file() and str(p) not in new};metadata={str(p):stat_tree(p) for p in D.iterdir() if p.is_dir() and p!=HERE};repos={}
    # Exclude diagnostics completely in git diff: input files there have their
    # exact hashes above; unrelated diagnostic directories may change freely.
    for parent in (ROOT,OFFICIAL):
        excluded=[':(exclude)diagnostics']+[f':(exclude){Path(p).relative_to(ROOT)}' for p in new] if parent==ROOT else []
        run=lambda *args:subprocess.check_output(['git',*args],cwd=parent)
        repos[str(parent)]=dict(head=run('rev-parse','HEAD').decode().strip(),status=run('status','--porcelain').decode(),diff_sha256=hashlib.sha256(run('diff','HEAD','--','.',*excluded)).hexdigest())
    return dict(utc=utc(),input_sha256=hashes,existing_document_sha256=docs,user_new_prompts_stat_only=new,diagnostic_metadata_stat_only=metadata,repositories=repos,queue=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%j|%T|%C|%b']).decode(),parallel_policy='All non-input diagnostic files are stat-only; changes never alter the input hash comparison; prohibited prefixes always stat-only')
if __name__=='__main__':
    assert not (HERE/'INTEGRITY_START.json').exists();r=snapshot();dump('INTEGRITY_START.json',r);dump('INPUT_MANIFEST.json',dict(paths=list(r['input_sha256']),parallel_stat_only_directories=list(r['diagnostic_metadata_stat_only'])));print('Opening audit:',len(r['input_sha256']),'explicit protected inputs; all other diagnostic content stat-only',flush=True)

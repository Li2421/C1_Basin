"""Explicit protected inputs; non-input/forbidden diagnostics get stat only, on disk."""
from pathlib import Path
import os,json,hashlib,datetime,subprocess,sqlite3
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];D=ROOT/'diagnostics'
POC=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004');OFFICIAL=Path('/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def forbidden(p):return any(s.startswith(('decentralized_','teacher_student_','multiscene_')) for s in Path(p).parts)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);assert p.resolve().is_relative_to(HERE);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
    tmp.replace(p)

def declare():
    old=D/'crosstype5_repair_gate_late_eta_v1';paths=set(map(Path,read(old/'INPUT_MANIFEST.json')['paths']))
    for addition in sorted(old.glob('INPUT_ADDITIONS_*.json')):paths.update(Path(r['path']) for r in read(addition)['files'])
    paths.add(old/'INPUT_MANIFEST.json');paths.update(old.glob('INPUT_ADDITIONS_*.json'))
    required={old:'REPORT.md HANDOFF.md GAPS.md COVERAGE.md COVERAGE_RESULTS.json AXIS_TABLE.md ITERATION_LOG.md FROZEN_PIPELINE.json TRAINING_BUNDLE.npz TRAINING_STATES.json BASE_PROTOCOL.json TRAINING_INPUTS.npz TRAINING_ADDED_ACTIONS.json CONTROLLER_FEATURES.npz TRAINING_BUNDLE.receipt.json DELIVERY.receipt.json RESOURCE_RELEASE_POLICY.json USER_EXECUTION_RULES.md RESOURCE_RELEASE_WATCH_COMPLETE.json evaluation_CASES.json evaluation_protocol.json training_CASES.json training_protocol.json REQUERY_EQUIVALENCE_v3.json EARLY_TERMINATION_EQUIVALENCE_v3.json',D/'crosstype4_no_repair_schedule_diagnosis_v1':'REPORT.md DIAGNOSIS.md HANDOFF.md TRAINING_BUNDLE.npz A_ALL_ACTION_OUTCOMES.csv',D/'crosstype3_unseen_controller_schedule_v1':'ZOO_MANIFEST.csv UNSEEN_CAPABILITY.json UNSEEN_FEATURE_GAPS.json',D/'crosstype1_learned_zoo_selector_v1':'features.py',D/'crosstype2_mixed_zoo_selector_v1':'PROBE_MANIFEST.json probes.py probe_prefix.py assemble_features.py',D/'axis4a_geometry_horizon_v1':'progress.py',D/'orthoflow3_generator_critic_v1':'REPORT.md ARCHITECTURE_AUDIT.md train_evaluate.py normalization.json',D/'orthoflow3_generator_critic_frozen_test_v1':'run_frozen_test.py'}
    for folder,names in required.items():paths.update(folder/n for n in names.split())
    paths.update(Path(r['source']) for r in read(HERE/'BOOTSTRAP_SOURCES.json'))
    for p in ('AGENTS.md','docs/generalization_axes_protocol_20261015.md','shared_rollout_db/README.md','docs/prompt_crosstype6_shared_correction_and_controller_information_20261019.md'):paths.add(ROOT/p)
    def model_paths(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k.endswith('_path') and isinstance(v,str):paths.add(Path(v))
                elif isinstance(v,(dict,list)):model_paths(v)
        elif isinstance(x,list):
            for v in x:model_paths(v)
    model_paths(read(old/'FROZEN_PIPELINE.json'))
    for parent in (ROOT,POC,OFFICIAL):
        for base,dirs,files in os.walk(parent):
            dirs[:]=[n for n in dirs if n not in ('.git','__pycache__','.venv','node_modules') and Path(base)/n!=D]
            paths.update(Path(base)/n for n in files if n.endswith('.py'))
    assert all(not forbidden(p) for p in paths)
    absent=[str(p) for p in paths if not p.exists()];assert not absent,absent
    write(HERE/'INPUT_MANIFEST.json',dict(utc=utc(),paths=sorted(str(p.resolve()) for p in paths),non_input_diagnostic_policy='stat-only; content never opened or hashed',parallel_boundary='decentralized_*, teacher_student_*, multiscene_* always stat-only'))
    return sorted(paths)

def metadata(path):
    assert path.resolve().is_relative_to(HERE) and not path.exists();db=sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA cache_size=-32768')
    db.execute('CREATE TABLE file(path TEXT PRIMARY KEY, root TEXT, size INTEGER, mtime_ns INTEGER) WITHOUT ROWID');batch=[];count=0;directories={};recent_cut=int((datetime.datetime.now(datetime.timezone.utc).timestamp()-86400)*1e9)
    for folder in sorted(D.iterdir()):
        if not folder.is_dir() or folder==HERE:continue
        nfiles=0;latest=0
        for base,dirs,files in os.walk(folder):
            dirs[:]=[n for n in dirs if n not in ('.git','__pycache__')]
            for name in files:
                p=Path(base)/name
                try:s=p.stat()
                except FileNotFoundError:continue
                batch.append((str(p),str(folder),s.st_size,s.st_mtime_ns));nfiles+=1;latest=max(latest,s.st_mtime_ns)
                if len(batch)>=2000:db.executemany('INSERT INTO file VALUES (?,?,?,?)',batch);count+=len(batch);batch=[]
        directories[str(folder)]=dict(files=nfiles,latest_mtime_ns=latest,recent_24h=latest>=recent_cut,content_stat_only=True,forbidden_prefix=forbidden(folder))
    if batch:db.executemany('INSERT INTO file VALUES (?,?,?,?)',batch);count+=len(batch)
    db.commit();db.close();return dict(file=str(path),rows=count,directories=directories)

def start():
    assert not (HERE/'INTEGRITY_START.json').exists();paths=declare()
    protected={str(p):sha(p) for p in paths};docs={str(p):sha(p) for p in (ROOT/'docs').rglob('*') if p.is_file()};repos={}
    for parent in (ROOT,OFFICIAL):
        run=lambda *args:subprocess.check_output(['git',*args],cwd=parent)
        repos[str(parent)]=dict(head=run('rev-parse','HEAD').decode().strip(),status=run('status','--porcelain').decode(),diff_sha256=hashlib.sha256(run('diff','HEAD','--','.',*([':(exclude)diagnostics'] if parent==ROOT else []))).hexdigest())
    meta=metadata(HERE/'AUDIT_METADATA_START.sqlite')
    result=dict(utc=utc(),input_sha256=protected,existing_document_sha256=docs,repositories=repos,diagnostic_metadata=meta,parallel_directories_stat_only=[p for p,r in meta['directories'].items() if r['recent_24h']],queue=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%j|%T|%C|%b']).decode(),only_new_directory_writes=True,prohibited_content_never_opened=True)
    write(HERE/'INTEGRITY_START.json',result);write(HERE/'INTEGRITY_START.receipt.json',dict(utc=utc(),files={str(p):sha(p) for p in (HERE/'INTEGRITY_START.json',HERE/'INPUT_MANIFEST.json',HERE/'audit_scope.py')},metadata_sqlite_rows=meta['rows'],metadata_database_is_only_stat_not_shared_rollout_DB=True))
    print('Opening audit complete:',len(paths),'protected inputs;',meta['rows'],'stat-only diagnostic files',flush=True)
if __name__=='__main__':start()

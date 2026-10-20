"""Read-only audit and final receipts for the blocked third attempt."""
from pathlib import Path
import json,hashlib,subprocess,datetime
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
start=read(H/'INTEGRITY_START.json');end={'utc':now}
for name,p in [('Basin_C1',R),('MACFlow_Official',R.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official')]:
 def g(*a):return subprocess.check_output(['git',*a],cwd=p)
 end[name]=dict(HEAD=g('rev-parse','HEAD').decode().strip(),status=g('status','--porcelain').decode(),diff_sha256=hashlib.sha256(g('diff')).hexdigest())
f=R.parent/'Basin_C1_flow_field_poc_20261004/poc';end['poc_py']={str(p):sha(p) for p in sorted(f.rglob('*.py'))};write('INTEGRITY_END.json',end)
comparison={n:{k:start[n][k]==end[n][k] for k in ['HEAD','status','diff_sha256']} for n in ['Basin_C1','MACFlow_Official']};comparison['poc_py_unchanged']=start['poc_py']==end['poc_py'];passed=all(all(comparison[n].values()) for n in ['Basin_C1','MACFlow_Official']) and comparison['poc_py_unchanged'];write('INTEGRITY_COMPARISON.json',dict(passed=passed,comparisons=comparison))
source_files={}
for p in [H/'GATE_IMPORTED_SOURCES.json',*H.glob('imports_data_*.json')]:source_files.update(read(p)['imported_sources'])
for p in [R/'shared_rollout_db/src/merger.py',R/'shared_rollout_db/src/historical_ingest.py',R/'shared_rollout_db/src/cache_writer.py',R/'shared_rollout_db/src/planner.py',R/'shared_rollout_db/src/integration.py',R/'shared_rollout_db/src/cache_hook.py']:
 source_files[str(p)]=sha(p)
write('SOURCE_HASHES.json',dict(utc=now,files=source_files,unchanged_since_recorded=all(Path(p).exists() and sha(p)==v for p,v in source_files.items()),cancelled_data_worker_project_dependencies='Same read-only teacher(), RingExchangeEnv and local_observation imports as completed data workers and noise gate; no new project module loaded by state iteration.'))
a=read(H/'ATTEMPT2_ARCHIVE.json');assert all(sha(H/'attempt2_gate_tolerance'/n)==v for n,v in a['files'].items())
a1=read(H/'attempt2_gate_tolerance/ATTEMPT1_ARCHIVE.json');assert all(sha(H/'attempt1_blocked'/n)==v for n,v in a1['sha256'].items())
assert sha(H/'PREREGISTRATION.md')==sha(H/'attempt2_gate_tolerance/PREREGISTRATION.md');assert sha(H/'PREREGISTRATION.receipt.json')==sha(H/'attempt2_gate_tolerance/PREREGISTRATION.receipt.json')
write('ARCHIVE_VERIFIED.json',dict(utc=now,attempt1_unchanged=True,attempt2_unchanged=True,original_preregistration_unchanged=True))
files=['shared_rollout_db/src/merger.py','shared_rollout_db/src/historical_ingest.py','shared_rollout_db/src/cache_writer.py','diagnostics/decentralized_pipeline_v1/design.py']
write('DB_INTERFACE_AUDIT.json',dict(utc=now,status='BLOCKED_EXACT_IDENTITY_REGISTRATION_WRITE_AUTHORIZATION',sources={p:sha(R/p) for p in files},
 required='Register new student checkpoint, pipeline, arms and pilot state exact execution identities before rollout preflight',
 allowed_outside_directory_writes=['append_journal to default shared_rollout_db/journals/<experiment_uid>','python -m shared_rollout_db.src.merger --experiment <experiment_uid>'],
 evidence=[dict(file=files[0],lines=[9,11],finding='Official merger creates the fixed Ingestor and calls ingest_jsonl; no supplied registration callback or metadata-only entry point.'),dict(file=files[1],lines=[137,141,142,143,145,147,148,149,150,151],finding='registered_exact_uid_v1 validates preexisting scenario/state/controller metadata; absent identity returns ambiguous, does not register it.'),dict(file=files[1],lines=[197],finding='Metadata-only journal rows without success/outcome/termination are skipped as ambiguous.'),dict(file=files[3],finding='round28 registers new exact identities using direct INSERT SQL in design.py; that write path is outside the two allowed methods in round30 section7.')],
 legacy_ingest_note='Legacy implicit registration exists, but infers/replaces scenario, physical-state and controller profiles rather than preserving the planned registered_exact_uid_v1 identity. It is not a generic metadata registration interface. No fabricated outcomes or imprecise/ambiguous identities used to bootstrap registration.',
 writes_performed=False,rollout_preflight_run=False,training_started=False))
ready=sorted(str(p.relative_to(H)) for p in (H/'data').glob('*/READY.json'))
write('RESOURCE_AUDIT.json',dict(utc=now,noise_gate_cpu_cores=1,data_job='21340',data_array_tasks=14,data_cpu_cores_per_task=1,GPU_shards_used=0,training_job_submitted=False,cancel_requested=True,ready_data=ready,remaining_data='Cancelled partial outputs preserved; no training data or closed-loop evidence deleted',maximum_concurrent_cpu_shards=14))
write('results.json',dict(status='BLOCKED_EXACT_IDENTITY_REGISTRATION_WRITE_AUTHORIZATION',attempt=3,source_integrity_passed=passed,teacher_checkpoints_verified=14,training_manifest_digests_matched=14,noise_gate={k:v for k,v in read(H/'NOISE_GATE.json').items() if k not in ['checks','failure']},training_started=False,new_rollouts=0,db_writes=0,TS1='NOT_RUN',TS2='NOT_RUN',TS3a='NOT_RUN',TS3b='NOT_RUN',TS4='NOT_RUN',TS5='NOT_RUN',delivery_complete=False))
print(json.dumps(dict(source_integrity_passed=passed,noise_gate=read(H/'NOISE_GATE.json')['status'],ready_data=len(ready),archives_unchanged=True)))

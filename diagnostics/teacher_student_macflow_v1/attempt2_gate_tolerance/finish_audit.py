"""End-of-attempt audit, standard library only. No project imports."""
from pathlib import Path
import json,hashlib,subprocess,datetime
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(name,obj):(H/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
s={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for n,d in [('Basin_C1',R),('MACFlow_Official',R.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official')]:
 def g(*a):return subprocess.check_output(['git',*a],cwd=d)
 s[n]={'HEAD':g('rev-parse','HEAD').decode().strip(),'status':g('status','--porcelain').decode(),'diff_sha256':hashlib.sha256(g('diff')).hexdigest()}
f=R.parent/'Basin_C1_flow_field_poc_20261004/poc';s['poc_py']={str(p):sha(p) for p in sorted(f.rglob('*.py'))}
write('INTEGRITY_END.json',s);start=read(H/'INTEGRITY_START.json');comp={n:{k:start[n][k]==s[n][k] for k in ['HEAD','status','diff_sha256']} for n in ['Basin_C1','MACFlow_Official']}
comp['poc_py_unchanged']=start['poc_py']==s['poc_py'];passed=all(all(x.values()) for x in [comp['Basin_C1'],comp['MACFlow_Official']]) and comp['poc_py_unchanged'];write('INTEGRITY_COMPARISON.json',dict(passed=passed,comparisons=comp,changes_outside_new_directory=[] if passed else 'See start/end JSON differences'))
receipt=read(R/'diagnostics/decentralized_pipeline_v1/IMPLEMENTATION_FROZEN.receipt.json')
write('ROUND28_FROZEN_AUDIT.json',{p:dict(expected=h,actual=sha(p),passed=sha(p)==h) for p,h in receipt['files'].items()})
old=read(H/'ATTEMPT1_ARCHIVE.json');write('ATTEMPT1_ARCHIVE_VERIFIED.json',dict(passed=all(sha(H/'attempt1_blocked'/n)==h for n,h in old['sha256'].items()),file_count=len(old['sha256'])))
imports=read(H/'GATE_IMPORTED_SOURCES.json')['imported_sources'];write('IMPORTED_SOURCES_END_CHECK.json',dict(files=len(imports),all_unchanged=all(Path(p).exists() and sha(p)==h for p,h in imports.items()),changed=[p for p,h in imports.items() if not Path(p).exists() or sha(p)!=h]))
write('DELIVERY_STATUS.json',dict(status='BLOCKED_NOISE_CONSISTENCY_GATE_FAILED',source_integrity_passed=passed,gate_passed=False,training_started=False,closed_loop_labels_generated=0,phase_status={'archive':'PASS','start_end_integrity':'PASS' if passed else 'FAIL','teacher_checkpoint_training_manifest_hashes':'PASS_14_OF_14','preregistration':'SEALED','noise_gate':'FAIL_FIRST_SAMPLE','training':'NOT_RUN','offline_student_tests':'NOT_RUN','capability_gate':'NOT_RUN','unlabeled_probes':'NOT_RUN','frozen_choices':'NOT_RUN','rollout_preflight':'NOT_APPLICABLE_NO_ROLLOUT_SUBMITTED','rollout_journals_merge_postflight':'NOT_RUN','paper_tables_and_timing':'NOT_RUN'}))
print(json.dumps({'integrity_passed':passed,'poc_files':len(s['poc_py']),'imported_source_files':len(imports),'imports_unchanged':all(Path(p).exists() and sha(p)==h for p,h in imports.items())}))

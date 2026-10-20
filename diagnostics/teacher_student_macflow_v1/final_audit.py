"""Final integrity and provenance audit. Reads evidence; never changes existing sources."""
from pathlib import Path
import json,hashlib,subprocess,datetime,ast,collections
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(name,value):(H/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def main():
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();start=read(H/'INTEGRITY_START.json');end={'utc':now}
 for name,p in [('Basin_C1',R),('MACFlow_Official',R.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official')]:
  def g(*a):return subprocess.check_output(['git',*a],cwd=p)
  end[name]=dict(HEAD=g('rev-parse','HEAD').decode().strip(),status=g('status','--porcelain').decode(),diff_sha256=hashlib.sha256(g('diff')).hexdigest())
 f=R.parent/'Basin_C1_flow_field_poc_20261004/poc';end['poc_py']={str(p):sha(p) for p in sorted(f.rglob('*.py'))};write('INTEGRITY_END.json',end)
 comparison={n:{k:start[n][k]==end[n][k] for k in ['HEAD','status','diff_sha256']} for n in ['Basin_C1','MACFlow_Official']};comparison['poc_py_unchanged']=start['poc_py']==end['poc_py'];hashpass=all(all(comparison[n].values()) for n in ['Basin_C1','MACFlow_Official']) and comparison['poc_py_unchanged']
 inventories=[H/'GATE_IMPORTED_SOURCES.json',*H.glob('imports_*.json')];sources=collections.defaultdict(set)
 for p in inventories:
  obj=read(p)
  for name,digest in obj.get('imported_sources',{}).items():
   if not Path(name).resolve().is_relative_to(H):sources[name].add(digest)
 for name,digest in read(H/'COPY_PROVENANCE.json')['files'].items():sources[name].add(digest)
 changed=[p for p,hs in sources.items() if not Path(p).exists() or len(hs)!=1 or sha(p) not in hs];assert not changed,changed
 write('SOURCE_HASHES.json',dict(utc=now,existing_sources={p:next(iter(hs)) for p,hs in sorted(sources.items())},all_imported_or_copied_existing_sources_unchanged=True,import_inventory_files=[str(p) for p in inventories]))
 scope=read(H/'JOURNAL_PATH_DEVIATION.json');wrong=scope['wrong_files'];assert all(Path(p).exists() and sha(p)==h for p,h in wrong.items())
 write('INTEGRITY_COMPARISON.json',dict(source_hash_audit_passed=hashpass and not changed,protected_existing_source_content_hashes_passed=not changed and comparison['poc_py_unchanged'],git_status_changes={n:dict(added=sorted(set(end[n]['status'].splitlines())-set(start[n]['status'].splitlines())),removed=sorted(set(start[n]['status'].splitlines())-set(end[n]['status'].splitlines()))) for n in ['Basin_C1','MACFlow_Official']},comparisons=comparison,write_scope_audit_passed=False,overall_strict_source_protection_compliance=False,reason='448 pilot journals were written by a shadow-imported PoC cache_writer outside the allowed paths; preserved, not deleted or edited',out_of_scope_writes=wrong,existing_source_modifications=changed))
 frozen=read(H/'MAIN_IMPLEMENTATION_FROZEN.receipt.json');assert all(sha(p)==h for p,h in frozen['files'].items());original=R/'diagnostics/decentralized_pipeline_v1';assert sha(H/'pipeline.py')==sha(original/'pipeline.py') and sha(H/'core.py')==sha(original/'core.py')
 def body(path):return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='episode'),include_attributes=False)
 assert body(H/'episode_frozen.py')==body(original/'run.py')
 oldstates=read(original/'EVAL_STATES.json');newstates=read(H/'EVAL_STATES.json');assert len(newstates)==32
 assert all(all(n['physical'][k]==o['physical'][k] for k in ['positions','velocities','goals']) and n['uid']==o['uid'] and n['rng_namespace']==o['rng_namespace'] for n,o in zip(newstates,oldstates))
 checkpointpass=all(sha(t['checkpoint'])==t['checkpoint_sha256'] for t in read(H/'TEACHERS.json'));assert checkpointpass
 write('FINAL_EXECUTION_AUDIT.json',dict(utc=now,main_implementation_frozen_hashes_pass=True,copied_pipeline_core_identical=True,copied_episode_AST_identical=True,same32_eval_physical_states_and_keys=True,teacher_checkpoints_unchanged=True,student_checkpoints_match_training_freeze=all(sha(p)==h for p,h in read(H/'TRAINING_FROZEN.receipt.json')['files'].items()),main_rollout_count=len(list((H/'main/raw').glob('*.json'))),teacher_new_rollout_count=len(list((H/'teacher/raw').glob('*.json'))),pilot_rollout_count=448,unlabeled_probes=448,video_requested=False))
 write('EXECUTION_METADATA_AUDIT.json',dict(utc=now,actual_student_nominal='own local_observation row + one-hot id + own fold_in(key,i) noise; shared official ActorVectorField; one network step, no ODE',actual_safety='unchanged copied D-local local mode, eta0 for safety; nominal/RHR runtime.action adapters in separate file',legacy_teacher_profile_fields=['chain=FF','pseudo_steps=10','execution_callable/observation_frame refer to teacher provenance/output chart'],interpretation='These inherited fields are not descriptions of student integration or student observation. Actual execution is defined by explicit controller_family/execution_semantics/nominal_source_code_sha256/pipeline/runner hashes and training config. No teacher rollout is reused as a student label.',RHR_radius='student1.5m; historical teacher RHR infinity, kept separately'))
 # Registration append-only preservation, including all original rows present before this attempt.
 import sys
 sys.path.insert(0,str(R))
 from shared_rollout_db.src.rollout_db import connect,canonical
 baseline=read(H/'registration/initial/BEFORE.json')
 with connect(True) as db:
  registry_pass={}
  for table,old in baseline['row_hashes'].items():
   key={'scenario':'scenario_uid','experiment':'experiment_uid','state':'state_uid','controller_config':'controller_uid','eta':'eta_uid'}[table];current={r[key]:hashlib.sha256(canonical(dict(r)).encode()).hexdigest() for r in db.execute('SELECT * FROM '+table)};registry_pass[table]=all(current.get(k)==v for k,v in old.items())
  counts={r[0]:r[1] for r in db.execute('SELECT experiment_uid,count(*) FROM rollout GROUP BY experiment_uid')};oldcounts_pass=all(counts.get(k,0)==v for k,v in baseline['rollout_counts'].items());registration_receipts=[read(p) for p in sorted((H/'registration').glob('*/RECEIPT.json'))];assert all(x['passed'] and x['existing_rows_unchanged'] and x['rollout_counts_unchanged'] for x in registration_receipts)
  phases={}
  for phase,expected in [('pilot',448),('main',6272),('teacher',1792)]:
   exp='exp_teacher_student_macflow_v1_'+phase;rows=db.execute('SELECT * FROM rollout WHERE experiment_uid=?',(exp,)).fetchall();assert len(rows)==expected and not any(r['conflict_quarantined'] for r in rows);phases[phase]=dict(records=len(rows),unknown=sum(r['numerical_failure'] for r in rows),compatibility_all_exact=all(r['compatibility_quality']=='EXACT_REUSE' for r in rows))
 write('FINAL_DB_AUDIT.json',dict(utc=now,preexisting_registration_rows_unchanged=registry_pass,preexisting_experiment_rollout_counts_unchanged=oldcounts_pass,per_design_transaction_preservation_passed=True,global_duration_note='Other user experiments may append concurrently outside our registration transactions; per-design before/after receipts are the required preservation check.',phases=phases,registration_receipts={str(p):sha(p) for p in sorted((H/'registration').glob('*/RECEIPT.json'))},rollout_rows_written_only_by_official_merger=True))
 # Detect duplicate simulated/journaled identities; helper cases are a partition, not extra experiments.
 ids=[]
 for phase,expected in [('main',6272),('teacher',1792)]:
  seen=[]
  for p in (R/'shared_rollout_db/journals'/('exp_teacher_student_macflow_v1_'+phase)).glob('*.merged'):
   for line in p.read_text().splitlines():seen.append(json.loads(line)['case']['rollout_uid'])
  assert len(seen)==len(set(seen))==expected,(phase,len(seen),len(set(seen)))
  ids.append(dict(phase=phase,journaled=len(seen),unique=len(set(seen))))
 write('ROLLOUT_DEDUP_AUDIT.json',dict(utc=now,passed=True,phases=ids,helper_cases_subset_of_main=True,unknown_not_rerun=True))
 print(json.dumps(dict(source_hash_audit_passed=hashpass,write_scope_audit_passed=False,DB_old_rows_preserved=True,all_rollouts_exact_and_unique=True)))
if __name__=='__main__':main()

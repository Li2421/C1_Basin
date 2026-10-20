from support import *
from flax import serialization
import subprocess,time
G=guard()
def main():
 prereg();checks={};existing=HERE/'DELIVERY.receipt.json'
 if existing.exists():
  for p,h in read(existing)['files'].items():assert sha(p)==h,p
  print('Existing sealed delivery verified; no evidence modified.',flush=True);return
 (HERE/'STOP_RESOURCE_WATCH').write_text(utc()+'\n')
 for i in range(30):
  if (HERE/'RESOURCE_SUMMARY.json').exists():break
  time.sleep(1)
 resources=read(HERE/'RESOURCE_SUMMARY.json');checks['resource_monitor_within_cap']=resources['peak_cpu']<=18 and resources['peak_gpu_shards']<=2 and resources['violations']==0
 inventories=list(HERE.glob('imports_*.json'));union={};conflicts=[]
 for p in inventories:
  r=read(p)
  for path,h in r['imported_sources'].items():
   if path in union and union[path]!=h:conflicts.append(path)
   union[path]=h
 changed=[p for p,h in union.items() if not Path(p).is_file() or sha(p)!=h];begin=read(HERE/'INTEGRITY_BEGIN.json');research_uncovered=[p for p in union if (Path(p).is_relative_to(FIELD) or Path(p).is_relative_to(ROOT)) and not Path(p).is_relative_to(HERE) and p not in begin['files']];write(HERE/'IMPORT_SOURCE_AUDIT.json',dict(passed=not changed and not conflicts and not research_uncovered,imported_files=len(union),sources=union,changed=changed,conflicting_inventory_hashes=conflicts,research_sources_missing_begin_snapshot=research_uncovered,guard=guard()));checks['all_imported_sources_unchanged']=not changed and not conflicts and not research_uncovered
 audit=read(HERE/'INTEGRITY_COMPARISON.json');checks['protected_integrity']=audit['passed'];external=audit['changed'];end=read(HERE/'INTEGRITY_END.json');attribution=dict(utc=utc(),global_integrity_passed=audit['passed'],existing_external_changes=[dict(path=p,begin_sha256=begin['files'].get(p),end_sha256=end['files'].get(p),current_sha256=sha(p),imported=p in union,written_by_this_task=False,author_independently_identified=False) for p in external],new_external_files=audit['new_external_files'],action='Experimental pipeline stopped at failed integrity gate; no external file restored or modified; existing frozen results retained and reported with qualification.');write(HERE/'EXTERNAL_CHANGE_ATTRIBUTION.json',attribution);checks['external_change_unimported_outside_reused_results']=all(p not in union and '/diagnostics/multiscene_generalization_v1/' in p for p in external);checks['supplemental_dependency_integrity']=read(HERE/'DEPENDENCY_COMPARISON.json')['passed'];checks['DB_package_and_default_journals']=Path(G['package']).resolve().is_relative_to(ROOT/'shared_rollout_db') and Path(G['journals']).resolve()==ROOT/'shared_rollout_db/journals';checks['x64_false']=not G['jax_enable_x64'];gate=read(HERE/'REPLAY_GATE.json');checks['sixteen_bitwise_replays']=gate['passed'] and gate['completed']==16
 for r in read(HERE/'COPY_AUDIT.json'):
  checks.setdefault('copied_frozen_execution_unchanged',True)
  if Path(r['destination']).name in ['worker.py','pipeline.py','runtime_new.py','fair_rhr.py','student.py','astar.py','controllers.py','legacy_controllers.py','gaussian.py']:checks['copied_frozen_execution_unchanged'] &= r['byte_identical'] and sha(r['destination'])==r.get('destination_sha256',r['sha256'])
 checks['all_receipts_valid']=True
 for receipt in HERE.rglob('*.receipt.json'):
  for p,h in read(receipt).get('files',{}).items():assert sha(p)==h,(receipt,p)
 labels=np.load(HERE/'LABELS.npz');y=labels['labels'];checks['complete_six_arm_tensor']=y.shape==(46,6,32,2) and np.isin(y,[-1,0,1]).all();checks['diagnostic_independent_audit']=read(HERE/'DIAGNOSTIC_INDEPENDENT_AUDIT.json')['passed'];checks['all_features_frozen']=len(read(HERE/'DIAGNOSTIC_FEATURES.json'))==46 and not read(HERE/'DIAGNOSTIC_GAPS.json');checks['diagnostic_no_task_journal']=all(not read(p)['task_journal_written'] and read(p)['complete'] for p in (HERE/'diagnostic/raw').glob('*.json'));checks['diagnostic_count']=len(list((HERE/'diagnostic/raw').glob('*.json')))==736
 checks['only_missing_executed']=len(list((HERE/'supplement/raw').glob('*.json')))==7936 and sum(read(p)['new'] for p in (HERE/'supplement').glob('worker*.json'))==7936;pf=read(HERE/'ALL_CACHE_PREFLIGHT.json')['summary'];post=read(HERE/'ALL_CACHE_POSTFLIGHT.json')['summary'];checks['official_pre_postflight']=pf['exact_reusable']==9728 and pf['genuinely_missing']==7936 and post['genuinely_missing']==int((y==-1).sum()) and post['ambiguous']==0 and post['exact_reusable']==17664-int((y==-1).sum());checks['registration_preserved_existing_rows']=read(HERE/'registration/supplement/RECEIPT.json')['existing_rows_unchanged'];checks['official_merger']=read(HERE/'supplement_MERGE_RECEIPT.json')['records']==7936
 checks['LOFO_source_only']=True;checks['critic_architecture_and_seed_step_protocol']=True;choices_time=[]
 for family in FAMILIES:
  dest=HERE/'folds'/family;z=np.load(dest/'SOURCE_ONLY.npz');assert family not in z['families'];assert len(z['held_names'])==(14 if family in ['ST','TD'] else 6);assert set(z.files)=={'features','labels','families','names','held_features','held_names'}
  st=read(dest/'STEP_SELECTION.json');assert st['selected_step']==list(range(500,12001,500))[int(np.argmin(st['mean_source_validation_nll']))];fits=list((dest/'full').glob('seed*/fit.json'));assert len(fits)==3 and {read(p)['seed'] for p in fits}=={17,23,41};assert len(list((dest/'cv').glob('*/seed*/fit.json')))==12
  for seed in [17,23,41]:
   path=dest/'full'/f'seed{seed}'/f"step{st['selected_step']}.msgpack";params=serialization.msgpack_restore(path.read_bytes())['params'];expected=[(13,64),(64,64),(64,16),(16,1)];assert [params[f'Dense_{i}']['kernel'].shape for i in range(4)]==expected;assert all(v.dtype==np.float32 for d in params.values() for v in d.values())
  receipt=read(dest/'CHOICES_FROZEN.receipt.json');assert not receipt['held_labels_read'];choices_time.append(receipt['utc'])
 checks['historical_teacher_features_preselection']=read(HERE/'TEACHER_HISTORICAL_FEATURES_FROZEN.receipt.json')['utc']<min(choices_time);checks['freeze_chronology']=read(HERE/'DIAGNOSTICS_FROZEN.receipt.json')['utc']<min(choices_time) and max(choices_time)<read(HERE/'ANALYSIS.receipt.json')['utc'];r=read(HERE/'results.json');checks['DEVELOPMENT_only']=r['evidence_level']=='DEVELOPMENT';checks['five_thousand_paired_bootstraps']=r['bootstrap']['replicates']==5000 and r['bootstrap']['paired'] and r['bootstrap']['seeds_clustered'];checks['reports_and_figures']=all((HERE/p).is_file() for p in ['REPORT.md','HANDOFF.md','CONTROLLER_SIX_ARM_TABLE.csv','figures/diagnostic_distributions.pdf','figures/selection_success.pdf','figures/held_family_success.pdf']);checks['full_names_first_paragraph']=all(FULL[f] in (HERE/'REPORT.md').read_text().split('\n\n')[0] for f in FAMILIES)
 write(HERE/'FINAL_VERIFICATION.json',dict(utc=utc(),passed=all(checks.values()),checks=checks,guard=guard()));assert all(v for k,v in checks.items() if k!='protected_integrity'),checks
 paths=[HERE/p for p in ['PREREGISTRATION.md','PREREGISTRATION.receipt.json','COPY_AUDIT.json','INTEGRITY_BEGIN.json','INTEGRITY_END.json','INTEGRITY_COMPARISON.json','IMPORT_SOURCE_AUDIT.json','DEPENDENCY_BASELINE.json','DEPENDENCY_END.json','DEPENDENCY_COMPARISON.json','DIAGNOSTICS_FROZEN.receipt.json','LABELS_FROZEN.receipt.json','ALL_CACHE_PREFLIGHT.json','ALL_CACHE_POSTFLIGHT.json','supplement_MERGE_RECEIPT.json','CONTROLLER_SIX_ARM_TABLE.csv','results.json','REPORT.md','HANDOFF.md','FINAL_VERIFICATION.json','RESOURCE_SUMMARY.json','EXTERNAL_CHANGE_ATTRIBUTION.json','TEACHER_HISTORICAL_FEATURES_FROZEN.receipt.json','HISTORICAL_PROTOCOL_OLD_CRITIC_COMPARATOR.json']]+[HERE/'folds'/f/'CHOICES_FROZEN.receipt.json' for f in FAMILIES]+list((HERE/'figures').glob('*'))
 seal('DELIVERY.receipt.json',paths,passed=all(checks.values()),research_evidence_validation_passed=all(v for k,v in checks.items() if k!='protected_integrity'),status='COMPLETE' if all(checks.values()) else 'QUALIFIED_REPORT_GLOBAL_INTEGRITY_NOT_PASSED',all_development=True,guard=guard());print('QUALIFIED DELIVERY: global integrity remains failed; research evidence checks follow',checks,flush=True)
if __name__=='__main__':main()

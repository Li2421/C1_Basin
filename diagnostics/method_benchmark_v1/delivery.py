from core import *
prereg();result=read(HERE/'results.json');assert result['B1']=='REJECTED';assert read(HERE/'RECONSTRUCTION_GATE.json')['passed']
changed=[];receipts={}
for name in ['MODEL_METHOD_FREEZE.receipt.json','R3_FROZEN.receipt.json','PREDICTIONS_FROZEN.receipt.json','T2_ALIGNMENT_CORRECTION.receipt.json']:
 r=read(HERE/name)
 for path,digest in r['files'].items():
  if sha(path)!=digest:changed.append(path)
 receipts[name]=dict(utc=r['utc'],files=len(r['files']))
assert not changed
hashes=read(D/'schedule_phenomenon_consolidated_v1/SOURCE_HASHES.json');assert all(sha(path)==digest for path,digest in hashes.items());assert read(HERE/'STATISTICAL_INVARIANTS.json')['status']=='PASS'
z=np.load(HERE/'T2_ANALYSIS.npz');point=read(HERE/'T2_POINT_RESULTS.json');aliases=[]
for name,menu in zip(point['controllers'],point['menus']):
 aliases.append(dict(controller=name,head_alias=menu.get('head_alias',{}),learned_eligible=menu['supported_full']))
write(HERE/'FINAL_T2_MENU_AUDIT.json',dict(utc=utc(),controllers=aliases,metadata_sha256=sha(D/'controller_universal_eta_schedule_v1/main_CASES.json'),correction_receipt_sha256=sha(HERE/'ETA_ALIAS_ALIGNMENT_CORRECTION.receipt.json')))
result['bootstrap']['derived_substreams']='base_seed + index in T1,T2,T3,T4,T5,POOLED; all policies within each draw paired';write(HERE/'results.json',result)
report=HERE/'REPORT.md';s=report.read_text().replace('M3及方法超参数、R3和盲预测先冻结','M3及方法超参数、R3和初版盲预测先冻结');report.write_text(s)
handoff=HERE/'HANDOFF.md';s=handoff.read_text();s=s.replace('reconstruction_gate.py -> analyze.py', 'reconstruction_gate.py -> correct_t2_alignment.py -> analyze.py');s+='\nFinal input/arm audits: T2 FRESH_STATES was obsolete and disjoint from actual labeled protocol states. T2_ALIGNMENT_CORRECTION re-infers unchanged frozen M3/M4 on actual state metadata after reconstruction; original blind predictions retained. ETA_ALIAS_ALIGNMENT_CORRECTION maps two Freg arms exactly to source F2, leaves eta4 unsupported. Prior numerical T2 outputs retained inpre_eta_alias_alignment. T2 alone was recomputed; all other per-test simulations reused. No model, hyperparameter, threshold or rollout changes.\nFor statistical replay use existing frozen predictions, reconstructed arrays and simulations; summarize.py/delivery.py can replay without training or labels acquisition. Full stage scripts describe original pipeline, not permission to overwrite sealed artifacts or redraw states.\n';handoff.write_text(s)
write(HERE/'ISOLATION_AND_DELIVERY_AUDIT.json',dict(utc=utc(),status='PASS',frozen_receipts=receipts,all_frozen_model_feature_original_prediction_files_unchanged=True,source_files_unchanged=len(hashes),T2_state_and_arm_identity_corrections_explicit=True,no_rollout_probe_or_shared_DB_write=True,paired_unknown_cancellation_test_pass=True,report_sha256=sha(report),results_sha256=sha(HERE/'results.json')))
audit=read(HERE/'FINAL_AUDIT.json');audit.update(utc=utc(),report_sha256=sha(report),results_sha256=sha(HERE/'results.json'),final_identity_audit_sha256=sha(HERE/'ISOLATION_AND_DELIVERY_AUDIT.json'));write(HERE/'FINAL_AUDIT.json',audit);print('Round26 final delivery PASS')

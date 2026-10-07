#!/usr/bin/env python3
from __future__ import annotations
import csv,json,hashlib,shutil,time,math
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_analytic_basin_margin_learning_v1'
def load(p):return json.load(open(p))
def rows(p):return list(csv.DictReader(open(p)))
def writecsv(p,rr,fields=None):
 rr=list(rr);fields=fields or (list(rr[0]) if rr else ['status'])
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
specs={
 'ncb_affine':('ncb_affine','ncb_affine',11),
 'racs':('racs','racs',20),
 'rfse':('rfse','rfse_44',12),
 'two_lobe':('two_lobe','two_lobe',22)}
summary=[]
for outname,(dirname,var,complexity) in specs.items():
 d=HERE/dirname;gate=load(d/f'gate_{var}.json');metrics=rows(d/f'cached_precision_recall_{var}.csv');neighbor=rows(d/f'neighbor_target_coverage_{var}.csv')
 shutil.copyfile(d/f'cached_precision_recall_{var}.csv',d/'cached_precision_recall.csv');shutil.copyfile(d/f'neighbor_target_coverage_{var}.csv',d/'neighbor_target_coverage.csv');shutil.copyfile(d/f'gate_{var}.json',d/'gate.json')
 (d/'complexity.json').write_text(json.dumps({'family':outname,'variant':var,'meaningful_scalar_parameters_per_state':complexity,'membership':'finite analytic inequalities intersected with E_bridge','training_time_lookup':False},indent=2)+'\n')
 writecsv(d/'val_validation.csv',[r for r in metrics if r['split']=='val']);writecsv(d/'fresh_inside_validation.csv',[{'status':'SKIPPED_CACHED_GEOMETRY_GATE_FAILED','new_eta':0,'confirmed_non_B63':0}])
 gate['fresh_validation_status']='SKIPPED_CACHED_GEOMETRY_GATE_FAILED';gate['final_gate_pass']=False;(d/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
 summary.append({'family':outname,'variant':var,'cached_retained_false_inclusion':gate['cached_retained_false_inclusion'],'median_full_B63_recall':gate['val_median_full_B63_recall'],
  'states_full_recall_ge_0_40':gate['val_states_full_recall_ge_0_40'],'median_retained_B63_recall':gate['val_median_retained_B63_recall'],
  'median_neighbor_target_coverage':gate['val_median_neighbor_target_coverage'],'nondegenerate_states':gate['val_nondegenerate_states'],
  'train_universal_fraction':gate['train_universal_retained_max_fraction'],'train_sets_fitted':gate['train_sets'],'complexity_scalars':complexity,'gate_pass':False})

(HERE/'candidate_protocol.md').write_text('''# Analytic Basin candidate protocol\n\nAll eta geometry is evaluated in the frozen normalized `E_bridge` coordinates and intersected with the 14 authoritative domain halfspaces. TEST states are quarantined. State-specific fits use exact-Q64 fitting positives and all exact non-B63 points as hard negative evidence; provenance-frozen B63 points are held out for recall. Sparse VAL acquisition used the nine robust target modes frozen by the prior 40-state dataset: even mode indices were fitting probes and odd mode indices were held out, plus one deterministic point-search holdout/state.\n\nAll fits use deterministic constrained grids. The optimization priority is zero retained false inclusion, retained fitting-positive coverage, full fitting-positive coverage, then volume/complexity. Erosion is frozen at `gamma_tan=0.85`, `gamma_norm=0.75`. A candidate that fails any cached recall, precision, nondegeneracy, multimodal-coverage, or universal-constant gate is not subjected to fresh retained-interior validation.\n\nCandidate order is NCB-AFFINE, RACS, RFSE `(2,2)/(4,2)/(4,4)`, then TWO_LOBE. RFSE `(4,4)` is the reported family representative because it had the highest VAL full recall among its three frozen exponent variants; none passed.\n''')

# Property diagnosis and synthesis records.
syn=[]
for name,complexity,definition in [
 ('clipped_ncb',19,'NCB-AFFINE intersected with at most two state-specific halfspaces u_l^T eta_tilde >= tau_l; retained clips use +0.025 normalized margin.'),
 ('rap',12,'Rotated asymmetric parallelotope: lo <= R^T(eta_tilde-c) <= hi; retained tangential bounds shrink by 0.85 and normal bounds by 0.75.'),
 ('two_rap',24,'Union of at most two RAP components fitted by deterministic two-cluster partition; each component is eroded independently.')]:
 d=HERE/'synthesized_forms'/name;g=load(d/f'gate_{name}.json');syn.append({'family':name,'analytic_definition':definition,'complexity_scalars':complexity,'gate':g})
(HERE/'property_diagnosis.md').write_text(f'''# Property diagnosis after A/B/C/D failure\n\nAll four predeclared families failed before fresh-interior validation. NCB-AFFINE was closest: it obtained full recall {summary[0]['median_full_B63_recall']:.3f} and retained recall {summary[0]['median_retained_B63_recall']:.3f}, but had {summary[0]['cached_retained_false_inclusion']} cached VAL false inclusion, neighbor-target coverage {summary[0]['median_neighbor_target_coverage']:.3f}, and universal retained coverage {summary[0]['train_universal_fraction']:.3f}. RACS curvature did not help; RFSE remained low-recall; splitting into two NCB lobes reduced recall.\n\nThe observed property is a precision/coverage conflict, not absence of robust eta. Smooth centered supports must globally shrink to avoid sharp local negatives, thereby discarding distant robust modes. Existing 88/96 successful interpolation and 16/16 hole probes argue against pervasive holes, while the new VAL neighbor-mode probes (17/43 B63) show strong state dependence and sharp mode-specific exclusion. The missing capability appeared to be sharp asymmetric clipping rather than more curvature.\n\nThree permitted synthesis cycles tested: (1) a clipped NCB with up to two analytic halfspaces; (2) a rotated asymmetric parallelotope for sharp planar/asymmetric bounds; and (3) a union of two such parallelotopes for limited nonconvexity. None passed the unchanged Section 10 gates. RAP reached full median recall 0.618, but only 5/8 states reached 0.40, retained recall was 0.333, neighbor-target coverage 0.25, and a common retained eta covered 21/22 fitted TRAIN sets. This demonstrates that simply enlarging a sharp-edged family recreates universal-constant degeneracy.\n\nNo further form is synthesized: the three-cycle limit is exhausted, and additional analytic flexibility would be unsupported by held-out evidence.\n''')
probe=[]
for r in rows(HERE/'exact_q64_inventory.csv'):
 if 'val_neighbor_' in r['sources']:
  probe.append({'state_id':r['state_id'],'eta1':r['eta1'],'eta2':r['eta2'],'eta3':r['eta3'],'Q64':r['Q64'],'B63':r['B63'],'role':r['evidence_role'],'source':r['sources']})
writecsv(HERE/'targeted_property_probes.csv',probe)
(HERE/'synthesis_decision.json').write_text(json.dumps({'cycles_used':3,'maximum_cycles':3,'forms':syn,'any_passed':False,'decision':'STOP_WITHOUT_TRAINING'},indent=2)+'\n')

# Selected Basin and training/evaluation stop artifacts.
selected=HERE/'selected_basin';selected.mkdir(exist_ok=True)
(selected/'selected_family.json').write_text(json.dumps({'selected_family':None,'reason':'No predeclared or synthesized analytic family passed all Section 10 gates.'},indent=2)+'\n')
(selected/'fitting_protocol.json').write_text(json.dumps({'coordinate_system':'frozen normalized eta','domain':'E_bridge','gamma_tan':.85,'gamma_norm':.75,'test_quarantined':True,'fitting_priority':['zero retained false inclusion','retained positive coverage','full positive coverage','volume','complexity']},indent=2)+'\n')
writecsv(selected/'retained_set_parameters.csv',[],['state_id','family','parameters_json'])
universal={r['family']:{'fitted_train_sets':r['train_sets_fitted'],'max_retained_fraction':r['train_universal_fraction']} for r in summary}
for s in syn:universal[s['family']]={'fitted_train_sets':s['gate']['train_sets'],'max_retained_fraction':s['gate']['train_universal_retained_max_fraction']}
(selected/'universal_solution_audit.json').write_text(json.dumps(universal,indent=2)+'\n')
failed=[]
for r in summary:
 reasons=[]
 if r['cached_retained_false_inclusion']>0:reasons.append('cached_retained_false_inclusion')
 if r['median_full_B63_recall']<.60:reasons.append('median_full_recall')
 if r['states_full_recall_ge_0_40']<6:reasons.append('states_recall_ge_0.40')
 if r['median_retained_B63_recall']<.35:reasons.append('median_retained_recall')
 if r['median_neighbor_target_coverage']<.60:reasons.append('neighbor_target_coverage')
 if r['train_universal_fraction']>.75:reasons.append('universal_constant_degeneracy')
 failed.append({'family':r['family'],'failed_criteria':reasons})
rep_gate={'status':'FAIL','selected_family':None,'predeclared_candidate_results':summary,'failed_criteria':failed,'fresh_inside_validation_run':False,'training_allowed':False}
(selected/'representation_gate.json').write_text(json.dumps(rep_gate,indent=2)+'\n')

train=HERE/'training';train.mkdir(exist_ok=True)
(train/'common_model_config.json').write_text(json.dumps({'status':'NOT_RUN_REPRESENTATION_GATE_FAILED','architecture':[214,128,128,3],'seeds':[17,23,41]},indent=2)+'\n')
for seed in (17,23,41):
 d=train/f'seed{seed}';d.mkdir(exist_ok=True);(d/'NOT_RUN.json').write_text(json.dumps({'reason':'representation_gate_failed'},indent=2)+'\n')
writecsv(train/'training_summary.csv',[{'status':'NOT_RUN','reason':'representation_gate_failed'}]);(train/'selected_checkpoint.json').write_text(json.dumps({'selected_checkpoint':None},indent=2)+'\n');(train/'checkpoint_sha256.txt').write_text('NOT_CREATED\n');(train/'collapse_audit.json').write_text(json.dumps({'status':'NOT_RUN'},indent=2)+'\n')
ev=HERE/'evaluation';ev.mkdir(exist_ok=True)
for name in ['val_closedloop.csv','test_predictions.csv','test_q64.csv','controller_comparison.csv','rescue_break.csv','fresh_wide_manifest.csv','fresh_wide_results.csv']:
 writecsv(ev/name,[{'status':'NOT_RUN_REPRESENTATION_GATE_FAILED'}])
(ev/'fresh_wide_manifest.json').write_text(json.dumps({'status':'NOT_RUN_REPRESENTATION_GATE_FAILED','episodes':0},indent=2)+'\n')
# Top-level compatibility copies matching the declared artifact names.
for name in ['selected_family.json','fitting_protocol.json','retained_set_parameters.csv','representation_gate.json','universal_solution_audit.json']:
 shutil.copyfile(selected/name,HERE/name)
for name in ['common_model_config.json','training_summary.csv','selected_checkpoint.json','checkpoint_sha256.txt','collapse_audit.json']:
 shutil.copyfile(train/name,HERE/name)
for name in ['val_closedloop.csv','test_predictions.csv','test_q64.csv','controller_comparison.csv','rescue_break.csv','fresh_wide_manifest.csv','fresh_wide_results.csv']:
 shutil.copyfile(ev/name,HERE/name)
shutil.copyfile(ev/'fresh_wide_manifest.json',HERE/'fresh_wide_manifest.json')
for seed in (17,23,41):
 d=HERE/f'seed{seed}';d.mkdir(exist_ok=True);shutil.copyfile(train/f'seed{seed}'/'NOT_RUN.json',d/'NOT_RUN.json')

# Runtime and final decision/report.
rts=[]
for p in (HERE/'raw/val_neighbor_q64').glob('*_runtime.json'):rts.append(load(p))
runtime={'existing_exact_q64_state_eta_reused_before_new':798,'test_exact_q64_records_quarantined':271,'new_exact_q64_eta':43,'new_continuations':sum(r['new_continuations'] for r in rts),'physical_steps':sum(r['physical_steps'] for r in rts),'rollout_critical_wall_seconds':max(r['wall_seconds'] for r in rts),'worker_wall_seconds_sum':sum(r['wall_seconds'] for r in rts),'max_gpu_shards':6,'gpu_memory_mib_per_worker_approx':614,'cpu_threads_max_total':12,'fresh_inside_validation_continuations':0,'training_seconds':0,'test_continuations':0}
(HERE/'runtime_statistics.json').write_text(json.dumps(runtime,indent=2)+'\n')
decision={'classification':'ANALYTIC_BASIN_REPRESENTATION_FAILS','analytic_low_complexity_basin_representation_existed_under_gates':False,'basin_margin_supervision_solved_point_target_ambiguity':False,'representation_gate':'FAIL','training_executed':False,'test_executed':False,'fresh_wide_executed':False,'next_step':'Collect a small state-held-out set of candidate-aware exact-Q64 boundary/interior probes that identifies state-specific sharp exclusions, then reassess representability without changing gates.'}
(HERE/'final_decision.json').write_text(json.dumps(decision,indent=2)+'\n')

tab='\n'.join(f"- {r['family']}: false={r['cached_retained_false_inclusion']}, full recall={r['median_full_B63_recall']:.3f}, retained recall={r['median_retained_B63_recall']:.3f}, neighbor coverage={r['median_neighbor_target_coverage']:.3f}, universal={r['train_universal_fraction']:.3f}, complexity={r['complexity_scalars']}, FAIL" for r in summary)
report=f'''# OrthoFlow3 analytic Basin search and margin learning v1\n\n## Decision\n\n**ANALYTIC_BASIN_REPRESENTATION_FAILS**. The representation gate failed, so no network, held-out TEST controller evaluation, or Fresh-WIDE cohort was run.\n\n## Evidence and acquisition\n\nThe audit reused 798 TRAIN/VAL exact-Q64 state–eta records (703 B63, 95 non-B63). All 271 historical TEST records were quarantined and omitted from fitting and selection. Because seven VAL states had inadequate independent multimodal evidence, 43 frozen neighbor-target eta probes were evaluated directly at Q64: 17 were B63 and 26 non-B63. This cost {runtime['new_continuations']:,} continuations and {runtime['physical_steps']:,} physical steps. The final selection inventory contains 841 exact TRAIN/VAL records.\n\n## Predeclared candidates\n\n{tab}\n\nRFSE reports the best of its frozen exponent variants, `(4,4)`; `(2,2)` and `(4,2)` also failed. No candidate reached all simultaneous requirements: zero retained false inclusion, full median recall >=0.60, 6/8 states >=0.40, retained median recall >=0.35, neighbor coverage >=0.60, and TRAIN universal coverage <=0.75. Therefore fresh retained-interior validation was not scientifically warranted.\n\n## Property diagnosis and synthesis\n\nThe main failure was a precision–coverage conflict around sharp, state-dependent exclusions. Three low-complexity forms were synthesized under the unchanged gates:\n\n1. **Clipped NCB (19 scalars):** NCB-AFFINE intersected with at most two learned halfspaces. Full recall 0.526, retained recall 0.224, neighbor coverage 0.500; FAIL.\n2. **RAP (12 scalars):** rotated asymmetric parallelotope. Full recall 0.618, retained recall 0.333, but only 5/8 states met recall 0.40, neighbor coverage 0.250, and universal retained coverage 0.955; FAIL.\n3. **TWO_RAP (24 scalars):** union of two independently eroded RAP components. Full recall 0.333, retained recall 0, neighbor coverage 0.583, universal coverage 0.800; FAIL.\n\nThe synthesis limit was exhausted. Adding more flexibility without new held-out boundary evidence would not be scientifically justified.\n\n## Scientific interpretation\n\n- Basin existence remains established: robust persistent eta values exist for every labeled true-t0 state.\n- Low-complexity analytic representability was **not established under the frozen gates**.\n- Precision could often be made perfect, but only by losing too much independent/multimodal recall.\n- Enlarging sets improved recall but recreated a near-universal retained eta, which would again permit constant-output margin-loss collapse.\n- Since no label representation passed, margin-loss learnability and closed-loop generalization were not tested. This is not evidence that Basin supervision itself fails.\n\n## Explicit answers\n\n- Did an analytic low-complexity Basin representation exist? **Not among the four predeclared and three justified synthesized families under the required precision/recall/degeneracy gates.**\n- Did Basin margin supervision solve point-target ambiguity? **Unresolved; training was correctly blocked before this question could be tested.**\n\n## Next step\n\nCollect a small, state-held-out candidate-aware exact-Q64 set focused on sharp exclusion boundaries and retained-interior coverage, then reassess whether a low-complexity conditional inequality family is identifiable. Do not train until the unchanged representation gate passes.\n'''
report=report.replace('\n## Scientific interpretation','\nNo representation was accepted, so retained set-label counts are TRAIN 0 and VAL 0; no checkpoint or held-out controller metrics exist.\n\n## Scientific interpretation')
(HERE/'final_report.md').write_text(report)

# Integrity manifest for scientific products (raw files remain referenced but unhashed here).
products=[]
for p in sorted(HERE.rglob('*')):
 if p.is_file() and 'raw/' not in str(p.relative_to(HERE)) and 'runs/' not in str(p.relative_to(HERE)) and p.name!='manifest.json':products.append({'path':str(p.relative_to(HERE)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(HERE/'manifest.json').write_text(json.dumps({'experiment':'ORTHOFLOW3_ANALYTIC_BASIN_SEARCH_AND_MARGIN_LEARNING_V1','authoritative_basis_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','test_quarantined':True,'network_training':False,'files':products},indent=2)+'\n')
print(json.dumps({'decision':decision,'predeclared':summary,'runtime':runtime},indent=2))

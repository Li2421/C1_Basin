"""Immutable experimental handoff: no checkpoint or outcome changes."""
import csv,json,hashlib,platform,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from .train import OUT,OLD,ROOT,FOLDS,KINDS,load,dump,sha
def read(name):return list(csv.DictReader((OUT/name).open()))
def write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 metrics=read('loso_results.csv');frozen=load(OUT/'models_frozen.json');pred=load(OUT/'target_predictions.json');feature=load(OUT/'feature_audit.json');lat=read('deployment_latency.csv')
 summaries=[];shuffle=[];stability=[];safety=[]
 truth=load(OLD/'cached_truth.json')
 for fold in FOLDS:
  by={r['method']:r for r in metrics if r['fold']==fold}
  score=np.asarray(pred['folds'][fold]['scores']['source_selected']);idx=score.argmax(1)
  safety.append({'fold':fold,'method':'source_selected','cached_selected_collision_count':sum(r['collision'][int(j)] for r,j in zip(truth[fold],idx)),
   'selected_unresolved_seed_slots':int(round(sum(16*(r['upper'][int(j)]-r['lower'][int(j)]) for r,j in zip(truth[fold],idx)))),
   'safety_semantics_changed':False,'new_full_rollouts':0})
  for method in ['C0','eta_only','C1_full','C1_additive','C3_full','source_selected','oracle']:
   summaries.append({**by[method],'selected_context':frozen['folds'][fold]['source_selected_kind'] if method=='source_selected' else ''})
  for kind in KINDS:
   b=[int(by[f'{kind}_seed{s}']['B15']) for s in (17,23,41)]
   stability.append({'fold':fold,'kind':kind,'B15_seeds_17_23_41':str(b),'mean':float(np.mean(b)),'std':float(np.std(b)),'median':float(np.median(b))})
   if kind.endswith('additive'):continue
   a=np.array(pred['folds'][fold]['scores'][kind]);b=np.array(pred['folds'][fold]['scores'][kind+'_context_shuffle'])
   c=np.array(pred['folds'][fold]['logits'][kind]);d=np.array(pred['folds'][fold]['logits'][kind+'_context_shuffle'])
   shuffle.append({'fold':fold,'kind':kind,'top1_agreement':float(np.mean(a.argmax(1)==b.argmax(1))),
      'mean_absolute_p_change':float(abs(a-b).mean()),'mean_absolute_logit_change':float(abs(c-d).mean()),
      'mean_within_state_rank_correlation':float(np.nanmean([spearmanr(x,y).statistic for x,y in zip(c,d)])),
      'B15_correct_context':int(by[kind]['B15']),'B15_wrong_context':int(by[kind+'_context_shuffle']['B15']),
      'C3_caveat':'response depends on eta; mismatch is not a pure controller-identity intervention' if kind=='C3_full' else ''})
 write('primary_summary.csv',summaries);write('seed_stability.csv',stability);write('context_shuffle_effect.csv',shuffle)
 write('safety_audit.csv',safety)
 ref=list(csv.DictReader((OLD/'target_supervised_references.csv').open()));write('target_supervised_references.csv',ref)
 budget={'new_full_continuations':0,'new_success_labels':0,'generator_modified':False,
   'response_cache_files_including_superseded_and_noise_diagnostics':len(list((OUT/'response_cache').glob('*.json'))),
   'response_cache_scope':'derived short-response features, NOT Q labels; cached by physical state, exact eta, checkpoint, runtime/protocol hash',
   'source_response_records':305+47486,'target_response_records':sum(len(r['state_uids'])*17 for r in pred['folds'].values()),
   'maximum_response_steps_per_call':3,'maximum_physical_horizon_seconds':.15,
   'model_runs_used':36,'superseded_model_runs':36,'all_success_label_artifacts_reused':True,
   'Q_database_writes':0,'postflight':'no new full continuation or Q label; journal/merger not applicable to feature cache',
   'cache_preflight':{f:load(OUT/f/'cache_preflight.json')['summary'] for f in FOLDS}}
 dump(OUT/'budget_audit.json',budget)
 b=load(OUT/'deployment_benchmark.json');b.update(context_measurement_steps=4*105*3*17,
    scope='first frozen state per scenario; repeated-input latency jitter, not cohort-wide tail bound; generator proposal generation excluded')
 b['cpu_hardware']=json.loads(subprocess.check_output(['lscpu','-J'],text=True));dump(OUT/'deployment_benchmark.json',b)
 states=load(OLD/'states.json');integrity=[]
 for fold in FOLDS:
  source=[r for r in states if r['scenario']!=FOLDS[fold]]
  a={(r['scenario'],r['family']) for r in source if r['split']=='train'};v={(r['scenario'],r['family']) for r in source if r['split']=='validation'}
  assert not a&v
  for kind in KINDS:
   n=load(OUT/fold/kind/'normalization.json');assert FOLDS[fold] not in n['source_scenes']
   for run in frozen['folds'][fold][kind]['runs']:
    assert FOLDS[fold] not in run['sources'] and not run['target_labels_used'];assert sha(run['checkpoint'])==run['sha256']
  integrity.append({'fold':fold,'source_train_val_family_overlap':len(a&v),'excluded_target_scene':FOLDS[fold],'target_input_replay':load(OUT/f'target_context_{fold}.json')['input_replay_matches_frozen'],
    'target_invalid_context':load(OUT/f'target_context_{fold}.json')['invalid']})
 dump(OUT/'integrity_audit.json',{'folds':integrity,'labels_sha':sha(OLD/'all_pairs.parquet'),
   'inherited_frozen_target_truth_sha':sha(OLD/'cached_truth.json'),'target_pool_changed':False,
   'target_labels_for_training_normalization_or_checkpoint_selection':False,'precision_issue_fixed_before_target_scoring':True,
   'source_probe_incomplete':{'solver_failures':3,'early_terminal_snapshots':2,'labels_dropped':0},
   'strictness_caveat':'fixed descriptors motivated by historical controller-swap evidence; no claim of target-naive research design or generator zero-shot'})
 decision={'classification':'CONTROLLER_CONTEXT_HELPFUL_BUT_NOT_TRANSFERABLE',
  'minimal_information_sufficient_context_found':False,'known_aliases_separated':'C1 and C3 each 64/64; this is not Q sufficiency',
  'state_aware_stably_beats_eta_only_LOSO':False,'Ring_source_selected_B15':1,'Ring_N':60,'Ring_eta_only_B15':27,
  'Ring_severe_high_confidence_FP':59,'source_selected_by_fold':{f:frozen['folds'][f]['source_selected_kind'] for f in FOLDS},
  'missing_controller_information_established_by_previous_swap':True,'missing_short_context_major_LOSO_cause_supported':False,
  'remaining_bottleneck':'UNDERRESOLVED: coarse/short noisy proxy versus joint physical/controller support versus learned non-invariant source correlations',
  'object':'Q(h,eta,C_controller) and B(h,C_controller); tested low-dimensional c is only a proxy for C, not a demonstrated sufficient statistic',
  'generator':'unchanged; gate for conditioning it on this c did not pass','new_full_continuations':0,
  'no_target_selected_winner':'DB C1 24/24 is secondary; source VAL chose C3 12/24. C1 seeds 24/13/24, context shuffle leaves selected seed at24/24.'}
 dump(OUT/'final_decision.json',decision)
 lines=['Controller-context LOSO experiment — completed','',
 'Conclusion: no tested minimal context repairs cross-scene feasibility selection. The correct object remains controller-conditioned, but a 0.15-second response summary is not shown to be sufficient.',
 '', 'Representations (fixed before validation):',
 'C0: inherited unified h + eta, partial-count pure-NLL shared critic.',
 'C1: eta-independent nominal Flow + safety, three steps; average steps1/2 into ten physical scalars (one identically zero correction field).',
 'C3: same ten scalars after candidate-specific OrthoFlow3 correction; three steps per eta.',
 'Features: goal-parallel mean/min, signed lateral mean/std, pair-closing mean/max, speed, safety intervention, executed correction magnitude, action change. One validity flag; no scene/checkpoint IDs.',
 'Critics keep entity encoder and128/64 trunk widths. C1 additive A(h,c)+B(eta) is a control; C3 was not called additive because c already depends on eta.',
 '', 'Scientific result: source-only selected models (B15 / frozen states):']
 for fold in FOLDS:
  by={r['method']:r for r in summaries if r['fold']==fold}
  def fmt(m):return f"{by[m]['B15']}/{by[m]['N']}"+(f" +{by[m]['unknown']} unresolved" if int(by[m]['unknown']) else '')
  lines.append(f"{fold}: C0={fmt('C0')}; eta-only={fmt('eta_only')}; selected {frozen['folds'][fold]['source_selected_kind']}={fmt('source_selected')}; oracle={fmt('oracle')}")
 lines+=['','Do not cherry-pick DB C1: 24/24 for the VAL-selected C1 seed, but seed results24/13/24; context shuffle still24/24. The source-only candidate gate instead selected C3,12/24. This is not robust causal evidence for controller conditioning.',
 'Ring: C1 seeds0/0/0; C3 seeds0/0/1. Selected C3 still has59 severe p>0.9 / Q16 upper<=0.5 errors. Eta-only27/60. Selected shared vs eta-only rescue0/break26, paired exact p=2.98e-8 (descriptive unadjusted).',
 'Four-Way23+1 unresolved cannot be called24/24 or an improvement over C0 purely from confirmed counts; retain numerical intervals. Logit-safe ranking diagnostic does not rescue these conclusions.',
 '', 'Necessary versus sufficient:',
 'Both descriptors separate all64 original controller-swap pairs. Median feature distance C1=.0383, C3=.0368; distance versus absolute delta-Q correlations .045 / -.208. Separation alone does not identify the direction of success change.',
 'Four source-family-fold ridge diagnostic on128 controller-condition examples: C0 NLL=.933; C1=.907; C3=.878; eta-only=.621. Large-delta controller ordering8/12 for C1/C3 vs baseline ties=.5. This small diagnostic is not sufficient to claim controller-response learning.',
 'Four independent probe-noise streams show between-controller signal and within-controller response noise of comparable scale (median signal/noise1.28/1.34). The main experiment used one fixed independent noise stream; it is not an expected-response oracle.',
 '', 'Source use versus transfer:',
 'Mean selected source VAL observed-trial NLL: C0=.1554; C1=.1465; C3=.1440 (exact values in source_heldout_diagnostics.csv). C1 context shuffle~.153; C3 mismatched-response shuffle~.375; additive~.365.',
 'C3 is used on source data, but its shuffle changes eta-conditioned response as well as controller information. It is not a pure controller-identity causal intervention. On Ring, context shuffle leaves B15 at1/60, top1 agreement91.7%; C1 agreement95%.',
 'Ring C1: every target state outside some source-context coordinate range; five coordinates outside on average. C3 is mostly within marginal ranges, yet fails. Scalar range/OOD alone is insufficient: joint state/controller/eta support and long-horizon response remain untested.',
 '', 'Root-cause adjudication:',
 'Established: current h omits future frozen-controller information (prior matched swap). Established here: short response distinguishes identity but does not yield transferable selection.',
 'Supported: source-conditioned relations are predictive within seen regimes yet extrapolate incorrectly; coarse/noisy short summaries and source physical/controller support are plausible limitations.',
 'UNDERRESOLVED: sufficiency of this compressed c versus learning/invariance failure. No controlled equal-input/equal-c contradictory-Q evidence was found here, so do not call the augmented map information-theoretically impossible.',
 'Minimal follow-up, not executed: source-family-only response-horizon/noise-repeat discrimination against held-out controller-swap labels before any further LOSO. Do not tune on these four target results.',
 '', 'Deployment (CPU, batch16 neural scoring; C3 physical probes currently serial):']
 for r in lat:
  if r['part']=='end_to_end':lines.append(f"{r['fold']} {r['kind']}: p50={float(r['p50_ms']):.2f} ms, p95={float(r['p95_ms']):.2f} ms; hypothetical {float(r['p50_theoretical_Hz']):.1f} Hz")
 lines+=['These timings include context construction, physical encoding, critic scoring and argmax once candidates are available; exclude unchanged generator proposal generation.100 end-to-end repetitions,1000 encoding/scoring repetitions after warm-up; first frozen state/scenario. Not a cohort-wide latency bound.',
 'One-shot eta selection does not set the low-level control frequency. Maximum probe horizon=.15 physical seconds, not a full rollout. No controller-conditioned model is deployed because LOSO did not pass.',
 '', 'Integrity / cost:',
 '47,486 original canonical training/validation pairs;305 states; observed partial-count likelihood retained, scene-balanced training, source-only normalization and selection,3 seeds,4 folds. No generator changes. No new full continuation/success labels.',
 'Initial global JAX x64 setting was caught by exact Ring input replay: it changes default Flow random-normal samples. Superseded features/models are quarantined in superseded_precision_audit. Corrected native sampler precision is Toy64, DB/Four/Ring32; safety stays numpy64. Ring input replay max error2.38e-7. All formal results use repaired contexts.',
 'Three source response projections failed and two source episodes terminated before the third probe step. All five labels were retained with context-unavailable mask; no fake failure/success labels. All target response contexts valid.',
 'Cache preflights reuse39,745 compatible target seed records, with191 pre-existing missing/numerical slots preserved as intervals; no certification jobs. Derived response_cache is not added as Q labels. Actual microprobe and benchmark cost is logged separately.',
 'Historical target knowledge motivated the task/schema; source-only retraining is clean but is not target-naive representation discovery. Historical generators supplying fixed candidates saw target-scene labels; no end-to-end generator zero-shot claim.',
 '', 'Reproduce: context.py build -> train.py materialize/train/freeze -> evaluate.py target_context/source_audit/predict/evaluate -> support_audit.py -> benchmark.py -> finalize.py. Slurm launch scripts and checkpoint hashes retained. Target outcome access is gated by all-model freeze.']
 (OUT/'final_report.txt').write_text('\n'.join(lines)+'\n')
 write('experiment_ledger.csv',[{'stage':s,'status':'complete','new_full_continuations':0,'artifact':a} for s,a in [
  ('counterexample','swap_audit.json'),('precision_repair','precision_repair_audit.json'),('source_features','feature_audit.json'),
  ('36_source_models','models_frozen.json'),('four_fold_LOSO','loso_results.csv'),('shuffle','context_shuffle_effect.csv'),('latency','deployment_latency.csv'),('handoff','final_decision.json')]])
 dump(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.csv','.npz','.msgpack','.txt'] and 'response_cache' not in p.parts and 'superseded_precision_audit' not in p.parts and p.name not in ['artifact_hashes.json','working_state.json']})
 dump(OUT/'working_state.json',{'stage':'complete','classification':decision['classification'],'generator_unchanged':True,'new_full_continuations':0,'jobs_remaining':0})
 print(json.dumps(decision,indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).parent
def read(p):return list(csv.DictReader(open(p)))
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cfg=json.load(open(H/'sparse_codebook.json'));ts=json.load(open(H/'test_summary.json'));fresh=json.load(open(H/'fresh_summary.json')) if (H/'fresh_summary.json').exists() else None;ctrl={r['model']:r for r in ts['controllers']};oracle=ctrl['SPARSE_ORACLE'];fixed=ctrl['BEST_FIXED_SPARSE_MODE'];selector=ctrl['SPARSE_SELECTOR'];multi=len(ts['selection_counts'])>=2 and ts['normalized_entropy']>=.20;coverage=oracle['B63_states']/oracle['states'];learn=ts['selector_mode_B63_on_coverable'];adv=(selector['B63_states']>=fixed['B63_states']+3 or selector['mean_Q64']>=fixed['mean_Q64']+.05);fresh_rep=(fresh is not None and fresh['selector_B63']>=fresh['fixed_B63']+3 and fresh['selector_mean_Q64']>=fresh['fixed_mean_Q64']+.03)
 if coverage<.60:classification='SPARSE_CODEBOOK_COVERAGE_TOO_LOW'
 elif not multi:classification='COMMON_MODE_COLLAPSE_REMAINS'
 elif learn<.70:classification='STATE_TO_MODE_LEARNING_WEAK'
 elif adv and (fresh is None or fresh_rep):classification='SPARSE_MODE_RELEARNING_SUPPORTED'
 elif selector['mean_Q64']<.75 or (fresh is not None and fresh['selector_mean_Q64']<fresh['fixed_mean_Q64']-.03):classification='DOMINANT_MODE_WAS_CRITICAL'
 else:classification='STATE_TO_MODE_LEARNING_WEAK'
 difficulty=read(H/'sparse_codebook_difficulty.csv');testm=[r for r in difficulty if r['split']=='test'];models=read(H/'test_model_metrics.csv');deploy=read(H/'test_model_deployment.csv');sel=json.load(open(H/'selected_model.json'));cal=json.load(open(H/'calibration.json'));thr=json.load(open(H/'selected_threshold.json'))
 decision={'classification':classification,'stress_design_limitation':'Every original mode exceeded the requested 35% TRAIN prevalence ceiling; the selected Pareto pair therefore has 47.7% and 59.4% TRAIN prevalence rather than 10-30%.','removed_original_mode_ids':cfg['removed_original_mode_ids'],'retained_original_mode_ids':cfg['selected_original_mode_ids'],'dominant_original_mode_0_removed':0 in cfg['removed_original_mode_ids'],'original_test':ts,'fresh_replication':fresh,'answers':{'best_fixed_test_B63':fixed['B63_states'],'sparse_oracle_test_B63':oracle['B63_states'],'selector_coverable_correct_fraction':learn,'selector_uses_multiple_modes':multi,'fresh_replicated_advantage':fresh_rep,'original_selector_depended_on_dominant_mode':False if classification=='SPARSE_MODE_RELEARNING_SUPPORTED' else None},'selected_model':sel,'calibration':cal,'threshold':thr,'next_step':'Do not add eta or train another model automatically. To test the intended 10-30% regime, construct a new TRAIN-only dictionary from genuinely narrow modes or move to a scenario whose feasible modes are naturally sparse.'};dump('final_decision.json',decision)
 lines=['# Sparse mode relearning stress test','',f'Classification: **{classification}**','',f"Frozen sparse codebook: original modes {cfg['selected_original_mode_ids']}; selected from TRAIN only. Removed modes: {cfg['removed_original_mode_ids']}.",'',"Important limitation: all 12 original modes exceeded 35% TRAIN strong-feasible prevalence. No subset could meet the requested 10--30% per-mode range. The Pareto pair is therefore a weaker sparsity stress test.",'','## Difficulty','','|Split|Mode 3 prevalence|Mode 9 prevalence|Union/oracle coverage|Oracle mean Q|','|---|---:|---:|---:|---:|']
 for sp in ('train','val','test'):
  q=[r for r in difficulty if r['split']==sp];m3=next(r for r in q if r['metric']=='mode' and r['original_mode_id']=='3');m9=next(r for r in q if r['metric']=='mode' and r['original_mode_id']=='9');u=next(r for r in q if r['metric']=='union_oracle');lines.append(f"|{sp}|{float(m3['prevalence']):.1%}|{float(m9['prevalence']):.1%}|{float(u['prevalence']):.1%}|{float(u['mean_Q']):.3f}|")
 lines += ['','## TEST controller result','','|Controller|B63|Mean Q64|Success|Deadlock|Timeout|Collision|','|---|---:|---:|---:|---:|---:|---:|']
 for k in ('BEST_FIXED_SPARSE_MODE','SPARSE_SELECTOR','SPARSE_ORACLE','SAFETY'):
  r=ctrl[k];lines.append(f"|{k}|{r['B63_states']}/{r['states']}|{r['mean_Q64']:.3f}|{r['successes']}/{r['trials']}|{r['deadlock']}|{r['timeout']}|{r['collision']}|")
 lines += ['',f"On {ts['coverable_states']} oracle-coverable TEST states, the selector chose a B63 sparse mode on {ts['selector_mode_B63_on_coverable']:.1%}; fixed mode success was {ts['fixed_B63_on_coverable']:.1%}.",f"Mode choices: {ts['selection_counts']}, normalized entropy {ts['normalized_entropy']:.3f}; mean mode regret {ts['selector_mean_mode_regret']:.3f}.",'','## Predictor comparison','','|Predictor|TEST NLL|Brier|Top-1 empirical Q|Rank correlation|','|---|---:|---:|---:|---:|']
 for r in models:
  if r['model'].endswith('_calibrated') or r['model'] in ('global_prior','linear','nearest_neighbor',sel['model']):lines.append(f"|{r['model']}|{float(r['nll']):.3f}|{float(r['brier']):.3f}|{float(r['top1_empirical_Q']):.3f}|{float(r['within_state_rank_correlation']):.3f}|")
 if fresh:
  lines += ['','## Fresh replication','',f"Fresh 64 states: fixed {fresh['fixed_B63']}/64 B63, mean Q64 {fresh['fixed_mean_Q64']:.3f}; selector {fresh['selector_B63']}/64 B63, mean Q64 {fresh['selector_mean_Q64']:.3f}.",f"Selector vs fixed rescue/break: {fresh['selector_rescue_vs_fixed']}/{fresh['selector_break_vs_fixed']}. Failure diagnosis: {fresh['failure_diagnosis']}.",'',"|Frozen action|States|Selector B63|Selector mean Q64|Fixed B63 on same states|",'|---|---:|---:|---:|---:|']
  lines += [f"|{r['action']}|{r['states']}|{r['selector_B63']}/{r['states']}|{r['selector_mean_Q64']:.3f}|{r['fixed_B63']}/{r['states']}|" for r in fresh['action_breakdown']]
 lines += ['','## Answers','',f"The dominant original mode 0 was removed: **YES**.",f"The relearned selector genuinely used multiple sparse modes: **{'YES' if multi else 'NO'}**.",f"Did the original selector depend critically on the dominant mode: **{'NO' if classification=='SPARSE_MODE_RELEARNING_SUPPORTED' else 'NOT RESOLVED'}**.","This result does not establish the requested 10--30% mode-prevalence regime because the frozen 12-mode dictionary contains no such modes."]
 (H/'final_report.md').write_text('\n'.join(lines)+'\n')
 runt=[]
 for d in ('fresh_raw','audit_raw'):
  for p in (H/d).glob('*runtime.json'):runt.append(json.load(open(p)))
 dump('runtime_statistics.json',{'training_new_rollouts':0,'fresh_new_continuations':sum(r['new_continuations'] for r in runt),'physical_steps':sum(r['physical_steps'] for r in runt),'worker_wall_seconds':sum(r['wall_seconds'] for r in runt),'max_gpu_shards':6})
 w=json.load(open(H/'working_state.json'));w.update(status='COMPLETE',classification=classification,completed=list(dict.fromkeys(w['completed']+['model_training','val_freeze','test_evaluation','fresh_replication','finalization'])),next_action='none');dump('working_state.json',w)
 artifacts={str(p.relative_to(H)):sha(p) for p in H.rglob('*') if p.is_file() and p.name!='manifest.json' and not any(x in str(p) for x in ('/fresh_raw/','/audit_raw/','/fresh_runs/','/audit_runs/','/logs/','__pycache__','/fresh_plans/','/audit_plans/'))};dump('manifest.json',{'task':'ORTHOFLOW3_SPARSE_MODE_RELEARNING_V1','status':'COMPLETE','artifacts':artifacts});print(json.dumps(decision,indent=2))
if __name__=='__main__':main()

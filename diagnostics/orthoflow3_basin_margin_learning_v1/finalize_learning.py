#!/usr/bin/env python3
"""Freeze decisions, runtime, quantitative report, and final manifest."""
from __future__ import annotations
import csv,glob,hashlib,json,statistics
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_basin_margin_learning_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.load(open(p))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def read_csv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def read_jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 integ=load(HERE/'frozen_dataset_integrity.json');assert integ['unchanged'] and not integ['source_family_leakage'] and integ['new_ball_labels_generated']==0
 test=load(HERE/'intermediate_test_summary.json');fresh=load(HERE/'fresh_wide_summary.json');overlap=load(HERE/'train_ball_overlap_analysis.json');geo=read_csv(HERE/'test_geometric_predictions.csv');val=load(HERE/'val_selection.json')['selected']
 # Decisions are intentionally driven by the frozen metrics, not by point error.
 point={'classification':'LOWJ_POINT_TARGET_REMAINS_COMPETITIVE','basis':'On TEST all three controllers were 9/9 B63, while LOWJ had far lower J_def. On fresh-WIDE LOWJ achieved 220/300 versus CENTER 175/300, with more rescues (74 vs 36), fewer breaks (65 vs 72), and lower successful J_def (0.273 vs 0.691).','g_lowj_compatible':True,'g_lowj_retrained':False}
 basin={'classification':'BASIN_MARGIN_SUPERVISION_HARMS','basis':'MARGIN did not improve intermediate TEST B63 or valid perturbation retention, had worse TEST rho/geometry, and on fresh-WIDE reduced success from 175 to 166 while increasing Safety-success breaks from 72 to 106. The paired break-rate increase was 0.1133 with bootstrap 95% CI [0.0633,0.1633].','set_supervision_adds_value_beyond_center':False}
 dump(HERE/'point_target_decision.json',point);dump(HERE/'basin_supervision_decision.json',basin)
 # Exact rollout accounting from frozen raw records.
 counts={};steps={};walls={}
 for stage in ('val16','test64','perturb','fresh_wide'):
  rr=[]
  for p in sorted((HERE/'raw'/stage).glob('shard*.jsonl')):rr+=read_jsonl(p)
  counts[stage]=len(rr);key='episode_steps' if stage=='fresh_wide' else 'continuation_steps';steps[stage]=sum(int(x[key]) for x in rr)
  rt=[load(p) for p in sorted((HERE/'raw'/stage).glob('*runtime.json'))];walls[stage]=max(float(x.get('wall_time_seconds',x.get('wall_seconds',0))) for x in rt)
 assert counts=={'val16':960,'test64':1728,'perturb':688,'fresh_wide':1200}
 train_s=[load(p) for p in glob.glob(str(HERE/'g_*'/'runs'/'*_summary.json'))]
 runtime={'training_runs':6,'training_critical_path_seconds':max(x['training_seconds'] for x in train_s),'training_sum_worker_seconds':sum(x['training_seconds'] for x in train_s),'new_continuations':sum(counts.values()),'new_continuations_by_stage':counts,'physical_steps':sum(steps.values()),'physical_steps_by_stage':steps,'rollout_critical_path_wall_seconds':sum(walls.values())+19.0,'rollout_stage_max_wall_seconds':{**walls,'perturb_interrupted_first_attempt_seconds':19.0},'max_gpu_shards':6,'gpu_memory_peak_mib':'not captured','cpu_threads_per_shard':2,'max_concurrent_cpu_threads':12,'ram_allocation_peak_gib':48,'new_ball_labels':0,'new_anchor_balls':0,'budget_continuations_cap':12000,'budget_steps_cap':5000000,'budget_compliant':sum(counts.values())<=12000 and sum(steps.values())<=5000000,'canceled_perturb_job_note':'one mistakenly overlapping perturb submission was canceled immediately; 32 completed records were retained exactly and the remaining plan was resumed without duplication'}
 dump(HERE/'runtime_statistics_training_eval.json',runtime)
 cgeo=[x for x in geo if x['model']=='g_center'];mgeo=[x for x in geo if x['model']=='g_margin']
 seeds={a:[load(HERE/f'g_{a}/runs/{a}_seed{s}_summary.json') for s in (17,23,41)] for a in ('center','margin')}
 ratios=read_csv(HERE/'robustness_vs_deformation.csv')
 ratio_summary={k:{'mean':statistics.fmean(float(x[k]) for x in ratios),'median':statistics.median(float(x[k]) for x in ratios)} for k in ('J_center_over_lowj','J_margin_over_lowj','J_margin_over_center')}
 report=f'''# OrthoFlow3 CENTER vs MARGIN learning v1

## Decisions

- Point target: **LOWJ_POINT_TARGET_REMAINS_COMPETITIVE**.
- Basin supervision: **BASIN_MARGIN_SUPERVISION_HARMS**.

Verified set-valued supervision did **not** add value beyond robust-center regression. On the held-out intermediate states both were robust despite failing to reproduce the verified-ball geometry; on the mandatory fresh t=0 extrapolation, MARGIN caused materially more Safety-success breaks than CENTER.

## Frozen dataset and geometry

- Dataset unchanged: TRAIN/VAL/TEST = 25/10/9; independent source families = 5/2/2; leakage = 0.
- New labels/anchors or center/radius modifications: 0/0/none.
- True t=0 labels: 0/44. All supervised examples are intermediate states.
- All 25 TRAIN retained 0.8-radius balls share the exact physical eta `(0.625, 0, 0.375)`; common-intersection fraction = 25/25 and minimum retained radius = {overlap['minimum_retained_radius']:.5f}. A state-independent zero-loss MARGIN solution therefore exists.

## Training and VAL selection

Both arms used the same 214→128→128→3 SiLU model (44,419 parameters), unconstrained normalized-eta output, AdamW 1e-3/1e-5, and seeds 17/23/41.

| arm | seed | best epoch | VAL loss | TRAIN rho | TRAIN retained | VAL rho | VAL retained | VAL true success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CENTER | 17 | {seeds['center'][0]['best_epoch']} | {seeds['center'][0]['best_val_loss']:.6f} | {seeds['center'][0]['train_metrics']['mean_rho']:.3f} | {seeds['center'][0]['train_metrics']['retained_fraction']:.2f} | {seeds['center'][0]['val_metrics']['mean_rho']:.3f} | {seeds['center'][0]['val_metrics']['retained_fraction']:.2f} | 160/160 |
| CENTER | 23 | {seeds['center'][1]['best_epoch']} | {seeds['center'][1]['best_val_loss']:.6f} | {seeds['center'][1]['train_metrics']['mean_rho']:.3f} | {seeds['center'][1]['train_metrics']['retained_fraction']:.2f} | {seeds['center'][1]['val_metrics']['mean_rho']:.3f} | {seeds['center'][1]['val_metrics']['retained_fraction']:.2f} | 80/160 |
| CENTER | 41 | {seeds['center'][2]['best_epoch']} | {seeds['center'][2]['best_val_loss']:.6f} | {seeds['center'][2]['train_metrics']['mean_rho']:.3f} | {seeds['center'][2]['train_metrics']['retained_fraction']:.2f} | {seeds['center'][2]['val_metrics']['mean_rho']:.3f} | {seeds['center'][2]['val_metrics']['retained_fraction']:.2f} | 80/160 |
| MARGIN | 17 | {seeds['margin'][0]['best_epoch']} | {seeds['margin'][0]['best_val_loss']:.6f} | {seeds['margin'][0]['train_metrics']['mean_rho']:.3f} | {seeds['margin'][0]['train_metrics']['retained_fraction']:.2f} | {seeds['margin'][0]['val_metrics']['mean_rho']:.3f} | {seeds['margin'][0]['val_metrics']['retained_fraction']:.2f} | 160/160 |
| MARGIN | 23 | {seeds['margin'][1]['best_epoch']} | {seeds['margin'][1]['best_val_loss']:.6f} | {seeds['margin'][1]['train_metrics']['mean_rho']:.3f} | {seeds['margin'][1]['train_metrics']['retained_fraction']:.2f} | {seeds['margin'][1]['val_metrics']['mean_rho']:.3f} | {seeds['margin'][1]['val_metrics']['retained_fraction']:.2f} | 80/160 |
| MARGIN | 41 | {seeds['margin'][2]['best_epoch']} | {seeds['margin'][2]['best_val_loss']:.6f} | {seeds['margin'][2]['train_metrics']['mean_rho']:.3f} | {seeds['margin'][2]['train_metrics']['retained_fraction']:.2f} | {seeds['margin'][2]['val_metrics']['mean_rho']:.3f} | {seeds['margin'][2]['val_metrics']['retained_fraction']:.2f} | 80/160 |

Selected CENTER: seed 17, SHA256 `{val['center']['checkpoint_sha256']}`. Selected MARGIN: seed 17, SHA256 `{val['margin']['checkpoint_sha256']}`. Both achieved VAL 160/160 and 10/10 states at 16/16. MARGIN's selected checkpoint had TRAIN zero-loss fraction {seeds['margin'][0]['train_metrics']['zero_loss_fraction']:.2f}, output mean pairwise distance {seeds['margin'][0]['train_metrics']['mean_pairwise_eta_distance']:.3f}; CENTER's was {seeds['center'][0]['train_metrics']['mean_pairwise_eta_distance']:.3f}. Thus MARGIN did not fully collapse numerically, despite an exact universal solution existing.

## Held-out intermediate TEST

| model | inside ball | retained | mean/median rho | B63 states | mean Q64 | successes | deadlock/timeout/collision | mean successful J_def |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CENTER | {test['controllers']['g_center']['inside_ball']}/9 | {test['controllers']['g_center']['inside_retained']}/9 | {test['controllers']['g_center']['rho_mean']:.3f}/{test['controllers']['g_center']['rho_median']:.3f} | 9/9 | 1.000 | 576/576 | 0/0/0 | {test['controllers']['g_center']['successful_J_def_mean']:.6f} |
| MARGIN | {test['controllers']['g_margin']['inside_ball']}/9 | {test['controllers']['g_margin']['inside_retained']}/9 | {test['controllers']['g_margin']['rho_mean']:.3f}/{test['controllers']['g_margin']['rho_median']:.3f} | 9/9 | 1.000 | 576/576 | 0/0/0 | {test['controllers']['g_margin']['successful_J_def_mean']:.6f} |
| LOWJ | n/a | n/a | n/a | 9/9 | 1.000 | 576/576 | 0/0/0 | {test['controllers']['g_lowj']['successful_J_def_mean']:.6f} |

Paired CENTER→MARGIN transitions: BOTH_B63 9; CENTER_FAIL→MARGIN_B63 0; CENTER_B63→MARGIN_FAIL 0; BOTH_FAIL 0. All 18 CENTER/MARGIN predictions were in the `rho>1` bin yet all were Q64=1, so neither rho nor raw center error is predictive within this nondiscriminative TEST cohort. The conservative balls are valid inner sets, not complete success basins.

Perturbations that remained in E_bridge were universally successful: CENTER 0.25r 232/232 and 0.50r 168/168; MARGIN 0.25r 152/152 and 0.50r 136/136. MARGIN shows no measurable tolerance gain. Fewer MARGIN perturbations were eligible because its nominal outputs were less often in-domain.

LOWJ is feature-compatible and was evaluated read-only. Among states where all policies were B63, mean statewise J ratios were CENTER/LOWJ {ratio_summary['J_center_over_lowj']['mean']:.2f}, MARGIN/LOWJ {ratio_summary['J_margin_over_lowj']['mean']:.2f}, and MARGIN/CENTER {ratio_summary['J_margin_over_center']['mean']:.3f}.

## Fresh-WIDE t=0 extrapolation (300 matched episodes)

| controller | success | deadlock | timeout | collision | rescue | break | net vs Safety | mean successful J_def |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 211 | 6 | 83 | 0 | – | – | – | 0 |
| LOWJ | 220 | 46 | 34 | 0 | 74 | 65 | +9 | {fresh['controllers']['g_lowj']['successful_J_def_mean']:.3f} |
| CENTER | 175 | 115 | 10 | 0 | 36 | 72 | -36 | {fresh['controllers']['g_center']['successful_J_def_mean']:.3f} |
| MARGIN | 166 | 123 | 11 | 0 | 61 | 106 | -45 | {fresh['controllers']['g_margin']['successful_J_def_mean']:.3f} |

MARGIN recovered 14 CENTER breaks and created 48 new breaks; it preserved 33 CENTER rescues and added 28 new rescues. Its success-rate difference from CENTER was -3.0 percentage points (95% bootstrap CI -9.33 to +3.33), while its break-rate difference was +11.33 points (95% CI +6.33 to +16.33).

Fresh h0 is far outside labeled support: nearest TRAIN distance mean/median {fresh['support_distance']['all']['mean']:.3f}/{fresh['support_distance']['all']['median']:.3f}; every nearest neighbor was a transferred intermediate state. Failures were not associated with larger distance within this uniformly far-OOD cohort (CENTER failure mean {fresh['support_distance']['center_failure']['mean']:.3f} vs success {fresh['support_distance']['center_success']['mean']:.3f}; MARGIN failure {fresh['support_distance']['margin_failure']['mean']:.3f} vs success {fresh['support_distance']['margin_success']['mean']:.3f}). The t0 distribution mismatch remains a major limitation, but does not explain away MARGIN's paired increase in nominal breaks.

## Direct answers

1. Robust-center regression does **not** improve on LOWJ regression: TEST robustness tied, while LOWJ strongly outperformed on fresh-WIDE and had much lower J_def.
2. Basin-margin supervision does **not** improve on center MSE; it materially increases fresh nominal breakage.
3. Normalized error relative to verified radius is **not shown to be more predictive** than point error here: every TEST prediction was outside the balls and still Q64=1.
4. Margin supervision provides **no measurable perturbation-tolerance gain**: both arms retained 100% success for every in-domain perturbation tested.

## Runtime and next step

Six training runs took {runtime['training_critical_path_seconds']:.1f}s critical-path. Evaluation used {runtime['new_continuations']:,} new continuations and {runtime['physical_steps']:,} physical steps; rollout critical path was {runtime['rollout_critical_path_wall_seconds']/60:.2f} minutes. Maximum allocation was 6 GPU shards, 12 CPU threads, and 48 GiB RAM; GPU peak memory was not captured. No new ball label was generated.

The smallest justified next experiment is **not** another margin loss. First add a small, source-diverse set of verified true-t0 ball labels (with complete source-family separation), then rerun the same CENTER-vs-MARGIN comparison. This directly tests whether the current failure is caused by the documented t0 support mismatch before considering richer losses, Q, J, or selectors.
'''
 (HERE/'basin_margin_learning_final_report.md').write_text(report)
 files=[]
 for p in sorted(HERE.rglob('*')):
  if not p.is_file() or p.name=='final_learning_manifest.json' or '__pycache__' in p.parts:continue
  files.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)})
 dump(HERE/'final_learning_manifest.json',{'schema':'orthoflow3_center_vs_margin_learning_v1','decision_A':point['classification'],'decision_B':basin['classification'],'dataset_sha256':sha(HERE/'verified_ball_dataset.csv'),'dataset_unchanged':True,'new_ball_labels':0,'orthoflow3_sha256':sha(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'),'g_center_sha256':val['center']['checkpoint_sha256'],'g_margin_sha256':val['margin']['checkpoint_sha256'],'g_lowj_sha256':load(HERE/'frozen_lowj_reference.json')['checkpoint']['checkpoint_sha256'],'controller_modified':False,'safety_projection_modified':False,'files':files})
 print(json.dumps({'decision_A':point['classification'],'decision_B':basin['classification'],'runtime':runtime,'report':str(HERE/'basin_margin_learning_final_report.md')},indent=2))
if __name__=='__main__':main()

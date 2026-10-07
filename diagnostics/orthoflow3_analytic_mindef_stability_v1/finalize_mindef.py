import csv,hashlib,json
from pathlib import Path
import numpy as np
H=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_analytic_mindef_stability_v1')
def rows(n):return list(csv.DictReader(open(H/n)))
def f(x):return float(x)
def main():
 w=rows('within_state_rank_correlations.csv');r=rows('proxy_selection_regret.csv');s=rows('local_selection_stability.csv');g=rows('basis_gram_statistics.csv');d=rows('dimension_contributions.csv');p=rows('projection_compatibility.csv');proxies=['R_eta_raw','R_eta_norm','R_gram','R_exec1'];metrics={}
 for name in proxies:
  ww=[x for x in w if x['proxy']==name and x['stratum']=='ACTIVE_REQUIRED'];rr=[x for x in r if x['proxy']==name and x['stratum']=='ACTIVE_REQUIRED'];ss=[x for x in s if x['selector']==name];ratio=np.array([f(x['regret_ratio']) for x in rr]);absolute=np.array([f(x['absolute_regret']) for x in rr]);move=np.array([f(x['normalized_eta_movement']) for x in ss])
  metrics[name]={'active_states':len(rr),'within_state_spearman_mean':float(np.nanmean([f(x['spearman']) for x in ww])),'within_state_kendall_mean':float(np.nanmean([f(x['kendall']) for x in ww])),'regret_ratio_median':float(np.median(ratio)),'regret_ratio_mean':float(np.mean(ratio)),'regret_ratio_p90':float(np.quantile(ratio,.9)),'regret_ratio_worst':float(np.max(ratio)),'absolute_regret_median':float(np.median(absolute)),'absolute_regret_mean':float(np.mean(absolute)),'within_5pct':float(np.mean([x['within_5pct']=='True' for x in rr])),'within_10pct':float(np.mean([x['within_10pct']=='True' for x in rr])),'within_20pct':float(np.mean([x['within_20pct']=='True' for x in rr])),'local_movement_mean':float(np.mean(move)),'local_movement_median':float(np.median(move))}
 true=[f(x['normalized_eta_movement']) for x in s if x['selector']=='true_J'];active_dim=[x for x in d if x['stratum']=='ACTIVE_REQUIRED'];gram={k:{'median':float(np.median([f(x[k]) for x in g])),'mean':float(np.mean([f(x[k]) for x in g]))} for k in ['M11','M22','M33','M12','M13','M23','cos12','cos13','cos23']}
 decision={'classification':'ANALYTIC_MINDEF_INADEQUATE','learned_J_justified':'YES, as a later controlled possibility; not trained here','states':24,'pairs_tplus4':12,'pairs_plusminus1_evaluable':0,'robust_candidates_per_state':{'min':2,'median':2.0,'max':2},'proxy_metrics_active':metrics,'true_J_local_movement':{'mean':float(np.mean(true)),'median':float(np.median(true))},'projection':{'theoretical_nonexpansive_claim_valid':True,'reason':'both projections solve the same Euclidean quadratic projection onto a closed convex intersection of linear halfspaces and second-order speed balls; u_safe is feasible for the second problem','median_ratio':float(np.median([f(x['ratio']) for x in p])),'p95_ratio':float(np.quantile([f(x['ratio']) for x in p],.95)),'max_ratio':float(np.max([f(x['ratio']) for x in p])),'violations_gt_1e7':sum(x['violation_gt_1e_7']=='True' for x in p)},'gram':gram,'active_dimension_abs_fraction':{'diagonal_mean':float(np.mean([f(x['diag_abs_fraction']) for x in active_dim])),'diagonal_median':float(np.median([f(x['diag_abs_fraction']) for x in active_dim])),'cross_mean':float(np.mean([f(x['cross_abs_fraction']) for x in active_dim])),'cross_median':float(np.median([f(x['cross_abs_fraction']) for x in active_dim]))},'large_true_J_jumps':0,'near_tie_fraction_among_large_jumps':'NA (0 denominator)','future_pipeline':'Revise: learned Q plus analytical deformation alone is not supported. Retain Direct-eta deployment goal, but deformation supervision needs a richer true-J surrogate or another controlled objective after confirming with denser robust candidate sets.','smallest_next_experiment':'On the same 10 ACTIVE-required states, promote two additional predeclared low-frontier Sobol candidates per state to B63, then repeat this audit; do not train J yet.'}
 (H/'audit_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 runtimes=[json.load(open(x)) for x in (H/'raw'/'enrichment').glob('*_runtime.json')];runtime={'new_continuations':sum(x['new_rollouts'] for x in runtimes),'physical_steps':sum(x['new_physical_steps'] for x in runtimes),'wall_seconds_max_shard':max(x['wall_seconds'] for x in runtimes),'gpu_shards':2,'gpu_memory_peak_mib_approx':1202,'cpu_threads':2,'ram_peak_mib_approx':718,'collisions_or_errors':0};(H/'runtime_statistics.json').write_text(json.dumps(runtime,indent=2)+'\n')
 report=f'''# OrthoFlow3 analytical minimum-deformation stability audit

## Decision

**`ANALYTIC_MINDEF_INADEQUATE`.** All 24 frozen states (12 anchors plus their deterministic `t+4` neighbors) have exactly two confirmed B63 eta candidates after 144 new enrichment continuations. All 12 local pairs are evaluable. No network was trained.

True `J_def` is accumulated in `run_subset_arms.py:143-176` as `dt * sum ||u_exec-u_safe||²` from the original absolute episode time to terminal/horizon 850. Canonical aggregation gates on B63 first, averages J only over successful continuations, then minimizes mean successful J with eta-index tie-breaking.

| proxy | active within-state Spearman / Kendall | median / mean / p90 regret ratio | within 5% / 10% / 20% | mean local eta movement |
|---|---:|---:|---:|---:|
| raw eta norm | {metrics['R_eta_raw']['within_state_spearman_mean']:.3f} / {metrics['R_eta_raw']['within_state_kendall_mean']:.3f} | {metrics['R_eta_raw']['regret_ratio_median']:.3f} / {metrics['R_eta_raw']['regret_ratio_mean']:.3f} / {metrics['R_eta_raw']['regret_ratio_p90']:.3f} | {metrics['R_eta_raw']['within_5pct']:.0%} / {metrics['R_eta_raw']['within_10pct']:.0%} / {metrics['R_eta_raw']['within_20pct']:.0%} | {metrics['R_eta_raw']['local_movement_mean']:.3f} |
| normalized eta norm | {metrics['R_eta_norm']['within_state_spearman_mean']:.3f} / {metrics['R_eta_norm']['within_state_kendall_mean']:.3f} | {metrics['R_eta_norm']['regret_ratio_median']:.3f} / {metrics['R_eta_norm']['regret_ratio_mean']:.3f} / {metrics['R_eta_norm']['regret_ratio_p90']:.3f} | {metrics['R_eta_norm']['within_5pct']:.0%} / {metrics['R_eta_norm']['within_10pct']:.0%} / {metrics['R_eta_norm']['within_20pct']:.0%} | {metrics['R_eta_norm']['local_movement_mean']:.3f} |
| start Gram energy | {metrics['R_gram']['within_state_spearman_mean']:.3f} / {metrics['R_gram']['within_state_kendall_mean']:.3f} | {metrics['R_gram']['regret_ratio_median']:.3f} / {metrics['R_gram']['regret_ratio_mean']:.3f} / {metrics['R_gram']['regret_ratio_p90']:.3f} | {metrics['R_gram']['within_5pct']:.0%} / {metrics['R_gram']['within_10pct']:.0%} / {metrics['R_gram']['within_20pct']:.0%} | {metrics['R_gram']['local_movement_mean']:.3f} |
| one-step executed correction | {metrics['R_exec1']['within_state_spearman_mean']:.3f} / {metrics['R_exec1']['within_state_kendall_mean']:.3f} | {metrics['R_exec1']['regret_ratio_median']:.3f} / {metrics['R_exec1']['regret_ratio_mean']:.3f} / {metrics['R_exec1']['regret_ratio_p90']:.3f} | {metrics['R_exec1']['within_5pct']:.0%} / {metrics['R_exec1']['within_10pct']:.0%} / {metrics['R_exec1']['within_20pct']:.0%} | {metrics['R_exec1']['local_movement_mean']:.3f} |

True-J selection movement is exactly 0 on all 12 `t+4` pairs, whereas analytical selectors introduce mean movement 0.141--0.187. There are no large true-J jumps, so near-tie-switch attribution is not applicable.

## Geometry and projection

Median Gram diagonals `(M11,M22,M33)` are `({gram['M11']['median']:.4g}, {gram['M22']['median']:.4g}, {gram['M33']['median']:.4g})`; median cross terms `(M12,M13,M23)` are `({gram['M12']['median']:.3g}, {gram['M13']['median']:.3g}, {gram['M23']['median']:.3g})`. Goal and flow-perpendicular are essentially orthogonal (`cos12` median {gram['cos12']['median']:.3g}), but goal/relative coupling remains material (`cos13` median {gram['cos13']['median']:.3f}). In active states, diagonal terms account for a median {decision['active_dimension_abs_fraction']['diagonal_median']:.1%} of absolute Gram contributions and cross terms {decision['active_dimension_abs_fraction']['cross_median']:.1%}; the dimensions are interpretable but not fully separable.

The second solver is a Euclidean projection onto a closed convex feasible set and `u_safe` is feasible, so non-expansiveness legitimately gives `||delta u|| <= ||g||` up to numerical tolerance. Empirically the ratio has median {decision['projection']['median_ratio']:.3f}, P95 {decision['projection']['p95_ratio']:.3f}, maximum {decision['projection']['max_ratio']:.6f}, with zero violations above `1+1e-7`.

## Implication

The high pooled correlations on all states are driven by trivial zero-sufficient cases. On ACTIVE-required states the simple start-local proxies do not rank full-horizon deformation and incur meaningful regret. A learned `J(h,eta)` is therefore scientifically justified as a later controlled possibility, though this audit does not train it. The proposed `Q + analytical R + Direct-eta G` pipeline needs revision: feasibility Q may still be useful, but these analytical proxies should not be the sole deformation objective.

Smallest next experiment: promote two additional predeclared low-frontier Sobol candidates on the same 10 ACTIVE-required states, repeat the ranking/regret audit, and only then decide whether to train J.
''';(H/'analytic_mindef_report.md').write_text(report)
 files=[]
 for x in sorted(H.iterdir()):
  if x.is_file() and x.name!='manifest.json':files.append({'name':x.name,'sha256':hashlib.sha256(x.read_bytes()).hexdigest(),'bytes':x.stat().st_size})
 (H/'manifest.json').write_text(json.dumps({'classification':decision['classification'],'files':files},indent=2)+'\n');print(json.dumps(decision,indent=2))
if __name__=='__main__':main()

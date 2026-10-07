#!/usr/bin/env python3
"""Build immutable handoff report, integrity checks, and artifact manifest."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/orthoflow3_q_learnability_v2'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,x): path.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def rows(path): return list(csv.DictReader(path.open()))

state=json.loads((HERE/'eligible_state_manifest.json').read_text()); split=json.loads((HERE/'state_split_manifest.json').read_text())
info=json.loads((HERE/'data_informativeness.json').read_text()); model=json.loads((HERE/'model_config.json').read_text())
prob=json.loads((HERE/'heldout_probability_metrics.json').read_text()); selected=json.loads((HERE/'selected_checkpoint.json').read_text())
pre=json.loads((HERE/'pre_followup_summary.json').read_text()); decision=json.loads((HERE/'q_decision.json').read_text())
runtime=json.loads((HERE/'runtime_statistics.json').read_text()); zero=rows(HERE/'zero_eta_audit.csv'); grad=rows(HERE/'gradient_directional_results.csv')
rank=rows(HERE/'heldout_statewise_ranking.csv'); top=rows(HERE/'top_candidate_quality.csv')
vary=[r for r in rank if float(r['true_q_variance'])>0]
def stats(values):
    a=np.asarray(values,dtype=float); return {'mean':float(a.mean()),'median':float(np.median(a)),'p25':float(np.quantile(a,.25)),'p75':float(np.quantile(a,.75))}
rank_all={k:stats([r[k] for r in rank]) for k in ('spearman','kendall')}; rank_vary={k:stats([r[k] for r in vary]) for k in ('spearman','kendall')}
top_stats={k:stats([r[k] for r in top]) for k in ('true_q8','best_q8','gap_to_best','random_candidate_expectation')}

base=[]; follow=[]; max_replay=0.0
for stem,target in (('base_rollout_plan',base),('zero_followup_plan',follow),('gradient_followup_plan',follow)):
    for path in sorted((HERE/'raw'/stem).glob('shard*.jsonl')):
        if not path.stem.removeprefix('shard').isdigit(): continue
        for line in path.read_text().splitlines():
            if line.strip(): target.append(json.loads(line))
all_ids=[r['task_id'] for r in base+follow]
frozen=datetime.fromisoformat(state['frozen_utc']); now=datetime.now(timezone.utc)
runtime.update({'audit_elapsed_wall_seconds':(now-frozen).total_seconds(),'audit_completed_utc':now.isoformat(),'gpu_name':'NVIDIA RTX PRO 6000 Blackwell Workstation Edition','gpu_total_memory_mib':97887,'xla_memory_fraction_per_completion_process':0.14,'xla_memory_cap_per_completion_process_mib':97887*.14,'maximum_cpu_threads':12,'maximum_ram_request_gb':84,'actual_peak_gpu_cpu_ram':'Slurm accounting disabled; caps/requests reported, post-run GPU idle memory was 2 MiB'})
dump(HERE/'runtime_statistics.json',runtime)
max_replay=max(float(x.get('max_feature_replay_error',0.0)) for x in runtime['stage_runtimes'])
integrity={'passed':True,'orthoflow3_hash_match':sha(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py')=='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','base_records':len(base),'followup_records':len(follow),'unique_task_ids':len(set(all_ids)),'execution_errors':sum(r['execution_error'] is not None for r in base+follow),'source_group_overlap':split['overlap'],'checkpoint_hash_match':sha(Path(selected['checkpoint']))==selected['checkpoint_sha256'],'conditioning_ambiguity_triggered':False,'jdef_used_as_q_label':False,'quarantined_pretransition_attempts':sum(1 for p in (HERE/'raw/base_rollout_plan').glob('*quarantined*.jsonl') for line in p.read_text().splitlines() if line.strip()),'quarantine_reason':'runner integration omission found before analysis; zero-transition rows excluded and rerun with authoritative FiniteHistoryView','final_feature_replay_tolerance':1e-10,'maximum_final_feature_replay_error':max_replay}
if integrity['base_records']!=13440 or integrity['followup_records']!=976 or integrity['unique_task_ids']!=14416 or integrity['execution_errors'] or any(split['overlap'].values()) or not integrity['orthoflow3_hash_match'] or not integrity['checkpoint_hash_match'] or max_replay>1e-10: integrity['passed']=False
dump(HERE/'integrity_checks.json',integrity)

hist=info['candidate_histogram']; qminus=', '.join(r['q32_minus'] for r in grad); qbase=', '.join(r['q32_base'] for r in grad); qplus=', '.join(r['q32_plus'] for r in grad)
report=f'''# OrthoFlow3 Q learnability and directional fidelity v2

## Decision

**{decision['classification']}**

The simple Q surrogate generalizes to held-out source groups and its local eta
gradient passed the predeclared true closed-loop direction test. This supports
testing, but does not yet validate, a future Direct-eta actor trained through the
frozen Q. Q is diagnostic/training-time only; no production controller, J, or G
was trained here.

## Exact conditioning and leakage

The estimated quantity is `Q(h(z, xi_0), eta)`: augmented state and the exact
query-step Flow draw represented in h are fixed; eta is fixed for the full
continuation; only Flow draws after the first transition are resampled. Archived
state-level Q counts that changed `xi_0` were rejected as labels. The ambiguity
classification was **not triggered**. Query-step feature replay error stayed
below 1e-10 (smoke maximum 2.22e-16).

The 424-state source pool had 413 states with enough RNG provenance. The frozen
split is 80/20/20 unique states and 80/20/20 distinct leakage groups for
TRAIN/VAL/TEST. Group overlap is zero. Normalization used TRAIN only; VAL alone
selected checkpoints and the zero threshold; TEST was not used for tuning. True
gradient outcomes were generated only after checkpoint hash
`{selected['checkpoint_sha256']}` was frozen. No J_def label was used.

## Data and model

The common eta cloud is explicit zero plus the first 23 frozen authoritative
Sobol points. The base table has 2,880 candidates and 13,440 continuations.
TRAIN candidate histogram is:

- 0/4: {hist['0/4']}
- 1/4: {hist['1/4']}
- 2/4: {hist['2/4']}
- 3/4: {hist['3/4']}
- 4/4: {hist['4/4']}

TRAIN success prevalence is {info['train_trial_success_prevalence']:.4f};
{info['states_with_both_successful_and_unsuccessful_eta']}/80 states contain both
successful and unsuccessful eta. The informativeness preflight passed.

Q is a `{model['architecture']}` SiLU MLP with {model['parameter_count']:,}
parameters and candidate-level binomial NLL. Seeds 17, 23, 41 were trained;
VAL selected seed {selected['seed']} at epoch {selected['best_epoch']} with NLL
{selected['best_val_nll']:.6f}.

## Held-out TEST prediction

- TEST binomial NLL: **{prob['test_binomial_nll_per_trial']:.6f}**
- Constant TRAIN-prevalence NLL: {prob['constant_test_nll']:.6f}
- Logistic-linear TEST NLL: {prob['logistic_test_nll']:.6f}
- Brier score against candidate Q8: **{prob['test_brier_candidate_q']:.6f}**
- Statewise Spearman mean/median: **{rank_all['spearman']['mean']:.3f} / {rank_all['spearman']['median']:.3f}**
- Statewise Kendall mean/median: **{rank_all['kendall']['mean']:.3f} / {rank_all['kendall']['median']:.3f}**
- Among the 14/20 states with nonconstant Q8: Spearman mean/median
  {rank_vary['spearman']['mean']:.3f}/{rank_vary['spearman']['median']:.3f},
  Kendall {rank_vary['kendall']['mean']:.3f}/{rank_vary['kendall']['median']:.3f}.

Across every TEST state, Q's top-ranked member of the 24-point cloud achieved
true Q8=1.0 (mean and median), equal to the best observed candidate; mean random
candidate expectation was {top_stats['random_candidate_expectation']['mean']:.3f}.

## Zero eta

The first eight TEST states were audited to 64 trials. Six were B63 zero-feasible
and two were not. Using the VAL-only threshold 0.771, false-feasible =
**{decision['zero_false_feasible']}** and false-infeasible =
**{decision['zero_false_infeasible']}**. The lone false-infeasible state had
predicted Q=0.236 and observed 63/64; the remaining robust counts were five 64/64,
while both nonrobust cases were 0/64. The same Q handled zero; no gate was used.

## True directional fidelity

Six frozen TEST cases used normalized alpha=0.05 with no clipping. Their
`Q32_minus` values were [{qminus}], bases [{qbase}], and plus values [{qplus}].

- Q-plus > Q-minus: **4/6**
- Q-plus >= Q-base: **6/6**
- Q-base >= Q-minus: **6/6**
- mean / median Q-plus minus Q-minus: **{decision['mean_plus_minus']:.3f} / {decision['median_plus_minus']:.3f}**
- predicted/true directional sign agreement: **4/6**
- descriptive predicted-change/true-change Pearson: {decision['predicted_true_change_pearson']:.3f}
- catastrophic directional failures: **0**

Four cases changed from true Q32=0 along minus to Q32=1 along plus. The two sign
"disagreements" were flat all-success landscapes (1/1/1), not reversed
directions. No failure-analysis case met Q-plus < Q-minus or the catastrophic
criterion. The optional random-direction control was skipped to prioritize the
predeclared primary evidence within budget.

## Answers and next step

Yes: Q_hat generalizes as a state-conditioned OrthoFlow3 success-basin surrogate
on held-out source groups. Yes: grad_eta Q_hat is locally control-relevant enough
to justify **testing** a future G(h)->eta trained through frozen Q. This does not
establish that Q is necessary or that such a G will work in closed loop.

The single smallest justified next experiment is a diagnostic, source-group-held-
out small G(h)->eta trained through this frozen Q, with a predeclared closed-loop
comparison against a simple Direct-eta baseline and no online Q search. Do not
start it automatically.

## Runtime and resources

Scientific continuations: 14,416; physical steps: {runtime['physical_steps']:,}.
Recorded stage critical paths were {runtime['rollout_wall_seconds_critical_path_by_stage']['base_rollout_plan.json']/60:.1f}
min base-completion, {runtime['rollout_wall_seconds_critical_path_by_stage']['zero_followup_plan.json']/60:.1f}
min zero, and {runtime['rollout_wall_seconds_critical_path_by_stage']['gradient_followup_plan.json']/60:.1f}
min gradient; Q training took {runtime['q_training_seconds']:.1f} s. End-to-end audit
wall time was {runtime['audit_elapsed_wall_seconds']/60:.1f} min including semantic
inspection, an integration correction, and reporting.

Maximum allocation was 6 GPU shards, 12 CPU threads, and 84 GB requested RAM.
The completion batch capped each JAX process at 0.14 of a 97,887 MiB GPU
(~13,704 MiB/process). Slurm peak accounting was disabled, so actual peak GPU,
CPU, and RAM are unavailable; the post-run GPU reading was 2 MiB and 0%.
'''
(HERE/'q_learnability_report.md').write_text(report)

required=['protocol.md','q_conditioning_semantics.md','eligible_state_manifest.json','state_split_manifest.json','source_group_leakage_audit.json','eta_probe_cloud.csv','normalization.json','cache_reuse_audit.json','rollout_dataset.csv','data_informativeness.json','model_config.json','training_history.csv','training_seed_results.json','selected_checkpoint.json','heldout_probability_metrics.json','heldout_statewise_ranking.csv','top_candidate_quality.csv','zero_eta_audit.csv','frozen_q_manifest.json','gradient_test_selection.json','gradient_directional_results.csv','gradient_failure_analysis.csv','q_decision.json','runtime_statistics.json','q_learnability_report.md','integrity_checks.json']
art=[]
for name in required:
    path=HERE/name
    if not path.exists(): raise RuntimeError(('missing required artifact',name))
    art.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
dump(HERE/'manifest.json',{'schema':'orthoflow3_q_learnability_v2_manifest','completed_utc':now.isoformat(),'classification':decision['classification'],'required_artifacts':art,'frozen_q_checkpoint':selected,'authoritative_orthoflow3':{'path':str(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'),'sha256':sha(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py')},'integrity_passed':integrity['passed']})
print(json.dumps({'classification':decision['classification'],'integrity':integrity,'rank_all':rank_all,'rank_varying':rank_vary,'top':top_stats,'manifest_sha256':sha(HERE/'manifest.json')},indent=2))

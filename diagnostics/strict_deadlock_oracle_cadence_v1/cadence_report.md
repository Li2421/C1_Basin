# Strict-deadlock fixed-eta oracle cadence audit

## Result

**ORACLE_CADENCE_MISMATCH_CONFIRMED**

This is a paired oracle-only cadence ablation. It uses the already selected minimum-deformation robust eta for each of the 17 frozen states; no eta search, learned G_phi, gate, or controller modification was performed. Dense H=1 was reproduced exactly before interpreting H=8.

## Integrity

- Frozen cases: 17 (historical 11, fresh unseen 6).
- Starting state: earliest robust queried augmented state (16 S0; old_r106 at S_8s), restored from the frozen full-state files.
- Eta: unchanged per-state minimum-J_def B63 eta from the capacity audit.
- Flow seeds: 64 matched continuations per state, seeds 95310001--95310064.
- Horizon: remaining global horizon through absolute step 850.
- Cadence phase: absolute global timestep; it is not reset at a queried state.
- Dense H=1 robust rollouts reused after exact state/eta/seed/code/horizon checks: 1088.
- Exact H=1 teacher replays matching outcome, terminal step, and J_def: 17/17.

## Exact inherited Flow

| cohort | episodes | H1 success | H8 success |
|---|---:|---:|---:|
| historical | 11 | 11 | 0 |
| fresh_unseen | 6 | 6 | 1 |
| combined | 17 | 17 | 1 |

## Matched 64-Flow robust result

| cohort | H1 success | H8 success | H1 Q | H8 Q | H8 B63 episodes | H1-success/H8-fail |
|---|---:|---:|---:|---:|---:|---:|
| historical | 704/704 | 16/704 | 1.0000 | 0.0227 | 0/11 | 688 |
| fresh_unseen | 384/384 | 66/384 | 1.0000 | 0.1719 | 0/6 | 318 |
| combined | 1088/1088 | 82/1088 | 1.0000 | 0.0754 | 0/17 | 1006 |

Combined Wilson 95% intervals: H1 [0.9965, 1.0000], H8 [0.0611, 0.0926].

Of the 1006 H8 robust-flow failures, 659 terminated as strict deadlock and 347 as timeout; there were 0 collisions. On inherited exact Flow, H8 produced 10 deadlocks, 6 timeouts, and 1 success.

## Per-episode robust counts

| case | cohort | state | H1/64 | H8/64 | classification |
|---|---|---|---:|---:|---|
| old_r002 | historical | old_r002__S0 | 64 | 1 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r018 | historical | old_r018__S0 | 64 | 0 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r029 | historical | old_r029__S0 | 64 | 2 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r052 | historical | old_r052__S0 | 64 | 1 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r096 | historical | old_r096__S0 | 64 | 2 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r106 | historical | old_r106__S_8s | 64 | 0 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r132 | historical | old_r132__S0 | 64 | 2 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r156 | historical | old_r156__S0 | 64 | 4 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r175 | historical | old_r175__S0 | 64 | 2 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r182 | historical | old_r182__S0 | 64 | 2 | SPARSE_CADENCE_BREAKS_ORACLE |
| old_r191 | historical | old_r191__S0 | 64 | 0 | SPARSE_CADENCE_BREAKS_ORACLE |
| fresh_r011 | fresh_unseen | fresh_r011__S0 | 64 | 1 | SPARSE_CADENCE_BREAKS_ORACLE |
| fresh_r058 | fresh_unseen | fresh_r058__S0 | 64 | 28 | SPARSE_CADENCE_BREAKS_ORACLE |
| fresh_r062 | fresh_unseen | fresh_r062__S0 | 64 | 0 | SPARSE_CADENCE_BREAKS_ORACLE |
| fresh_r114 | fresh_unseen | fresh_r114__S0 | 64 | 0 | SPARSE_CADENCE_BREAKS_ORACLE |
| fresh_r133 | fresh_unseen | fresh_r133__S0 | 64 | 34 | CADENCE_DEGRADED_BUT_USABLE |
| fresh_r139 | fresh_unseen | fresh_r139__S0 | 64 | 3 | SPARSE_CADENCE_BREAKS_ORACLE |

Classification counts: {'SPARSE_CADENCE_BREAKS_ORACLE': 16, 'CADENCE_DEGRADED_BUT_USABLE': 1}. B63 (>=63/64) is the sole robust threshold. Below B63, `CADENCE_DEGRADED_BUT_USABLE` is used only when a strict majority of matched continuations still succeeds; otherwise the condition is mostly failure and classified `SPARSE_CADENCE_BREAKS_ORACLE`. Raw counts above remain the primary evidence.

## Omitted correction and liveness

Across states, the mean counterfactual executed dense correction omitted on H8-off steps is 0.068467 m/s. The mean fraction of off steps with norm greater than the diagnostic numerical threshold 1e-6 is 1.0000. This threshold is reporting-only and never affects control.

Among 82 paired continuations successful under both conditions, H8 minus H1 completion time has mean 5.694 s, median 5.650 s, and P95 8.695 s. Failures that remain failures at the fixed horizon are not relabeled as merely slow.

## Deformation and projection

Across all matched continuations, mean J_def is 0.406249 for H1 and 0.031104 for H8 (ratio 0.0766). Lower H8 deformation is not interpreted as beneficial when task success is lost. Active-step correction and second-projection statistics are in `projection_analysis.csv`; omitted dense corrections are in `omitted_dense_correction_stats.csv`.

Mean active-step raw / executed / projection-rewrite norms are 0.449707 / 0.127804 / 0.370184 for H1 and 0.479123 / 0.069815 / 0.439551 for H8. Projection retry counts were zero in both conditions.

## Interpretation

Dominant remaining bottleneck: **temporal persistence / deployment cadence**.

Smallest justified next experiment: With eta and all controller semantics frozen, test one preregistered H=8 trigger followed by an 8-step dense fixed-eta burst on the same 17 states.

The aggregate conclusion was assigned from the preregistered semantics: broad H8 preservation requires nearly all episodes to remain B63 with near-dense aggregate success; coexistence of robust and cadence-sensitive subgroups is mixed; otherwise substantial dense-to-sparse rescue loss confirms cadence mismatch.

## Resource use

- New rollouts: 1122 (H8 robust plus exact H1/H8 trajectory replays).
- Reused validated H1 robust rollouts: 1088.
- New physical steps: 580205.
- Rollout wall time: 271.4 s (maximum shard).
- Resources: 2 GPU shards, 6 CPU cores requested, 48 GB memory requested.

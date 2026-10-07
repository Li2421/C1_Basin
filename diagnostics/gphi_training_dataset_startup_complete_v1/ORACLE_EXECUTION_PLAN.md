# Startup oracle execution plan (frozen, not yet launched)

## State rule

- Source episodes: frozen Safety-baseline rollout IDs `0..122`.
- One state per source episode.
- Physical step: `rollout_id mod 41`.
- Result: 123 independent source episodes and exactly three states at each
  physical step `0..40`.
- Selection is independent of terminal outcome, oracle result, geometry,
  category, previous gate errors, and recovery semantics.

The 26 source groups absent from V3 were assigned before oracle evaluation to
bring the combined V3+startup source-group inventory to 110/24/24
train/validation/test groups. Existing V3 groups inherit their original split.

## Frozen oracle contract

- Matched seeds: exactly the V3 64-seed list `95710001..95710064`.
- First evaluate `eta=(0,0,0)`.
- Only eta-zero failures receive the exact eight V3 nonzero candidate etas.
- A cell is B63-feasible only with a complete 64-seed cohort and at least 63
  successes.
- Non-feasible cells stop after the second physical failure.
- The minimum-success-deformation B63 cell and compact paired-CI `E_near` are
  reconstructed exactly as in V3.
- Target is mean `u_exec-u_safe` across `E_near` at the identical state and
  Flow seed.

## Maximum work and expected scale

- Eta-zero maximum: 7,872 continuations / 6,533,760 physical steps.
- Nonzero worst case: 62,976 continuations / 52,270,080 physical steps.
- Absolute worst case: 70,848 continuations / 58,803,840 physical steps.

The actual count should be far smaller because startup states are expected to
be predominantly eta-zero feasible and every infeasible eta arm stops after
two failures. Based on V1--V3 throughput, a practical estimate is roughly
10--30 minutes for eta-zero and 10--60 additional minutes for candidates on
one shard; two permitted shards should approximately halve the rollout wall
time. These are planning estimates, not measured results.

## Smoke-tested commands

```bash
export PYTHONPATH=/home/zhihan/research/Basin_C1:/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_startup_complete_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OUT=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_startup_complete_v1

$PY $OUT/freeze_startup_states.py
$PY $OUT/audit_startup_states.py
$PY $OUT/audit_startup_feature_builder.py
$PY $OUT/plan_oracle.py --phase zero
$PY $OUT/smoke_test_pipeline.py
```

No continuation rollout is performed by those commands. After the parent
checks server usage, partition and run the frozen arms:

```bash
$PY $OUT/split_arms.py --arms eta_zero_arms.json --shards 2 --prefix eta_zero
$PY $OUT/run_oracle.py --arms eta_zero_shard0_arms.json --stage eta_zero_shard0 --device gpu --batch 32
$PY $OUT/run_oracle.py --arms eta_zero_shard1_arms.json --stage eta_zero_shard1 --device gpu --batch 32

$PY $OUT/plan_oracle.py --phase candidates
$PY $OUT/split_arms.py --arms candidate_arms.json --shards 2 --prefix candidate
# Run the two candidate partitions analogously.
$PY $OUT/plan_oracle.py --phase assessment
$PY $OUT/finalize_startup_dataset.py
```

The shard runners are restart-safe, hash-bind their arms/protocol, flush after
every batch, and never repeat a tuple already present in the isolated cache.

# Part 3 Round 2 execution log — 2026-10-12

## Evidence and freeze, before new labels

1. Audited P1 controller/checkpoint inventory, `P1_LABELS.npz`, true-t0 state generator, official cache README, archived FF adapter and runner. The selected pair was already fixed by the user. η was ranked from P1 seeds 0–1 only; `η₁/η₂` won the frozen score. The five sequential generated state UIDs **and content hashes** were read-only checked absent from `shared_rollout_db/rollout.sqlite` before experiment registration.
2. Wrote `PREREGISTRATION.md`, then ran:

   ```bash
   /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_necessity_representation_round2_20261012/freeze.py
   ```

   Frozen `protocol.json` SHA256: `c7a09486d9f88c82bde0b37368819cecc5759e3199d211127669494924282873`. `freeze_hashes.sha256` records preregistration, case and planned-request hashes. The `freeze.py` replay is idempotent after registration and verifies registered physical content.
3. Official preflight before any continuation:

   ```bash
   /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m shared_rollout_db.plan --manifest diagnostics/part3_necessity_representation_round2_20261012/planned_rollouts.json --output diagnostics/part3_necessity_representation_round2_20261012/cache_preflight.json
   ```

   `total_requested=320`, `genuinely_missing=320`, `exact_reusable=0`, `ambiguous=0`. No cached label or aggregate was silently repurposed.

## Execution, merge, and postflight

```bash
sbatch --parsable diagnostics/part3_necessity_representation_round2_20261012/rollout.slurm
# job 19417: four CPU workers, array 0–3, max concurrency 4
sbatch --parsable --dependency=afterok:19417 diagnostics/part3_necessity_representation_round2_20261012/merge.slurm
# job 19421: one CPU merge/postflight after every worker succeeded
```

Each worker wrote immutable `raw/roll_*.json` records and append-only `shared_rollout_db/journals/exp_part3_round2_ring_real_flow_q16_v1/` shards. Four workers produced exactly 80 new raw records each. Aggregate measured simulator time: 1,684.18 CPU seconds. The single-process merger in `run.py` follows the existing P1 exact-identity journal merger: it checks frozen `case`, controller/checkpoint, seed key and outcome before inserting into the authoritative SQLite cache. This experiment uses the P1-style merger rather than the generic `shared_rollout_db.src.merger` because the generic historical-ingest path normalizes `future_index` differently from P1's exact root/namespace seed key; changing that identity would break exact reuse. No journal or DB row was rewritten to conceal an unknown result.

Merge: 40 journals, 320 inserted, 0 duplicate. Database postflight: 320 total, 318 valid, 2 numerical-unknown, 0 missing, 0 quarantine. Postflight re-run of the official planner reports 288 fully exact cell seeds plus 30 partial-cell reusable seeds; the 2 numerical-unknown seed keys remain nonreusable under policy. The 320 physical continuations were not exceeded.

## Outcome-free representation work

```bash
sbatch --parsable diagnostics/part3_necessity_representation_round2_20261012/features.slurm
# job 19422: one GPU shard, two CPU cores, 14 checkpoints × 24 old P1 states
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_necessity_representation_round2_20261012/representation_analysis.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_necessity_representation_round2_20261012/alias_counterexample.py
```

Feature job used 0 simulator steps and 0 outcome reads; elapsed 39.63 seconds. It replayed archived R3 to max `2.38e−7`. `representation_analysis.py` reads only the old P1 Q4 labels for exploratory scoring after the static descriptors exist; held checkpoint labels are never used for LOCO feature scaling or neighbor choice. `alias_counterexample.py` is explicitly a synthetic tensor permutation, not another checkpoint.

## Frozen analysis

```bash
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_necessity_representation_round2_20261012/analyze.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m py_compile diagnostics/part3_necessity_representation_round2_20261012/*.py
```

`necessity_results.json` contains all five states, 20 Q16 cells, uncertainty, opposite-preference and crossfit measures; `analysis.log` has the summary. `representation_results.json` includes all 14 held-checkpoint nearest-source rows and 24 discovery-pair state rows; `representation_analysis.log` has the summary. The preregistered empirical gates were not changed after seeing outcomes. The later hierarchical paired-seed bootstrap is clearly identified as supplemental sensitivity and does not enter the gates.

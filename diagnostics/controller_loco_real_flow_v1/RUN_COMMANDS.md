# Reproduction commands and execution order

From `/home/zhihan/research/Basin_C1`, use `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python` as `PY`. Read `PREREGISTRATION_STAGE_A.md`, its receipt, `PREREGISTRATION_RESOURCE_SUPPLEMENT.md`, `HISTORICAL_EVIDENCE_AUDIT.json`, and `ENVIRONMENT.json` first. Source runtime and checkpoints are external to this directory but their exact paths/hashes are in `protocol.json` and `ENVIRONMENT.json`.

1. `PY diagnostics/controller_loco_real_flow_v1/design.py create` freezes the 192 states, seven profiles, balanced η masks and 86,016 exact requests. `STATE_DISJOINT_RECEIPT.json` records no historical state overlap.
2. `PY -m shared_rollout_db.plan --manifest diagnostics/controller_loco_real_flow_v1/PLANNED_ROLLOUTS.json --output diagnostics/controller_loco_real_flow_v1/CACHE_PREFLIGHT.json`; inspect `CACHE_INVENTORY.json` and `COST_GATE.json`. `PY diagnostics/controller_loco_real_flow_v1/design.py register` registered the experiment **before merge**.
3. Static features: `PY diagnostics/controller_loco_real_flow_v1/features.py 0` through `6` in Slurm GPU shards. Actual Slurm array job 19355 (indices 0–6; index 4 repaired as 19362); all static queries forbid `env.step`.
4. Timed pilot: `PY diagnostics/controller_loco_real_flow_v1/run.py pilot` (Slurm 19354, resumed as 19363). Its 98 exact UIDs are in `PILOT_CASE_UIDS.json`; the sampling deviation is separately documented. The 22-CPU cost gate did not require shrinking test states.
5. Missing-only workers: `PY diagnostics/controller_loco_real_flow_v1/run.py worker --index I --workers 19` for I=0,…,18. Initial contiguous array 19364 was stopped for an outcome-blind load rebalance; `SCHEDULING_ADJUSTMENT.md` records why. Replacement array **19386** uses round-robin frozen UIDs with at most 19 one-CPU workers. Each writes immutable raw JSON and append-only shared-DB journal. Resumption checks the exact case before skipping an existing raw record.
6. After all workers: `PY diagnostics/controller_loco_real_flow_v1/run.py merge` (single merger), then `postflight`, `PY diagnostics/controller_loco_real_flow_v1/labels.py`, and `PY diagnostics/controller_loco_real_flow_v1/gate.py`; replacement dependent Slurm job **19405**.
7. GPU fits: `PY diagnostics/controller_loco_real_flow_v1/train.py FOLD ARM` for 0…6 and `R0|R3|R4_action|R4_openloop|R3_R4_action`, 3 fit seeds within each run; replacement array **19406** with at most five `shard:1` jobs. Each saves checkpoint, normalization, predictions and validation NLL history.
8. `PY diagnostics/controller_loco_real_flow_v1/analyze.py` computes held-controller seed4–7 selection, paired unknown lower bounds and simultaneous max-T intervals; replacement dependent Slurm job **19407**. Conditional Stage B/C design and branch rules are frozen in `PREREGISTRATION_CONDITIONAL_BRANCHES.md`. Slurm job **19409** depends on successful job 19407 and runs `dispatch_conditional.py`, which reads the frozen gate and primary result, records `CONDITIONAL_DISPATCH.json`, and submits only the applicable Stage B or C dependency chain. If fewer than two folds are necessary, it submits no additional rollout. After the applicable branch, `render_report.py` writes the final REPORT/HANDOFF from machine-readable results, then `final_audit.py` checks their provenance and consistency. All downstream jobs use `afterok` dependencies.

If Stage B is triggered, run `stage_b_prepare.sh` (which does design → official preflight → inventory → DB experiment registration), then `stage_b_features.py 0…6` in at most five GPU shards and `stage_b_run.py worker --index I --workers 19` in at most 19 CPU workers. After all workers, run `finalize_stage_b.sh` (single merger, postflight, exact labels). Run `train.py FOLD ARM --stage-b` for each necessary fold and the five arms in at most five GPU shards, then `analyze.py --stage-b` and `stage_b_compare.py`. Stage B uses the already opened Stage A held panel and is DEVELOPMENT.

## Actual Stage C after Stage B

Read `PREREGISTRATION_STAGE_C_AFTER_B.md` and its UTC/SHA256 receipt. This prospective addendum restores the user's confirmation condition after Stage B's development pass. Exact design/preflight/registration used `bash stage_c_prepare.sh`; 8,192 missing requests and no reuse/conflict. Actual scheduling is `STAGE_C_AFTER_B_DISPATCH.json` (jobs19507–19513), modified by `STAGE_C_RESOURCE_UPDATE.json` to infer frozen models before labels complete, with inference/merger both required by analysis. Seven feature tasks reused `stage_c_features.py`; 19 CPU workers used `stage_c_run.py worker --index I --workers 19`; `finalize_stage_c.sh` single-merges, postflights and writes labels. Six frozen model tasks are listed in `stage_c/PREDICTION_TASKS.json`; `stage_c_predict.py FOLD ARM` uses Stage B checkpoints and reads no labels. No Stage C fitting occurred. Run `stage_c_analyze.py` to recompute the independent confirmation (seed4–7 endpoints, 5,000 paired max-T draws).

## Recompute final analyses without rollout or training

From repository root, with the stated Python:

```bash
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
BASE=diagnostics/controller_loco_real_flow_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" "$BASE/stage_c_analyze.py"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" "$BASE/controller_prior_analyze.py"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" "$BASE/controller_q_decomposition.py"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" "$BASE/render_report.py"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" "$BASE/final_audit.py"
```

`render_report.py` uses frozen Stage A results, Stage B/C results and auxiliary summaries, renders PNG/PDF in `figures/`, and aggregates `results.json` while preserving Stage A top-level fields. `PLOT_DEPENDENCIES.json` records isolated `plot_vendor` packages; the shared Python environment was not modified. The old unknown-as-zero auxiliary energy analysis was corrected to interval sensitivity; see `DECOMPOSITION_UNKNOWN_POLICY_AUDIT.md` and preserved backup. `source_snapshots/` and `SOURCE_SNAPSHOT_INDEX.json` preserve nine external implementation/data dependencies; full Flow checkpoints remain at the exact hash-checked paths in protocols. `RAW_CONTENT_INDEX_STAGE_AB.jsonl` and `_C.jsonl` contain every raw file SHA256 (jobs19588/19589); generated raw is never rerun.

Auxiliary controller-prior inference used `controller_prior_predict.py TASK_INDEX` with tasks in `controller_prior/PREDICTION_TASKS.json` (GPU array19543, concurrency upper8 subject to available CPU slots). It chooses one η from the average frozen Q over 128 source initial states queried with held-controller descriptors, without any target label. The auxiliary comparison is EXPLORATORY and does not replace the main gate.

If Stage C is triggered instead, run `stage_c_prepare.sh` (same preflight/registration order), `stage_c_features.py 0…6`, and `stage_c_run.py worker --index I --workers 19`; then `finalize_stage_c.sh`. Predict using `stage_c_predict.py FOLD ARM` for every necessary fold and `R0` plus each Stage A passing arm, with frozen Stage A checkpoints; finish with `stage_c_analyze.py`. The budget-limited allocation is frozen in `stage_c/protocol.json`; no Stage C labels enter training.

Do not rerun numerical-unknown rollout UIDs. The shared DB is an index/cache; experiment raw and journals remain provenance. Stage A is pure Q and requires no video under `AGENTS.md`.

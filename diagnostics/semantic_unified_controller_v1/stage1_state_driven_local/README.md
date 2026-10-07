# Stage 1: state-driven local correction

This directory implements only the semantic `S` versus `L` experiment.  It
reuses the frozen generic Safety anchors and complete augmented-state I/O from
`single_segment_recovery_training_v1`.  No rollout or training is launched by
importing any module.

Semantics:

- `S`: execute Safety now, then use the same downstream S/L policy.
- `L`: query frozen Direct-g `c29800f6...` once and execute the projected local
  correction now, then use the same downstream S/L policy.
- A new state decision is made on every later transition.  There is no period,
  phase, cooldown, burst, or recovery mode.

Pipeline interfaces:

1. `build_local_branch_manifest.py --stage local_bootstrap_safety ...`
2. `run_local_branches.py --manifest ... --shard-index ...`
3. `finalize_local_results.py --manifest ... --result-directory ...`
4. `train_local_head_grid.py --dataset-manifest ... --execute`
5. Repeat 1--4 with `local_second_pass --downstream-checkpoint ...`, then use
   the existing audited `merge_finalized_head_datasets.py` to aggregate both
   immutable passes without erasing target lineage.
6. `build_local_full_loop_manifest.py` and `run_local_full_loop.py` evaluate
   complete development episodes.  `H8_REFERENCE` is supported only as an
   external comparator. `analyze_local_full_loop.py` reports Q, timeout and
   strict-deadlock rescue, break, deformation, event rate, and emergent event
   spacing.

All branch manifests freeze 32 matched future streams/input, keep root-source
splits intact, and fail before exceeding 20,000 continuations or 10 million
maximum physical steps.  This Stage-1 implementation intentionally forbids the
final-test split.

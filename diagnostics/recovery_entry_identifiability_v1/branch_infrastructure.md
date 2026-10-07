# Matched branch infrastructure

Status: implemented and CPU-unit-tested; **no branch rollout has been launched**.

## Frozen semantics

- `N` restores the queried augmented state and executes Safety until the
  original absolute horizon or an earlier terminal event.
- `R` restores the identical state, forces persistent structured-eta entry on
  that physical step, predicts and latches eta once, uses the authoritative
  learned exit head beginning on the next step, and executes Safety forever
  after exit. Re-entry is impossible.
- Both members of a pair replay the saved current Flow key and use the same
  future stream ID and per-absolute-step Flow keys.
- The final deployment exit override is frozen at probability `0.25`, logit
  `-1.0986122886681098`. The checkpoint's embedded training-local threshold is
  deliberately ignored.

## Staged commands

These commands document the handoff and do not run automatically.

```bash
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11
HERE=/home/zhihan/research/Basin_C1/diagnostics/recovery_entry_identifiability_v1

$PY "$HERE/build_branch_manifest.py" initial \
  --state-manifest "$HERE/queried_state_manifest.json" \
  --output "$HERE/branch_seed_manifest.json"

# Explicit execution, if separately authorized, is sharded with
# run_paired_branches.py --manifest ... --result-directory ... --shard-index ...

$PY "$HERE/finalize_paired_branches.py" \
  --manifest "$HERE/branch_seed_manifest.json" \
  --result-directory "$HERE/runs/branches_initial" \
  --output-directory "$HERE/initial_branch_summary"

$PY "$HERE/build_branch_manifest.py" confirmation \
  --initial-manifest "$HERE/branch_seed_manifest.json" \
  --initial-label-uncertainty "$HERE/initial_branch_summary/label_uncertainty.csv" \
  --output "$HERE/branch_seed_manifest_confirmation.json"
```

The confirmation builder requires the initial finalizer's content-hashed
lineage record. It selects at most 135 eligible states, adds exactly future
indices 16--31, and fails if initial plus confirmation work exceeds 12,000
branch continuations. It never launches the confirmation wave.

For final aggregation, pass initial and confirmation `--manifest` and
`--result-directory` arguments in that order. The finalizer rejects missing,
extra, duplicated, unpaired, errored, horizon-reset, replay-mismatched, or
controller-semantics-violating results.

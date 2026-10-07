#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhihan/research/Basin_C1
EXP="$ROOT/diagnostics/intervention_window_and_gate_v1"
STAGE1="$EXP/stage1_window"
STAGE2="$EXP/stage2_gate"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python

export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

stamp() {
  date --iso-8601=seconds
}

note() {
  echo "[$(stamp)] $*"
}

wait_for_job() {
  local job_id="$1"
  local label="$2"
  while squeue -h -j "$job_id" | grep -q .; do
    note "waiting for $label job $job_id"
    sleep 30
  done
  note "$label job $job_id left scheduler"
}

cd "$ROOT"
note "pipeline supervisor started"
note "waiting for preregistered coarse jobs 262 and 263"
wait_for_job 262 coarse_short
wait_for_job 263 coarse_long

test -s "$STAGE1/raw/coarse_short_shard0/manifest.json"
test -s "$STAGE1/raw/coarse_long_shard0/manifest.json"
note "running frozen coarse statistics"
"$PY" "$STAGE1/stage1_statistics.py" coarse

note "preparing adaptive jobs"
"$PY" "$STAGE1/adaptive_job_orchestrator.py" prepare
adaptive_count=$("$PY" -c "import json; print(json.load(open('$STAGE1/adaptive_job_manifest.json'))['job_count'])")
if [ "$adaptive_count" -gt 0 ]; then
  adaptive_job=$(sbatch --parsable "$STAGE1/run_adaptive_jobs.sh")
  note "submitted adaptive job $adaptive_job with $adaptive_count stages"
  wait_for_job "$adaptive_job" adaptive_stage1
  test -s "$STAGE1/adaptive_execution_complete.json"
else
  note "no adaptive rollout jobs required"
fi

# The original adaptive selection is intentionally immutable.  Follow-up
# rounds consume its completed evidence and only add missing tuples needed to
# advance transition-obstructing ambiguity from 64 -> 128 -> 256.  Replanning
# after each completed round also admits states whose d64 result only becomes
# resolved-safe after escalation into the d128 extension.  The loop stops only
# at a hash-bound no-job convergence manifest; it never finalizes merely due to
# a fixed iteration budget.
followup_round=1
while true; do
  if [ "$followup_round" -gt 32 ]; then
    note "follow-up exceeded 32 rounds without certified convergence"
    exit 1
  fi
  note "preparing Stage 1 adaptive follow-up round $followup_round"
  "$PY" "$STAGE1/adaptive_followup_orchestrator.py" prepare --round "$followup_round"
  followup_manifest="$STAGE1/adaptive_followup/round_$(printf '%02d' "$followup_round")/job_manifest.json"
  followup_count=$(
    "$PY" -c "import json; print(json.load(open('$followup_manifest'))['job_count'])"
  )
  if [ "$followup_count" -eq 0 ]; then
    note "adaptive follow-up converged in planning round $followup_round"
    "$PY" "$STAGE1/adaptive_followup_orchestrator.py" verify-convergence
    break
  fi
  followup_job=$(sbatch --parsable --export=ALL,FOLLOWUP_ROUND="$followup_round" "$STAGE1/run_adaptive_followup_round.sh")
  note "submitted follow-up job $followup_job for round $followup_round ($followup_count sequential stages)"
  wait_for_job "$followup_job" "adaptive_followup_round_$followup_round"
  followup_complete="$STAGE1/adaptive_followup/round_$(printf '%02d' "$followup_round")/execution_complete.json"
  test -s "$followup_complete"
  followup_round=$((followup_round + 1))
done

note "finalizing Stage 1 and freezing candidate horizons"
"$PY" "$STAGE1/stage1_statistics.py" final
test -s "$STAGE1/candidate_windows.json"

# USER-REQUESTED TERMINAL CONDITION (2026-09-24): Stage 1 is the complete
# task for this run.  Preserve the already implemented Stage-2 code below,
# but do not prepare targets, launch training, or consume any Stage-2 compute.
note "writing Stage-1-only parent handoff; Stage 2 explicitly disabled"
"$PY" "$EXP/synthesize_stage1_only.py"
test -s "$EXP/parent_summary.md"
test -s "$EXP/manifest.json"
note "Stage-1-only pipeline complete"
exit 0

note "freezing Stage 2 targets and LOGO folds"
"$PY" "$STAGE2/prepare_stage2.py"
"$PY" "$STAGE2/build_training_jobs.py"
test -s "$STAGE2/TRAINING_READY.json"
test -s "$STAGE2/training_jobs.json"

worker0=$(sbatch --parsable "$STAGE2/run_worker0.sh")
worker1=$(sbatch --parsable "$STAGE2/run_worker1.sh")
note "submitted Stage 2 workers $worker0 and $worker1"
wait_for_job "$worker0" stage2_worker0
wait_for_job "$worker1" stage2_worker1

note "finalizing Stage 2"
"$PY" "$STAGE2/finalize_stage2.py"
note "synthesizing parent report"
"$PY" "$EXP/synthesize_results.py"
test -s "$EXP/parent_summary.md"
test -s "$EXP/manifest.json"
note "pipeline complete"

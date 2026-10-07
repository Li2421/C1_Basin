#!/usr/bin/env bash
#SBATCH --job-name=iwg-stage1-long
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/intervention_window_and_gate_v1/stage1_window/slurm-long-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/intervention_window_and_gate_v1/stage1_window/slurm-long-%j.err

set -euo pipefail

AUDIT_ROOT=/home/zhihan/research/Basin_C1
AUDIT_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
STAGE_DIR="$AUDIT_ROOT/diagnostics/intervention_window_and_gate_v1/stage1_window"

# These hashes bind this launcher to the frozen 277-state plan and common
# resumable runner.  Refuse execution if either file is changed in place.
EXPECTED_PLAN_SHA=0d4a8ac8696b5284c577d764c28c5200e776b26aa8ce400724575f6ba5741a0f
EXPECTED_INDEX_SHA=c30a73b57bf77d75555cb6495be725f203d93c92a5aafc831b9d24fa32bc0ab9
EXPECTED_RUNNER_SHA=517f408b08572dd3b271cabf715d61a23d49e21474ccf8ac1e516c3ef06f5e55

test "$(sha256sum "$STAGE_DIR/audit_plan.json" | awk '{print $1}')" = "$EXPECTED_PLAN_SHA"
test "$(sha256sum "$STAGE_DIR/tuple_index.jsonl" | awk '{print $1}')" = "$EXPECTED_INDEX_SHA"
test "$(sha256sum "$STAGE_DIR/run_delay_rollouts.py" | awk '{print $1}')" = "$EXPECTED_RUNNER_SHA"

export PYTHONPATH="$AUDIT_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

cd "$AUDIT_ROOT"

# The common runner is append-only and fsyncs every completed batch.  Calling
# this exact command again resumes missing tuples and rejects a changed spec.
"$AUDIT_PY" "$STAGE_DIR/run_delay_rollouts.py" \
  --delays 32,64 --stage coarse_long --shard 0 --shards 1 \
  --device gpu --batch 16

"$AUDIT_PY" "$STAGE_DIR/check_completed_stage.py" \
  --stage coarse_long --shard 0 --expected-delays 32,64 --expected-new 15360

"$AUDIT_PY" "$STAGE_DIR/check_coarse_long_integrity.py"

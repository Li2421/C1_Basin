#!/usr/bin/env bash
#SBATCH --job-name=gphi-v2-oracle
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v2/logs/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v2/logs/slurm-%j.err

set -euo pipefail

ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OUT="$ROOT/diagnostics/gphi_training_dataset_v2"
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

run_pair() {
  local phase="$1"
  local reuse="$2"
  "$PY" "$OUT/adaptive_run.py" \
    --arms "${phase}_shard0_arms.json" --stage "${phase}_shard0" \
    --device gpu --batch 64 --reuse-jsonl "$reuse" \
    >"$OUT/logs/${phase}_shard0.out" 2>"$OUT/logs/${phase}_shard0.err" &
  local pid0=$!
  "$PY" "$OUT/adaptive_run.py" \
    --arms "${phase}_shard1_arms.json" --stage "${phase}_shard1" \
    --device gpu --batch 64 --reuse-jsonl "$reuse" \
    >"$OUT/logs/${phase}_shard1.out" 2>"$OUT/logs/${phase}_shard1.err" &
  local pid1=$!
  local status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  return "$status"
}

run_pair baseline "$OUT/v1_reuse.jsonl"
"$PY" "$OUT/plan_candidates.py"
run_pair candidate "$OUT/candidate_reuse.jsonl"
"$PY" "$OUT/assess_oracle.py"


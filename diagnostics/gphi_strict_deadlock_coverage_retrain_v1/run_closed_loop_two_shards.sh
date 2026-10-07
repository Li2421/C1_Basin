#!/usr/bin/env bash
#SBATCH --job-name=gphi-sd-eval
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_strict_deadlock_coverage_retrain_v1/eval-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_strict_deadlock_coverage_retrain_v1/eval-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
HERE="$ROOT/diagnostics/gphi_strict_deadlock_coverage_retrain_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$ROOT:$HERE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.28
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd "$ROOT"
pids=()
for shard in 0 1; do
  "$PY" "$HERE/run_closed_loop.py" --suite both --shard-index "$shard" --shard-count 2 --device gpu \
    >"$HERE/eval_shard${shard}.out" 2>"$HERE/eval_shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"

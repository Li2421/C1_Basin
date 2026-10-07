#!/usr/bin/env bash
#SBATCH --job-name=gphi-h8-fresh
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_h8_fresh_unseen_generalization_v1/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_h8_fresh_unseen_generalization_v1/slurm-%j.err

set -euo pipefail

ROOT=/home/zhihan/research/Basin_C1
OUT="$ROOT/diagnostics/gphi_h8_fresh_unseen_generalization_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python

export PYTHONPATH="$ROOT:$OUT:$ROOT/diagnostics/gphi_wide_ic_cadence_v1:$ROOT/diagnostics/gphi_closed_loop_pilot_v1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

pids=()
for worker in 0 1; do
  start=$((worker * 100))
  stop=$(((worker + 1) * 100))
  "$PY" "$OUT/run_fresh_evaluation.py" \
    --controller both \
    --device gpu \
    --start-index "$start" \
    --stop-index "$stop" \
    >"$OUT/fresh_shard${worker}.out" \
    2>"$OUT/fresh_shard${worker}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"

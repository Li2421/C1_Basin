#!/usr/bin/env bash
#SBATCH --job-name=eta-deadlock-exact6
#SBATCH --partition=gpu
#SBATCH --gres=shard:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=70G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/strict_deadlock_success_basin_capacity_v1/exact6-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/strict_deadlock_success_basin_capacity_v1/exact6-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
OUT="$ROOT/diagnostics/strict_deadlock_success_basin_capacity_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:$ROOT:$OUT
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.11
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
pids=()
for shard in 0 1 2 3 4 5; do
  "$PY" "$OUT/run_capacity.py" --mode exact --shard-index "$shard" --shard-count 6 --device gpu --batch 64 \
    >"$OUT/exact6_shard${shard}.out" 2>"$OUT/exact6_shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"

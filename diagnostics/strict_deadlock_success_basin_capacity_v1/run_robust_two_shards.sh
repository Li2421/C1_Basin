#!/usr/bin/env bash
#SBATCH --job-name=eta-deadlock-robust
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=56G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/strict_deadlock_success_basin_capacity_v1/robust-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/strict_deadlock_success_basin_capacity_v1/robust-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
OUT="$ROOT/diagnostics/strict_deadlock_success_basin_capacity_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:$ROOT:$OUT
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT"
pids=()
for shard in 0 1; do
  "$PY" "$OUT/run_capacity.py" --mode robust --shard-index "$shard" --shard-count 2 --device gpu --batch 64 \
    >"$OUT/robust_shard${shard}.out" 2>"$OUT/robust_shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"

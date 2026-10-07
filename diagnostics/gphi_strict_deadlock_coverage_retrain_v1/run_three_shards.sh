#!/usr/bin/env bash
#SBATCH --job-name=gphi-sd-cover
#SBATCH --partition=gpu
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_strict_deadlock_coverage_retrain_v1/train-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_strict_deadlock_coverage_retrain_v1/train-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
HERE="$ROOT/diagnostics/gphi_strict_deadlock_coverage_retrain_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.18
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
pids=()
for seed in 17 23 41; do
  "$PY" "$HERE/run_seed.py" --seed "$seed" >"$HERE/seed${seed}.out" 2>"$HERE/seed${seed}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"

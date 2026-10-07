#!/usr/bin/env bash
#SBATCH --job-name=gphi-extended-horizon
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_extended_horizon_audit_v1/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_extended_horizon_audit_v1/slurm-%j.err

set -euo pipefail

ROOT=/home/zhihan/research/Basin_C1
OUT="$ROOT/diagnostics/gphi_extended_horizon_audit_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python

export PYTHONPATH="$ROOT:$OUT:$ROOT/diagnostics/gphi_wide_ic_cadence_v1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

"$PY" "$OUT/run_extended.py" --device gpu --start-task 0 --stop-task 79 \
  >"$OUT/worker0.out" 2>"$OUT/worker0.err" &
pid0=$!
"$PY" "$OUT/run_extended.py" --device gpu --start-task 79 --stop-task 157 \
  >"$OUT/worker1.out" 2>"$OUT/worker1.err" &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PY" "$OUT/analyze_extended.py"

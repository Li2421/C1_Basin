#!/usr/bin/env bash
#SBATCH --job-name=gphi-wide-cadence
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_wide_ic_cadence_v1/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_wide_ic_cadence_v1/slurm-%j.err

set -euo pipefail

ROOT=/home/zhihan/research/Basin_C1
OUT="$ROOT/diagnostics/gphi_wide_ic_cadence_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python

export PYTHONPATH="$ROOT:$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.14
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

starts=(0 50 100 150)
stops=(50 100 150 200)
pids=()

for worker in 0 1 2 3; do
  "$PY" "$OUT/run_evaluation.py" \
    --controller all \
    --namespace production \
    --device gpu \
    --start-index "${starts[$worker]}" \
    --stop-index "${stops[$worker]}" \
    >"$OUT/wide_shard${worker}.out" \
    2>"$OUT/wide_shard${worker}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PY" "$OUT/analyze_results.py" \
  --episodes 200 \
  --bootstrap-replicates 200000 \
  --bootstrap-seed 20260925

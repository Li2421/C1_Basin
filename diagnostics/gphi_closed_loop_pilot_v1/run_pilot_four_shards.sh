#!/usr/bin/env bash
#SBATCH --job-name=gphi-full-pilot
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_closed_loop_pilot_v1/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_closed_loop_pilot_v1/slurm-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
OUT="$ROOT/diagnostics/gphi_closed_loop_pilot_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$ROOT:$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.14
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
starts=(0 32 64 96)
stops=(32 64 96 128)
pids=()
for shard in 0 1 2 3; do
  "$PY" "$OUT/run_evaluation.py" \
    --controller both --namespace production --device gpu \
    --start-index "${starts[$shard]}" --stop-index "${stops[$shard]}" \
    >"$OUT/pilot_shard${shard}.out" \
    2>"$OUT/pilot_shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
"$PY" "$OUT/analyze_results.py" --episodes 128 --namespace production --bootstrap 100000

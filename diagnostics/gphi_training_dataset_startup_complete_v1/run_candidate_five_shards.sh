#!/usr/bin/env bash
#SBATCH --job-name=gphi-startup-cand
#SBATCH --partition=gpu
#SBATCH --gres=shard:5
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_startup_complete_v1/logs/candidate-slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_startup_complete_v1/logs/candidate-slurm-%j.err

set -euo pipefail

ROOT=/home/zhihan/research/Basin_C1
OUT="$ROOT/diagnostics/gphi_training_dataset_startup_complete_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$ROOT:$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUT/logs"
cd "$OUT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

pids=()
for shard in 0 1 2 3 4; do
  "$PY" "$OUT/run_oracle.py" \
    --arms "candidate_shard${shard}_arms.json" \
    --stage "candidate_shard${shard}" \
    --device gpu --batch 32 \
    >"$OUT/logs/candidate_shard${shard}.out" \
    2>"$OUT/logs/candidate_shard${shard}.err" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"

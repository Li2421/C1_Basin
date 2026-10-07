#!/usr/bin/env bash
#SBATCH --job-name=gphi-v4-nonzero
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v4/slurm-nonzero-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v4/slurm-nonzero-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OUT="$ROOT/diagnostics/gphi_training_dataset_v4"
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
"$PY" "$OUT/adaptive_run.py" --arms nonzero_candidate_shard0_arms.json --stage nonzero_candidate_shard0 --device gpu --batch 64 &
pid0=$!
"$PY" "$OUT/adaptive_run.py" --arms nonzero_candidate_shard1_arms.json --stage nonzero_candidate_shard1 --device gpu --batch 64 &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
exit "$status"

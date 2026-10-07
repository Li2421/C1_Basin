#!/usr/bin/env bash
#SBATCH --job-name=gphi-v3-zero
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v3/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v3/slurm-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OUT="$ROOT/diagnostics/gphi_training_dataset_v3"
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
"$PY" "$OUT/collect_candidates.py" --device gpu --per-group 8
"$PY" "$OUT/audit_candidates.py"
"$PY" "$OUT/plan_zero_oracle.py"
"$PY" "$OUT/adaptive_run.py" --arms zero_oracle_arms.json --stage zero_oracle --device gpu --batch 64

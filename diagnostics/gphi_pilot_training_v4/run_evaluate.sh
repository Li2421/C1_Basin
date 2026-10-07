#!/usr/bin/env bash
#SBATCH --job-name=gphi-v4-eval
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_pilot_training_v4/slurm-eval-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_pilot_training_v4/slurm-eval-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
"$PY" diagnostics/gphi_pilot_training_v4/evaluate.py

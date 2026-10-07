#!/usr/bin/env bash
#SBATCH --job-name=hard-stable-cv
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/hard_stable_boundary_crossval/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/hard_stable_boundary_crossval/slurm-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
"$PY" diagnostics/hard_stable_boundary_crossval/run_crossval.py

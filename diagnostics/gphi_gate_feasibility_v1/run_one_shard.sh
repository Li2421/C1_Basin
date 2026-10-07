#!/usr/bin/env bash
#SBATCH --job-name=gphi-gate-v1
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_gate_feasibility_v1/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_gate_feasibility_v1/slurm-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.08
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
"$PY" diagnostics/gphi_gate_feasibility_v1/run_gate.py

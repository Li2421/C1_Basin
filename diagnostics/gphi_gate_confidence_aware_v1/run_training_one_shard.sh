#!/usr/bin/env bash
#SBATCH --job-name=gate-conf-train
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:45:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_gate_confidence_aware_v1/slurm-train-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_gate_confidence_aware_v1/slurm-train-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OUT="$ROOT/diagnostics/gphi_gate_confidence_aware_v1"
export PYTHONPATH="$ROOT" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.08
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
"$PY" "$OUT/finalize_and_train.py"

#!/bin/bash
#SBATCH --job-name=controller_q_ff
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=07:30:00
#SBATCH --output=diagnostics/controller_q_diversity_v1/logs/%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -u -m diagnostics.controller_q_diversity_v1.experiment worker --batch "$1" --index "${SLURM_ARRAY_TASK_ID:-0}" --workers "${2:-12}"

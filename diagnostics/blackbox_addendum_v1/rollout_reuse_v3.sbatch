#!/bin/bash
#SBATCH --job-name=r37_roll
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/blackbox_addendum_v1/roll_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/blackbox_addendum_v1/worker_reuse_v3.py "$1" "$SLURM_ARRAY_TASK_ID"

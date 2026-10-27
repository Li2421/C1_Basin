#!/bin/bash
#SBATCH --job-name=baseline2-paper
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-15%16
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/baseline2_literature_scenes_v1/logs/paper_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu
unset XLA_FLAGS
exec diagnostics/baseline2_literature_scenes_v1/.venv-ext/bin/python -B diagnostics/baseline2_literature_scenes_v1/worker_paper.py "$SLURM_ARRAY_TASK_ID" 16

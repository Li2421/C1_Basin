#!/bin/bash
#SBATCH --job-name=baseline2-secondary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/baseline2_literature_scenes_v1/logs/secondary_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec diagnostics/baseline2_literature_scenes_v1/.venv-ext/bin/python -B diagnostics/baseline2_literature_scenes_v1/worker_secondary.py "$BASELINE2_STAGE"

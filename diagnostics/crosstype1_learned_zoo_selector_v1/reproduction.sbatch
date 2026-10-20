#!/bin/bash
#SBATCH --job-name=crosstype1_reproduction
#SBATCH --partition=cpu
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/reproduction_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/reproduction.py run "$SLURM_ARRAY_TASK_ID"

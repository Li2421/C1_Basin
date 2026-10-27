#!/bin/bash
#SBATCH --job-name=xp2_worker
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/zhihan/research/Basin_C1
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/cross_scene_pipeline_v2/worker_%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/cross_scene_pipeline_v2/worker.py "$XP2_STAGE"

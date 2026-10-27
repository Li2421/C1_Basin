#!/bin/bash
#SBATCH --job-name=lit_resume_stats
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_stage2_collision_fix_v1/statistics_%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
SCENES=(lit_narrow_passage lit_crowded_random lit_circle_five)
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/baseline_literature_scenes_stage2_collision_fix_v1/part1_statistics.py "${SCENES[$SLURM_ARRAY_TASK_ID]}"

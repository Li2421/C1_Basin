#!/bin/bash
#SBATCH --job-name=stage2_fix_units
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_stage2_collision_fix_v1/unit_%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
TASK_SCENES=(lit_narrow_passage lit_crowded_random lit_circle_five)
TASK_SCENE=${TASK_SCENES[$((SLURM_ARRAY_TASK_ID / 10))]}
TASK_CHUNK=$((SLURM_ARRAY_TASK_ID % 10))
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/baseline_literature_scenes_stage2_collision_fix_v1/snapshot_gate.py "$TASK_SCENE" "$TASK_CHUNK"

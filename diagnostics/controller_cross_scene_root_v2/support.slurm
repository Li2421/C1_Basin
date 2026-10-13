#!/bin/bash
#SBATCH --job-name=q_root_support
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --array=0-1
#SBATCH --output=diagnostics/controller_cross_scene_root_v2/support_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
folds=(toy ring)
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_cross_scene_root_v2.support_audit "${folds[$SLURM_ARRAY_TASK_ID]}"

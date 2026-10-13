#!/bin/bash
#SBATCH --job-name=q_root_four_fit
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:40:00
#SBATCH --array=0-44%24
#SBATCH --output=diagnostics/controller_four_scene_mechanism_v1/fit_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_four_scene_mechanism_v1.learn fit --index "$SLURM_ARRAY_TASK_ID"

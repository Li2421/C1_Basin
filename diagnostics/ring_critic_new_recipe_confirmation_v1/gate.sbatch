#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r17_gate
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_critic_new_recipe_confirmation_v1/gate_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_critic_new_recipe_confirmation_v1/run.py gate "$SLURM_ARRAY_TASK_ID" 8

#!/bin/bash
#SBATCH --job-name=crosstype2_conservative_ppo
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype2_mixed_zoo_selector_v1/conservative_ppo_%A_%a.log
#SBATCH --gres=shard:1
#SBATCH --array=0-1%2
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype2_mixed_zoo_selector_v1/conservative_training.py run "$SLURM_ARRAY_TASK_ID"

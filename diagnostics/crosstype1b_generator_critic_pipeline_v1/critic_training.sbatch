#!/bin/bash
#SBATCH --job-name=crosstype1b_critic_training
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype1b_generator_critic_pipeline_v1/critic_training_%A_%a.log
#SBATCH --gres=shard:1
#SBATCH --array=0-1%2
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype1b_generator_critic_pipeline_v1/train_critic.py run $SLURM_ARRAY_TASK_ID

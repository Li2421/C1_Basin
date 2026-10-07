#!/bin/bash
#SBATCH --job-name=ortho3-modefree-hard
#SBATCH --partition=gpu
#SBATCH --array=0-5%6
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/logs/%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/logs/%A_%a.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/run_rollouts.py \
  --shard "${SLURM_ARRAY_TASK_ID}" --shards 6 \
  --cache-preflight diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/cache_preflight_resume.json

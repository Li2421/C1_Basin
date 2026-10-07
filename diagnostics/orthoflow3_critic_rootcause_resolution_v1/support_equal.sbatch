#!/bin/bash
#SBATCH --job-name=c1-support-equal
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-29%6
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/support_equal_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/support_equal_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export JAX_PLATFORMS=cuda,cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.support_train --index "$SLURM_ARRAY_TASK_ID" --pair-equal

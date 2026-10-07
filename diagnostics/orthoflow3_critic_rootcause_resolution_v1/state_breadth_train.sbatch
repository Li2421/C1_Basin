#!/bin/bash
#SBATCH --job-name=c1-breadth-critic
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-17%3
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/breadth_train_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/breadth_train_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.state_breadth_train train --index "$SLURM_ARRAY_TASK_ID"

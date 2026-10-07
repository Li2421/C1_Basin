#!/bin/bash
#SBATCH --job-name=c1-motion-newflow
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%1
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_confirmation_flow_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_confirmation_flow_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.motion_confirmation flow --replicate "$SLURM_ARRAY_TASK_ID"

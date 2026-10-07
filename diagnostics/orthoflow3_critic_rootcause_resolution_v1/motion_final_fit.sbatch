#!/bin/bash
#SBATCH --job-name=c1-motion-final
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:15:00
#SBATCH --array=0-2%1
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_final_fit_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_final_fit_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.motion_final_fit train --index "$SLURM_ARRAY_TASK_ID"

#!/bin/bash
#SBATCH --job-name=c1-physical-kernel
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-2%3
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_kernel_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_kernel_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.motion_kernel test
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.motion_kernel train --fold "$SLURM_ARRAY_TASK_ID"

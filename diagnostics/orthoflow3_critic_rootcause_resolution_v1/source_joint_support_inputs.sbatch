#!/bin/bash
#SBATCH --job-name=c1-source-joint-support-inputs
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:45:00
#SBATCH --array=0-4%5
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/joint_support_inputs_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/joint_support_inputs_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.source_joint_support_inputs features --worker "$SLURM_ARRAY_TASK_ID"

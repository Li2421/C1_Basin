#!/bin/bash
#SBATCH --job-name=c1-factorial-C
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_state_residual_factorial_v1/features_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_state_residual_factorial_v1/features_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export JAX_PLATFORMS=cuda,cpu
controllers=(base alt second)
controller=${controllers[$((SLURM_ARRAY_TASK_ID / 2))]}
shard=$((SLURM_ARRAY_TASK_ID % 2))
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_controller_state_residual_factorial_v1.context_features --controller "$controller" --shard "$shard" --shards 2

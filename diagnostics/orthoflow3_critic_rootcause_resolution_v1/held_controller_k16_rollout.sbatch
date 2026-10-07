#!/bin/bash
#SBATCH --job-name=c1-k16-native-Q16
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --gres=shard:1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/k16_rollout_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/k16_rollout_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.held_controller_k16_runtime run --index "$1" --batch "$2" --worker "$SLURM_ARRAY_TASK_ID"

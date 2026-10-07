#!/bin/bash
#SBATCH --job-name=c1-diversity-test
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/diversity_test_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/diversity_test_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
worker_ids=()
for logical_offset in 0 1 2; do
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.diversity_confirmation run --index "$((SLURM_ARRAY_TASK_ID * 3 + logical_offset))" &
  worker_ids+=("$!")
done
for worker_pid in "${worker_ids[@]}"; do wait "$worker_pid"; done

#!/bin/bash
#SBATCH --job-name=c1-diversity-source
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/diversity_rollouts_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/diversity_rollouts_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
run_logical_worker() {
  for controller_name in extra_a extra_b; do
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.controller_diversity run --controller "$controller_name" --index "$1"
  done
}
run_logical_worker "$((SLURM_ARRAY_TASK_ID * 2))" &
worker_a=$!
run_logical_worker "$((SLURM_ARRAY_TASK_ID * 2 + 1))" &
worker_b=$!
wait "$worker_a"
wait "$worker_b"

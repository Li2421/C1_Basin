#!/bin/bash
#SBATCH --job-name=c1-function-support
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=3
#SBATCH --mem=9G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/function_rollouts_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/function_rollouts_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu
run_worker() {
  for controller_index in 0 1 2 3 4 5 6 7; do
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.function_support run --controller "function_$controller_index" --index "$1"
  done
}
run_worker "$((SLURM_ARRAY_TASK_ID * 3))" &
worker_a=$!
run_worker "$((SLURM_ARRAY_TASK_ID * 3 + 1))" &
worker_b=$!
run_worker "$((SLURM_ARRAY_TASK_ID * 3 + 2))" &
worker_c=$!
wait "$worker_a"
wait "$worker_b"
wait "$worker_c"

#!/bin/bash
#SBATCH --job-name=c1-goal-target-Q16
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=9G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%6
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_confirm_rollouts_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_confirm_rollouts_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu
run_worker() {
  for replica_index in 0 1; do
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.goal_confirmation run --index "$1" --replicate "$replica_index"
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

#!/bin/bash
#SBATCH --job-name=c1-goal-velocity-input
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --array=0-3%2
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_velocity_inputs_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_velocity_inputs_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu
task_pids=()
for task_offset in 0 1 2; do
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.goal_velocity_response build --index "$((SLURM_ARRAY_TASK_ID*3+task_offset))" &
  task_pids+=("$!")
done
for task_pid in "${task_pids[@]}"; do wait "$task_pid"; done

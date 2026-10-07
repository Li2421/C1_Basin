#!/bin/bash
#SBATCH --job-name=c1-goal-target-input
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_confirm_inputs_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_confirm_inputs_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu
task_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.goal_confirmation physical --replicate "$SLURM_ARRAY_TASK_ID"
for context_index in 0 1 2 3 4; do
  "$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.goal_confirmation features --replicate "$SLURM_ARRAY_TASK_ID" --index "$context_index"
  "$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.goal_confirmation goal --replicate "$SLURM_ARRAY_TASK_ID" --index "$context_index"
done

#!/bin/bash
#SBATCH --job-name=c1-raw-held-setup
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:30:00
#SBATCH --array=0-1%2
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/raw_setup_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/raw_setup_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
for action in bridge flow prepare physical inputs preflight predict; do
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.raw_bypass_confirmation "$action" --replicate "$SLURM_ARRAY_TASK_ID"
done
target=$((88136+SLURM_ARRAY_TASK_ID))
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m shared_rollout_db.plan --manifest "diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_independent_confirmation_${target}/planned_rollouts.json" --output "diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_independent_confirmation_${target}/cache_preflight.json"

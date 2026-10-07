#!/bin/bash
#SBATCH --job-name=c1-function-test-plan
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/function_confirm_prepare_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/function_confirm_prepare_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
task_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
for replica_index in 0 1; do
  "$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.function_confirmation prepare --replicate "$replica_index"
  if [ "$replica_index" -eq 0 ]; then
    task_out=diagnostics/orthoflow3_critic_rootcause_resolution_v1/controller_function_confirmation
  else
    task_out=diagnostics/orthoflow3_critic_rootcause_resolution_v1/controller_function_confirmation_b
  fi
  "$task_python" -m shared_rollout_db.plan --manifest "$task_out/planned_rollouts.json" --output "$task_out/cache_preflight.json"
  "$task_python" -m shared_rollout_db.plan --manifest "$task_out/planned_held.json" --output "$task_out/cache_preflight_held.json"
  "$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.function_confirmation validate --replicate "$replica_index"
  "$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.function_confirmation filter --replicate "$replica_index"
done

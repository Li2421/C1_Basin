#!/bin/bash
#SBATCH --job-name=c1-k16-freeze-preflight
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/k16_preflight_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/k16_preflight_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
for replica in 0 1; do
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.held_controller_k16_runtime assemble --index "$replica"
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.held_controller_k16_evaluate predict --index "$replica"
    task_folder="diagnostics/orthoflow3_critic_rootcause_resolution_v1/held_controller_k16_v1/target_$((88138 + replica))/batch0"
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m shared_rollout_db.plan --manifest "$task_folder/planned_rollouts.json" --output "$task_folder/cache_preflight.json"
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m shared_rollout_db.plan --manifest "$task_folder/planned_held.json" --output "$task_folder/cache_preflight_held.json"
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.held_controller_k16_runtime validate --index "$replica" --batch 0
done

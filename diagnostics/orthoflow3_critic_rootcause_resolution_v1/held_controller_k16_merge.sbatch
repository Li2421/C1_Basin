#!/bin/bash
#SBATCH --job-name=c1-k16-atomic-postflight
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/k16_merge_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/k16_merge_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.held_controller_k16_runtime merge --index "$1" --batch "$2"
task_folder="diagnostics/orthoflow3_critic_rootcause_resolution_v1/held_controller_k16_v1/target_$((88138 + $1))/batch$2"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m shared_rollout_db.plan --manifest "$task_folder/planned_rollouts.json" --output "$task_folder/cache_postflight.json"
